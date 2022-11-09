import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate as nemo_wer
from torchmetrics.functional import word_error_rate as torch_wer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from reverse_dp import reverse_dp


def key_from_path(path):
    portions = path.split("/")
    return portions[-3] + "_" + portions[-2] + "_" + portions[-1].split(".")[0]


def get_values(audios):
    return {key_from_path(j["audio_filepath"]): j["text"] for j in audios}


def get_quartz_predictions(model, audio_files):
    return {
        # TOOD: È una comprehension
        # key_from_path(fname): transcription
        key_from_path(fname): reverse_dp(transcription)
        for fname, transcription in zip(
            audio_files, model.transcribe(paths2audio_files=audio_files)
        )
    }


def get_seq2seq_predictions(model, tokenizer, texts):
    preds = {}
    for key, text in tqdm(texts.items()):
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        outputs = model.generate(input_ids)
        preds[key] = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return preds


def output_eval_dictionary(predictions, values, with_metrics=False):
    ds = {}
    preds = []
    targets = []
    for key, predict in predictions.items():
        ds[key] = {"predict": predict, "value": values[key]}
        if with_metrics:
            metrics = compute_metrics([predict], [values[key]])
            ds[key].update(metrics)
        preds.append(predict)
        targets.append(values[key])
    return ds, preds, targets


def read_manifest(json_path):
    with open(json_path) as f:
        return [json.loads(line) for line in f]


def read_target_preds(json_path):
    with open(json_path) as f:
        return json.load(f)


def eval(base_dir, json_full_paths, model_type):
    paired_datasets = {}
    if model_type == "nemo":
        model = (
            nemo_asr.models.EncDecCTCModel.restore_from(model_full_path).eval().cuda()
        )

        for key, json_full_path in json_full_paths.items():
            audios = read_manifest(json_full_path)
            preds = get_quartz_predictions(
                model, [os.path.join(base_dir, j["audio_filepath"]) for j in audios]
            )
            targets = get_values(audios)
            ds, preds, targets = output_eval_dictionary(
                preds, targets, with_metrics=True
            )
            paired_datasets[key] = {"ds": ds, "preds": preds, "targets": targets}
    elif model_type == "seq2seq":
        json_path = list(json_full_paths.values())[0]  # Only one json for every dataset
        tokenizer = AutoTokenizer.from_pretrained(model_full_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_full_path, max_length=300).cuda()

        datasets = read_target_preds(json_path)

        for key, dataset in datasets.items():
            targets = {key: el["value"] for key, el in dataset.items()}
            texts = {key: el["predict"] for key, el in dataset.items()}
            preds = get_seq2seq_predictions(model, tokenizer, texts)
            ds, preds, targets = output_eval_dictionary(
                preds, targets, with_metrics=True
            )
            paired_datasets[key] = {"ds": ds, "preds": preds, "targets": targets}
    else:
        raise argparse.ArgumentError(f"Unknown model type {args.model_type}")
    return paired_datasets


def compute_metrics(preds, targets):
    return {
        "torch_wer": torch_wer(preds=preds, target=targets).item(),
        "nemo_wer": nemo_wer(hypotheses=preds, references=targets),
        "nemo_cer": nemo_wer(
            hypotheses=preds,
            references=targets,
            use_cer=True,
        ),
    }


def compute_metrics_dataset(paired_datasets):
    metrics = {"torch_wer": {}, "nemo_wer": {}, "nemo_cer": {}}
    for key, paired_dataset in paired_datasets.items():
        m = compute_metrics(paired_dataset["preds"], paired_dataset["targets"])
        metrics["torch_wer"][key] = m["torch_wer"]
        metrics["nemo_wer"][key] = m["nemo_wer"]
        metrics["nemo_cer"][key] = m["nemo_cer"]
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir")
    parser.add_argument("--model-path")
    parser.add_argument("--train-json-path")
    parser.add_argument("--valid-json-path")
    parser.add_argument("--test-json-path")
    parser.add_argument("--model-type", choices=["nemo", "seq2seq"])
    parser.add_argument(
        "--dump-target-predict-pairs", action="store_true", default=False
    )
    args = parser.parse_args()

    json_paths = {}
    if args.train_json_path:
        json_paths["train"] = args.train_json_path
    if args.valid_json_path:
        json_paths["valid"] = args.valid_json_path
    if args.test_json_path:
        json_paths["test"] = args.test_json_path

    model_full_path = os.path.join(args.base_dir, args.model_path)
    json_full_paths = {
        key: os.path.join(args.base_dir, json_path)
        for key, json_path in json_paths.items()
    }

    paired_datasets = eval(args.base_dir, json_full_paths, args.model_type)

    unique_name_prefix = os.path.join(
        args.base_dir,
        args.model_path.replace(".", "_")
        + "_"
        + datetime.now().strftime("%y%m%d%H%M%S"),
    )
    if args.dump_target_predict_pairs:
        with open(unique_name_prefix + "_target_predicts.json", "w") as outfile:
            json.dump(
                {
                    key: paired_dataset["ds"]
                    for key, paired_dataset in paired_datasets.items()
                },
                outfile,
            )

    metrics = compute_metrics_dataset(paired_datasets)
    with open(unique_name_prefix + "_eval.json", "w") as outfile:
        json.dump(metrics, outfile)
