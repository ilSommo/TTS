import json
from argparse import ArgumentParser
from copy import copy
from nemo.collections.asr.metrics.wer import word_error_rate as nemo_wer
from torchmetrics.functional import word_error_rate as torch_wer


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
        predicts = [x["predict"] for x in paired_dataset.values()]
        values = [x["value"] for x in paired_dataset.values()]
        m = compute_metrics(predicts, values)
        metrics["torch_wer"][key] = m["torch_wer"]
        metrics["nemo_wer"][key] = m["nemo_wer"]
        metrics["nemo_cer"][key] = m["nemo_cer"]
    return metrics

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--k", type=int)
    parser.add_argument("--filepath")
    parser.add_argument("--metric")
    parser.add_argument("--keys", default=None)
    args = parser.parse_args()

    with open(args.filepath) as f:
        datasets = json.load(f)

    top = {}
    l = lambda d1, d2: d1.update(d2) or d1
    for k, samples in datasets.items():
        if args.keys:
            keys_split = args.keys.split(",")
            keys = [l(x, {"key": k}) for k, x in samples.items() if k in keys_split]

            top[k] = {"keys": keys}
        else:
            samples = {k: l(x, {"value": x["value"].lower()}) for k, x in samples.items()}
            datasets[k] = samples
            new_samples = [l(x, {"key": k}) for k, x in samples.items()]
            new_samples = [l(x,compute_metrics([x["predict"]],[x["value"]])) for x in new_samples]
            positive = sorted(new_samples, reverse=False, key=lambda s: s[args.metric])
            negative = sorted(new_samples, reverse=True, key=lambda s: s[args.metric])

            top[k] = {"positive": positive[: args.k], "negative": negative[: args.k]}

        line_length = 90
        for type in top[k].keys():
            print("=" * line_length)
            print(f"{k} dataset - {type}")
            for idx, sample in enumerate(top[k][type]):
                metrics = ", ".join(
                    [
                        f"{m}: {sample[m]}"
                        for m in sample.keys()
                        if not m in ["predict", "value", "key"]
                    ]
                )
                print("=" * line_length)
                print(f"{idx+1}° - {metrics}")
                print("-" * line_length)
                print("key: " + sample["key"])
                print("-" * line_length)
                print("Predict: " + sample["predict"])
                print("-" * line_length)
                print("Target: " + sample["value"])
                print("=" * line_length)
                print("\n")
            print("\n")
    
    # print(compute_metrics_dataset(datasets))

