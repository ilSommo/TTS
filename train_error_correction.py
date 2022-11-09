import json
import os
import torch
from argparse import ArgumentParser

import evaluate
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    pipeline,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", default="gsarti/it5-base")
    parser.add_argument("--train-filepath", default="err_corr.json")
    parser.add_argument("--seed", default=42)
    parser.add_argument("--wandb", action="store_true", default=False)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "false" if args.wandb else "true"

    with open(args.train_filepath) as f:
        err_corr = json.load(f)

    predicts, targets = [x["predict"] for x in err_corr["train"].values()], [
        x["value"] for x in err_corr["train"].values()
    ]

    print("Loading dataset...")
    dataset = Dataset.from_dict({"predicts": predicts, "targets": targets})
    dataset = dataset.train_test_split()

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    generator = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer)

    def tokenize_function(examples):
        return {
            "input_ids": tokenizer(
                examples["predicts"],
                padding=False,
                max_length=300,
                truncation=True,
                # return_tensors="pt",
            ).input_ids,
            "labels": tokenizer(
                examples["targets"],
                padding=False,
                max_length=300,
                truncation=True,
                # return_tensors="pt",
            ).input_ids,
        }

    print("Start tokenization...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    metric = evaluate.load("wer")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        return preds, labels

    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits[0], dim=-1)
        return pred_ids, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            preds = np.argmax(preds, axis=-1)
        print(type(preds), type(labels))
        print(preds.shape, labels.shape)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if True:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"wer": result}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        # max_length=300,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="test_trainer_phonemized", evaluation_strategy="steps", eval_steps=5000, save_steps=5000
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()
    trainer.save_model('test_trainer_phonemized/final')
