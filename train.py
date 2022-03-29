import argparse
import logging

from datasets import load_dataset, load_metric
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer


def main(DATASET_NAME, model_checkpoint, batch_size):
    logging.basicConfig(
        filename=str(DATASET_NAME) + "-train-logs.log",
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.DEBUG,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/train-{}.json".format(DATASET_NAME),
            "val": "data/val.json",
            "test": "data/test.json",
        },
        field="data",
    )

    metric = load_metric("accuracy")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    def preprocess_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=256, padding="max_length"
        )

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    model_name = model_checkpoint.split("/")[-1]

    args = TrainingArguments(
        f"{model_name}-finetuned-cc-{DATASET_NAME}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model_checkpoint", default="distilbert-base-uncased")
    parser.add_argument("--batch_size", default=64)
    args = parser.parse_args()

    main(
        DATASET_NAME=args.dataset_name,
        model_checkpoint=args.model_checkpoint,
        batch_size=args.batch_size,
    )
