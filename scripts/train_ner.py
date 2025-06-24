import yaml
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import torch

def read_config(path="config/training_config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return tokenizer, model

if __name__ == "__main__":
    cfg = read_config()
    tokenizer, model = get_model_and_tokenizer(cfg['model_name'])

    dataset = load_dataset("json", data_files={"train": "data/labels/train.json", "validation": "data/labels/val.json"})

    def tokenize(batch):
        return tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)

    dataset = dataset.map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir=cfg['output_dir'],
        evaluation_strategy="epoch",
        learning_rate=cfg['learning_rate'],
        per_device_train_batch_size=cfg['train_batch_size'],
        per_device_eval_batch_size=cfg['eval_batch_size'],
        num_train_epochs=cfg['num_train_epochs'],
        weight_decay=0.01,
        logging_dir=cfg['logging_dir'],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()
