import os
import sys
from time import time

import torch
from datasets import load_dataset
import evaluate
import numpy as np
from transformers import AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
# import wandb
import pathlib

accuracy_metric = evaluate.load("accuracy")


def compute_metrics(p):
    return {"accuracy": accuracy_metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)['accuracy']}

# wandb.login(key='1106fae45f2bf456d9ae3c0014900989d20a2863')
# os.environ["WANDB_PROJECT"] = "anlp_ex1"
# os.environ["WANDB_LOG_MODEL"] = "true"
# os.environ["WANDB_WATCH"] = "false"

parent_dir = str(pathlib.Path(__file__).parent.resolve())

seeds_num: int = int(sys.argv[1])
train_samps: int = int(sys.argv[2])
val_samps: int = int(sys.argv[3])
test_samps: int = int(sys.argv[4])

sst2 = load_dataset("sst2")
if train_samps == -1:
    sst2_train = sst2["train"]
else:
    sst2_train = sst2["train"].select(range(train_samps))
if val_samps == -1:
    sst2_val = sst2["validation"]
else:
    sst2_val = sst2["validation"].select(range(val_samps))
if test_samps == -1:
    sst2_test = sst2["test"]
else:
    sst2_test = sst2["test"].select(range(test_samps))

bert_accs_map = {}
roberta_accs_map = {}
electra_accs_map = {}
models_accs_map = {}

train_time = 0

bert_conf = AutoConfig.from_pretrained("bert-base-uncased")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("##### BERT #####")

for i in range(seeds_num):
    # run = wandb.init(project="ANLPex1", name="bert-base-uncased_" + str(i))
    bert = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=bert_conf)
    bert_hyperparams = TrainingArguments(output_dir=parent_dir + "/bert-base-uncased_" + str(i), seed=i, save_strategy="epoch", save_total_limit=1)
    sst2_train_bert = sst2_train.map(lambda x: bert_tokenizer(x['sentence'], truncation=True, padding=True, return_tensors='pt'), batched=True)
    sst2_val_bert = sst2_val.map(lambda x: bert_tokenizer(x['sentence'], truncation=True, padding=True, return_tensors='pt'), batched=True)
    bert_trainer = Trainer(model=bert, args=bert_hyperparams, train_dataset=sst2_train_bert, eval_dataset=sst2_val_bert, tokenizer=bert_tokenizer, compute_metrics=compute_metrics)
    train_start = time()
    bert_train_res = bert_trainer.train()
    train_time += time() - train_start
    metrics = bert_trainer.evaluate(eval_dataset=sst2_val_bert)
    print(f"seed = {i}, metrics = {metrics}")
    bert_accs_map[(bert, bert_trainer, bert_tokenizer)] = metrics["eval_accuracy"]
    # run.finish()

print(f"BERT time {train_time}")
bert_mean = np.mean(list(bert_accs_map.values()))
bert_std = np.std(list(bert_accs_map.values()))
print(bert_mean)
print(bert_std)

roberta_conf = AutoConfig.from_pretrained("roberta-base")
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
print("##### ROBERTA #####")

for i in range(seeds_num):
    # run = wandb.init(project="ANLPex1", name="roberta-base_" + str(i))
    roberta = AutoModelForSequenceClassification.from_pretrained("roberta-base", config=roberta_conf)
    roberta_hyperparams = TrainingArguments(output_dir=parent_dir + "/roberta-base_" + str(i), seed=i, save_strategy="epoch", save_total_limit=1)
    sst2_train_roberta = sst2_train.map(lambda x: roberta_tokenizer(x['sentence'], truncation=True, padding=True, return_tensors='pt'), batched=True)
    sst2_val_roberta = sst2_val.map(lambda x: roberta_tokenizer(x['sentence'], truncation=True, padding=True, return_tensors='pt'), batched=True)
    roberta_trainer = Trainer(model=roberta, args=roberta_hyperparams, train_dataset=sst2_train_roberta, eval_dataset=sst2_val_roberta, tokenizer=roberta_tokenizer, compute_metrics=compute_metrics)
    train_start = time()
    roberta_train_res = roberta_trainer.train()
    train_time += time() - train_start
    metrics = roberta_trainer.evaluate(eval_dataset=sst2_val_roberta)
    print(f"seed = {i}, metrics = {metrics}")
    roberta_accs_map[(roberta, roberta_trainer, roberta_tokenizer)] = metrics["eval_accuracy"]
    # run.finish()

print(f"ROBERTA time {train_time}")
roberta_mean = np.mean(list(roberta_accs_map.values()))
roberta_std = np.std(list(roberta_accs_map.values()))
print(roberta_mean)
print(roberta_std)

electra_conf = AutoConfig.from_pretrained("google/electra-base-generator")
electra_tokenizer = AutoTokenizer.from_pretrained("google/electra-base-generator")
print("##### ELECTRA #####")

for i in range(seeds_num):
    # run = wandb.init(project="ANLPex1", name="google-electra-base-generator_" + str(i))
    electra = AutoModelForSequenceClassification.from_pretrained("google/electra-base-generator", config=electra_conf)
    electra_hyperparams = TrainingArguments(output_dir=parent_dir + "/google-electra-base-generator_" + str(i), seed=i, save_strategy="epoch", save_total_limit=1)
    sst2_train_electra = sst2_train.map(lambda x: electra_tokenizer(x['sentence'], truncation=True, padding=True, return_tensors='pt'), batched=True)
    sst2_val_electra = sst2_val.map(lambda x: electra_tokenizer(x['sentence'], truncation=True, padding=True, return_tensors='pt'), batched=True)
    electra_trainer = Trainer(model=electra, args=electra_hyperparams, train_dataset=sst2_train_electra, eval_dataset=sst2_val_electra, tokenizer=electra_tokenizer, compute_metrics=compute_metrics)
    train_start = time()
    electra_train_res = electra_trainer.train()
    train_time += time() - train_start
    metrics = electra_trainer.evaluate(eval_dataset=sst2_val_electra)
    print(f"seed = {i}, metrics = {metrics}")
    electra_accs_map[(electra, electra_trainer, electra_tokenizer)] = metrics["eval_accuracy"]
    # run.finish()

print(f"ELECTRA time {train_time}")
electra_mean = np.mean(list(electra_accs_map.values()))
electra_std = np.std(list(electra_accs_map.values()))
print(electra_mean)
print(electra_std)

best_idx = np.argmax([bert_mean, roberta_mean, electra_mean])
if best_idx == 0:
    best_accs_map = bert_accs_map
elif best_idx == 1:
    best_accs_map = roberta_accs_map
else:
    best_accs_map = electra_accs_map
model, model_trainer, model_tokenizer = max(best_accs_map, key=best_accs_map.get)

model.eval()
with open("res.txt", 'w') as res, open("predictions.txt", 'w') as predictions:
    res.write(f"bert-base-uncased,{bert_mean} +- {bert_std}\n")
    res.write(f"roberta-base,{roberta_mean} +- {roberta_std}\n")
    res.write(f"google/electra-base-generator,{electra_mean} +- {electra_std}\n")
    res.write("----\n")
    res.write(f"train time,{train_time}\n")

    pred_start = time()
    for samp in sst2_test:
        preds = model_trainer.prediction_step(model, model_tokenizer(samp['sentence'], truncation=True, return_tensors='pt'), prediction_loss_only=False)
        pred = torch.argmax(preds[1]).item()
        predictions.write(f"{samp['sentence']}###{pred}\n")
    pred_time = time() - pred_start

    res.write(f"predict time,{pred_time}")

# wandb.finish()
