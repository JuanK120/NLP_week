import pandas as pd
import numpy as np
from readcsv import get_dataframe
from datasets import Dataset
from transformers import AutoTokenizer, ModernBertForSequenceClassification, TrainingArguments, Trainer
import evaluate

def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding='max_length')

accuracy = evaluate.load("accuracy")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return accuracy.compute(predictions=preds, references=p.label_ids)

print(TrainingArguments.__module__)

def get_train_args(results_route):
    return TrainingArguments(
            output_dir=results_route,
            eval_strategy="epoch", 
            num_train_epochs=3,
            weight_decay=0.01,
            use_cpu=False,
            auto_find_batch_size= True,
        )


dataset_location = './dataset/olid-training-v1.0.tsv' 

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model_a = ModernBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model_b = ModernBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model_c = ModernBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

_,subset_a,subset_b,subset_c=get_dataframe(dataset_location)

dataset_dict_a = Dataset.from_pandas(subset_a.rename(columns={'tweet': 'text', 'subtask_a': 'label'}))
label_map_a = {'OFF': 0, 'NOT': 1}
dataset_dict_a = dataset_dict_a.map(lambda x: {'label': label_map_a[x['label']]})


dataset_dict_b = Dataset.from_pandas(subset_b.rename(columns={'tweet': 'text', 'subtask_b': 'label'}))
label_map_b = {'UNT': 0, 'TIN': 1}
dataset_dict_b = dataset_dict_b.map(lambda x: {'label': label_map_b[x['label']]})

dataset_dict_c = Dataset.from_pandas(subset_c.rename(columns={'tweet': 'text', 'subtask_c': 'label'}))
label_map_c = {'IND': 0, 'GRP': 1, 'OTH': 2}
dataset_dict_c = dataset_dict_c.map(lambda x: {'label': label_map_c[x['label']]})

tokenized_dataset_a = dataset_dict_a.map(tokenize, batched=True)
tokenized_dataset_b = dataset_dict_b.map(tokenize, batched=True)
tokenized_dataset_c = dataset_dict_c.map(tokenize, batched=True)


print(tokenized_dataset_a,tokenized_dataset_b,tokenized_dataset_c)

trainer_a = Trainer(
    model=model_a,
    args=get_train_args("./results/a"),
    train_dataset=tokenized_dataset_a.train_test_split(test_size=0.2)["train"],
    eval_dataset=tokenized_dataset_a.train_test_split(test_size=0.2)["test"],
    compute_metrics=compute_metrics,
)

trainer_b = Trainer(
    model=model_b,
    args=get_train_args("./results/b"),
    train_dataset=tokenized_dataset_b.train_test_split(test_size=0.2)["train"],
    eval_dataset=tokenized_dataset_b.train_test_split(test_size=0.2)["test"],
    compute_metrics=compute_metrics,
)

trainer_c = Trainer(
    model=model_c,
    args=get_train_args("./results/c"),
    train_dataset=tokenized_dataset_c.train_test_split(test_size=0.2)["train"],
    eval_dataset=tokenized_dataset_c.train_test_split(test_size=0.2)["test"],
    compute_metrics=compute_metrics,
)

trainer_a.train()
trainer_b.train()
trainer_c.train()


#########



