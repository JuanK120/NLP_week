import os
import pandas as pd
import numpy as np
from readcsv import get_test_dataframe, get_test_labels
from datasets import Dataset
from transformers import AutoTokenizer, ModernBertForSequenceClassification, TrainingArguments, Trainer
import torch
import evaluate 

# Load model and tokenizer
dataset_location = './dataset/'

model_a = ModernBertForSequenceClassification.from_pretrained("./results/a/checkpoint-3972")
model_a.eval()
model_b = ModernBertForSequenceClassification.from_pretrained("./results/b/checkpoint-1320")
model_b.eval()
model_c = ModernBertForSequenceClassification.from_pretrained("./results/c/checkpoint-1164")
model_c.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# Load test sets and labels
test_set_a = get_test_dataframe(dataset_location + 'testset-levela.tsv')
label_map_a = {'OFF': 0, 'NOT': 1}
test_labels_a = get_test_labels(dataset_location + 'labels-levela.csv')
test_labels_a['label'] = test_labels_a['label'].apply(lambda x: label_map_a[x])
count_OFF = (test_labels_a['label'] == 0).sum()
count_NOT = (test_labels_a['label'] == 1).sum()

test_set_b = get_test_dataframe(dataset_location+'testset-levelb.tsv')
label_map_b = {'UNT': 0, 'TIN': 1}
test_labels_b = get_test_labels(dataset_location+'labels-levelb.csv')
test_labels_b['label'] = test_labels_b['label'].apply(lambda x: label_map_b[x])
count_UNT = (test_labels_b['label'] == 0).sum()
count_TIN = (test_labels_b['label'] == 1).sum()


test_set_c = get_test_dataframe(dataset_location+'testset-levelc.tsv')
label_map_c = {'IND': 0, 'GRP': 1, 'OTH': 2}
test_labels_c = get_test_labels(dataset_location+'labels-levelc.csv') 
test_labels_c['label'] = test_labels_c['label'].apply(lambda x: label_map_c[x])
count_IND = (test_labels_c['label'] == 0).sum()
count_GRP = (test_labels_c['label'] == 1).sum()
count_OTH = (test_labels_c['label'] == 2).sum()


# Evaluate model on test set A 'OFF': 0, 'NOT': 1
number_correct_answers_a = 0

true_positives_OFF = 0
false_positives_OFF = 0
false_negatives_OFF = 0

true_positives_NOT = 0
false_positives_NOT = 0
false_negatives_NOT = 0

len_set_a = len(test_set_a)

wrong_lines = ''

for i in range(len_set_a):
    text = test_set_a.iloc[i]['tweet']
    true_label = test_labels_a.iloc[i]['label']

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Predict
    with torch.no_grad():
        outputs = model_a(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()

    # Count correct predictions
    if predicted_class_id == true_label:
        number_correct_answers_a += 1

        if predicted_class_id == 0:
            true_positives_OFF += 1
        elif predicted_class_id == 1:
            true_positives_NOT += 1
    else:
        if predicted_class_id == 0:
            false_positives_OFF += 1
        elif predicted_class_id == 1:
            false_positives_NOT += 1

        if true_label == 0:
            false_negatives_OFF += 1
        elif true_label == 1:
            false_negatives_NOT += 1

        wrong_lines += f"Tweet: {text}\n"
        wrong_lines += f"True label: {true_label}\n"
        wrong_lines += f"Predicted label: {predicted_class_id}\n"
        wrong_lines += "-" * 40 + "\n"
    
    os.makedirs('./tests', exist_ok=True)
    with open('./tests/logs_a.txt', 'w', encoding='utf-8') as f:
        f.writelines(wrong_lines)



accuracy_a = number_correct_answers_a / len_set_a
precision_OFF = true_positives_OFF / (true_positives_OFF + false_positives_OFF + 1e-6)
precision_NOT = true_positives_NOT / (true_positives_NOT + false_positives_NOT + 1e-6)
recall_OFF = true_positives_OFF / (true_positives_OFF + false_negatives_OFF + 1e-6)
recall_NOT = true_positives_NOT / (true_positives_NOT + false_negatives_NOT + 1e-6)
f1_OFF = 2 * precision_OFF * recall_OFF / (precision_OFF + recall_OFF + 1e-6)
f1_NOT = 2 * precision_NOT * recall_NOT / (precision_NOT + recall_NOT + 1e-6)

 
print('-'*40 + '\n')
print(f"Number of OFF items: {count_OFF}")
print(f"Number of NOT items: {count_NOT}")
print(f"Accuracy on Level A: {accuracy_a:.4f}")
print(f"Precision_OFF: {precision_OFF:.4f}")
print(f"Recall_OFF:    {recall_OFF:.4f}")
print(f"F1_OFF:        {f1_OFF:.4f}")
print(f"Precision_NOT: {precision_NOT:.4f}")
print(f"Recall_NOT:    {recall_NOT:.4f}")
print(f"F1_NOT:        {f1_NOT:.4f}\n")


# Evaluate model on test set B 'UNT': 0, 'TIN': 1
number_correct_answers_b = 0

true_positives_UNT = 0
false_positives_UNT = 0
false_negatives_UNT = 0

true_positives_TIN = 0
false_positives_TIN = 0
false_negatives_TIN = 0

len_set_b = len(test_set_b)

wrong_lines = ''

for i in range(len_set_b):
    text = test_set_b.iloc[i]['tweet']
    true_label = test_labels_b.iloc[i]['label']

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Predict
    with torch.no_grad():
        outputs = model_b(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        
    # Count correct predictions
    if predicted_class_id == true_label:
        number_correct_answers_b += 1

        if predicted_class_id == 0:
            true_positives_UNT += 1
        elif predicted_class_id == 1:
            true_positives_TIN += 1
    else:
        if predicted_class_id == 0:
            false_positives_UNT += 1
            false_negatives_TIN += 1
        elif predicted_class_id == 1:
            false_positives_TIN += 1
            false_negatives_UNT += 1

        wrong_lines += f"Tweet: {text}\n"
        wrong_lines += f"True label: {true_label}\n"
        wrong_lines += f"Predicted label: {predicted_class_id}\n"
        wrong_lines += "-" * 40 + "\n"
    
    os.makedirs('./tests', exist_ok=True)
    with open('./tests/logs_b.txt', 'w', encoding='utf-8') as f:
        f.writelines(wrong_lines)



accuracy_b = number_correct_answers_b / len_set_b 
precision_UNT = true_positives_UNT / (true_positives_UNT + false_positives_UNT + 1e-6)
precision_TIN = true_positives_TIN / (true_positives_TIN + false_positives_TIN + 1e-6) 
recall_UNT = true_positives_UNT / (true_positives_UNT + false_negatives_UNT + 1e-6)
recall_TIN = true_positives_TIN / (true_positives_TIN + false_negatives_TIN + 1e-6)
f1_UNT = 2 * precision_UNT * recall_UNT / (precision_UNT + recall_UNT + 1e-6)
f1_TIN = 2 * precision_TIN * recall_TIN / (precision_TIN + recall_TIN + 1e-6)


print('-'*40 + '\n')
print(f"Number of UNT items: {count_UNT}")
print(f"Number of TIN items: {count_TIN}")
print(f"Accuracy on Level B: {accuracy_b:.4f}")
print(f"Precision_UNT: {precision_UNT:.4f}")
print(f"Recall_UNT:    {recall_UNT:.4f}")
print(f"F1_UNT:        {f1_UNT:.4f}")
print(f"Precision_TIN: {precision_TIN:.4f}")
print(f"Recall_TIN:    {recall_TIN:.4f}")
print(f"F1_TIN:        {f1_TIN:.4f}\n")

# Evaluate model on test set C 'IND': 0, 'GRP': 1, 'OTH': 2
number_correct_answers_c = 0

true_positives_IND = 0
false_positives_IND = 0
false_negatives_IND = 0

true_positives_GRP = 0
false_positives_GRP = 0
false_negatives_GRP = 0

true_positives_OTH = 0
false_positives_OTH = 0
false_negatives_OTH = 0

len_set_c = len(test_set_c)

wrong_lines = ''

for i in range(len_set_c):
    text = test_set_c.iloc[i]['tweet']
    true_label = test_labels_c.iloc[i]['label']

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Predict
    with torch.no_grad():
        outputs = model_c(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        
    # Count correct predictions
    if predicted_class_id == true_label:
        number_correct_answers_c += 1

        if predicted_class_id == 0:
            true_positives_IND += 1
        elif predicted_class_id == 1:
            true_positives_GRP += 1
        elif predicted_class_id == 2:
            true_positives_OTH += 1
    else:  
        if predicted_class_id == 0:
            false_positives_IND += 1
        elif predicted_class_id == 1:
            false_positives_GRP += 1
        elif predicted_class_id == 2:
            false_positives_OTH += 1

        if true_label == 0:
            false_negatives_IND += 1
        elif true_label == 1:
            false_negatives_GRP += 1
        elif true_label == 2:
            false_negatives_OTH += 1

        wrong_lines += f"Tweet: {text}\n"
        wrong_lines += f"True label: {true_label}\n"
        wrong_lines += f"Predicted label: {predicted_class_id}\n"
        wrong_lines += "-" * 40 + "\n"
    
    os.makedirs('./tests', exist_ok=True)
    with open('./tests/logs_c.txt', 'w', encoding='utf-8') as f:
        f.writelines(wrong_lines)


# Print accuracy
accuracy_c = number_correct_answers_c / len_set_c
precision_IND = true_positives_IND / (true_positives_IND + false_positives_IND + 1e-6)
precision_GRP = true_positives_GRP / (true_positives_GRP + false_positives_GRP + 1e-6)
precision_OTH = true_positives_OTH / (true_positives_OTH + false_positives_OTH + 1e-6)
recall_IND = true_positives_IND / (true_positives_IND + false_negatives_IND + 1e-6)
recall_GRP = true_positives_GRP / (true_positives_GRP + false_negatives_GRP + 1e-6)
recall_OTH = true_positives_OTH / (true_positives_OTH + false_negatives_OTH + 1e-6)
f1_IND = 2 * precision_IND * recall_IND / (precision_IND + recall_IND + 1e-6)
f1_GRP = 2 * precision_GRP * recall_GRP / (precision_GRP + recall_GRP + 1e-6)
f1_OTH = 2 * precision_OTH * recall_OTH / (precision_OTH + recall_OTH + 1e-6)

print('-'*40 + '\n')
print(f"Number of IND items: {count_IND}")
print(f"Number of GRP items: {count_GRP}")
print(f"Number of OTH items: {count_OTH}")
print(f"Accuracy on Level C: {accuracy_c:.4f}")
print(f"Precision_IND: {precision_IND:.4f}")
print(f"Recall_IND:    {recall_IND:.4f}")
print(f"F1_IND:        {f1_IND:.4f}")
print(f"Precision_GRP: {precision_GRP:.4f}")
print(f"Recall_GRP:    {recall_GRP:.4f}")
print(f"F1_GRP:        {f1_GRP:.4f}")
print(f"Precision_OTH: {precision_OTH:.4f}")
print(f"Recall_OTH:    {recall_OTH:.4f}")
print(f"F1_OTH:        {f1_OTH:.4f}\n")

