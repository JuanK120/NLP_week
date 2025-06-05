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


test_set_b = get_test_dataframe(dataset_location+'testset-levelb.tsv')
label_map_b = {'UNT': 0, 'TIN': 1}
test_labels_b = get_test_labels(dataset_location+'labels-levelb.csv')
test_labels_b['label'] = test_labels_b['label'].apply(lambda x: label_map_b[x])


test_set_c = get_test_dataframe(dataset_location+'testset-levelc.tsv')
label_map_c = {'IND': 0, 'GRP': 1, 'OTH': 2}
test_labels_c = get_test_labels(dataset_location+'labels-levelc.csv') 
test_labels_c['label'] = test_labels_c['label'].apply(lambda x: label_map_c[x])


# Evaluate model on test set A
number_correct_answers_a = 0
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
    else:
        wrong_lines = wrong_lines + "Tweet:" + text + "\n"
        wrong_lines = wrong_lines + "True label:" + str(true_label) + "\n"
        wrong_lines = wrong_lines + "Predicted label:"+ str(predicted_class_id) + "\n"
        wrong_lines += "-" * 40 + "\n"
    
    os.makedirs('./tests', exist_ok=True)
    with open('./tests/logs_a.txt', 'w', encoding='utf-8') as f:
        f.writelines(wrong_lines)


# Print accuracy
accuracy = number_correct_answers_a / len_set_a
print(f"Accuracy on Level A: {accuracy:.4f}")


# Evaluate model on test set A
number_correct_answers_b = 0
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
    else:
        wrong_lines = wrong_lines + "Tweet:" + text + "\n"
        wrong_lines = wrong_lines + "True label:" + str(true_label) + "\n"
        wrong_lines = wrong_lines + "Predicted label:"+ str(predicted_class_id) + "\n"
        wrong_lines += "-" * 40 + "\n"
    
    os.makedirs('./tests', exist_ok=True)
    with open('./tests/logs_b.txt', 'w', encoding='utf-8') as f:
        f.writelines(wrong_lines)


# Print accuracy
accuracy = number_correct_answers_b / len_set_b
print(f"Accuracy on Level B: {accuracy:.4f}")

# Evaluate model on test set A
number_correct_answers_c = 0
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
    else:
        wrong_lines = wrong_lines + "Tweet:" + text + "\n"
        wrong_lines = wrong_lines + "True label:" + str(true_label) + "\n"
        wrong_lines = wrong_lines + "Predicted label:"+ str(predicted_class_id) + "\n"
        wrong_lines += "-" * 40 + "\n"
    
    os.makedirs('./tests', exist_ok=True)
    with open('./tests/logs_c.txt', 'w', encoding='utf-8') as f:
        f.writelines(wrong_lines)


# Print accuracy
accuracy = number_correct_answers_c / len_set_c
print(f"Accuracy on Level C: {accuracy:.4f}")

