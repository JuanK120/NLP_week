import pandas as pd
import numpy as np
from readcsv import get_test_dataframe, get_test_labels
from datasets import Dataset
from transformers import AutoTokenizer, ModernBertForSequenceClassification, TrainingArguments, Trainer
import evaluate 

#  model_a = ModernBertForSequenceClassification.from_pretrained("./results/a/checkpoint-3972")

dataset_location = './dataset/'

test_set_a = get_test_dataframe(dataset_location+'testset-levela.tsv')
test_labels_a = get_test_labels(dataset_location+'labels-levela.csv')
 

for i in range(len(test_set_a)): 
    print(test_set_a.iloc[i]['tweet'])
    print(test_labels_a.iloc[i]['label'])
