import pandas as pd
import numpy as np
import emoji
import wordsegment as wsgmt
import re

dataset_location = './dataset/olid-training-v1.0.tsv'

wsgmt.load()

def segment_tweet(tweet):
    hashtags = re.findall(r"#\w+", tweet)
    for tag in hashtags:
        segmented = " ".join(wsgmt.segment(tag[1:]))  # remove '#' and segment
        tweet = tweet.replace(tag, segmented)
    return tweet

def replace_url(tweet):
    urls = re.findall(r"http\w+", tweet)
    for url in urls:
        tweet = tweet.replace(url, "URL")
    return tweet

def segment_hashtags(dataframe):
    dataframe['tweet'] = dataframe['tweet'].apply(segment_tweet)
    return dataframe    

def eliminate_urls(dataframe):
    dataframe['tweet'] = dataframe['tweet'].apply(replace_url)
    return dataframe   

def get_dataframe(file):

    dataset = pd.read_csv(dataset_location,sep='\t')

    dataset['tweet'] = dataset['tweet'].apply(emoji.emojize)
    dataset = segment_hashtags(dataset)
    dataset = eliminate_urls(dataset)

    subset_a = dataset[['id','tweet','subtask_a']]

    temp = dataset[['id','tweet','subtask_b']]
    subset_b = temp[pd.notna(temp['subtask_b'])]

    temp = dataset[['id','tweet','subtask_c']]
    subset_c = temp[pd.notna(temp['subtask_c'])]

    return dataset, subset_a, subset_b, subset_c



