import numpy as np
import pandas as pd
import spacy
import random

df = pd.read_csv("IMDB Dataset.csv")

df["sentiment"].replace({"positive": 1, "negative": 0}, inplace=True)


def load_data(csv_file, split=0.9):
    data = csv_file
    
    # Shuffle data
    train_data = data.sample(frac=1, random_state=7)
    
    texts = train_data.review.values
    labels = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)}
              for y in train_data.sentiment.values]
    split = int(len(train_data) * split)
    
    train_labels = [{"cats": labels} for labels in labels[:split]]
    val_labels = [{"cats": labels} for labels in labels[split:]]
    
    return texts[:split], train_labels, texts[split:], val_labels

train_texts, train_labels, val_texts, val_labels = load_data(df)
