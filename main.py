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

nlp = spacy.blank("en")

textcat = nlp.create_pipe(
    "textcat",
    config={
        "exclusive_classes":True,
        "architecture":"bow"
    }
)

nlp.add_pipe(textcat)

textcat.add_label("NEGATIVE")
textcat.add_label("POSITIVE")

from spacy.util import minibatch

def train(model, train_data, optimizer):
    losses ={}
    random.seed(1)
    random.shuffle(train_data)

    batches = minibatch(train_data, size=8)
    for batch in batches:
        texts, labels = zip(*batch)

    model.update(texts, labels, sgd=optimizer, losses=losses)

    return losses

spacy.util.fix_random_seed(1)
random.seed(1)

optimizer = nlp.begin_training()
train_data = list(zip(train_texts, train_labels))
losses = train(nlp, train_data, optimizer)

def predict(model, texts):
    docs = [nlp.tokenizer(text) for text in texts]

    textcat = model.get_pipe('textcat')
    scores, _ = textcat.predict(docs)

    predicted_class = scores.argmax(axis=1)

    return predicted_class

def evaluate(model, texts, labels):
    predicted_class = predict(model, texts)

    true_class = [int(each['cats']['POSITIVE'])for each in labels]

    correct_predictions = predicted_class == true_class

    accuracy = correct_predictions.mean()

    return accuracy

n_iters = 5
for i in range(n_iters):
    losses = train(nlp, train_data, optimizer)
    accuracy = evaluate(nlp, val_texts, val_labels)
    print(f"Loss: {losses['textcat']:.3f} \t Accuracy: {accuracy:.3f}")

