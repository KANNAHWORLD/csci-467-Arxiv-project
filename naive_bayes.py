import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datasets import load_dataset

from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import HashingVectorizer

NAME_DATASET = "ccdv/arxiv-classification"

# 0 is for original Math classes, 1 is for original CS classes
classConversion = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]

print("=== Loading Data ===")
train_data = load_dataset(NAME_DATASET, split = "train")
validation_data = load_dataset(NAME_DATASET, split = "validation")
test_data = load_dataset(NAME_DATASET, split = "test")

print("=== Data Loaded, Processing ===")
train_y = [classConversion[x] for x in train_data['label']]
test_y = [classConversion[x] for x in test_data['label']]
validation_y = [classConversion[x] for x in validation_data['label']]

print("=== Data Processed, Vectorizing ===")
vectorizer = HashingVectorizer(n_features=30000)

train_X = vectorizer.fit_transform(train_data['text'])
test_X = vectorizer.transform(test_data['text'])
validation_X = vectorizer.transform(test_data['text'])

print("=== Vectorized, Training Model ===")
gnb = GaussianNB()
gnb.fit(train_X.toarray(), train_y)

training_accurracy = gnb.score(train_X.toarray(), train_y)
print(f"Accurracy train data points: {training_accurracy}")

validation_accurracy_score = gnb.score(validation_X.toarray(), validation_y)
print(f"Accurracy on validation: {validation_accurracy_score}")

features = [10, 50, 100, 200, 500, 1000, 2000, 20000]
training_accurracy = []
validation_accurracy = []


for vectorSize in features:
    print(f"Starting feature size: {vectorSize}")
    
    vectorizer = HashingVectorizer(n_features=vectorSize)
    
    # Fit to vectorizer
    train_X = vectorizer.fit_transform(train_data['text'])
    validation_X = vectorizer.transform(validation_data['text'])
    
    # Train Model
    gnb = GaussianNB()
    gnb.fit(train_X.toarray(), train_y)
    
    # Append accurracies
    training_accurracy.append(gnb.score(train_X.toarray(), train_y))
    validation_accurracy.append(gnb.score(validation_X.toarray(), validation_y))
    
    print(f"Finished feature size: {vectorSize}")

print(f"Training accuracies: {training_accurracy}")
print(f"Validation accuracies: {validation_accurracy}")
