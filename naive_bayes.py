import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datasets import load_dataset
from matplotlib import pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import HashingVectorizer

NAME_DATASET = "ccdv/arxiv-classification"

ORIGINAL_LABELS = ['math.AC', 'cs.CV', 'cs.AI', 'cs.SY', 'math.GR', 'cs.DS', 'cs.CE', 'cs.PL', 'cs.IT', 'cs.NE', 'math.ST']

TEST_SCORE = True
TEST_FEATURE_SIZE = 1000

# 0 is for original Math classes, 1 is for original CS classes
classConversion = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]

print("=== Loading Data ===")
train_data = load_dataset(NAME_DATASET, split = "train")
validation_data = load_dataset(NAME_DATASET, split = "validation")
test_data = load_dataset(NAME_DATASET, split = "test")

print("=== Data Loaded, Processing ===")
train_y = train_data['label']
test_y = test_data['label']
validation_y = validation_data['label']

if TEST_SCORE:
    vectorizer = HashingVectorizer(n_features=TEST_FEATURE_SIZE)
    
    # Fit to vectorizer
    train_X = vectorizer.fit_transform(train_data['text'])
    test_X = vectorizer.transform(test_data['text'])
    
    # Train Model
    gnb = GaussianNB()
    gnb.fit(train_X.toarray(), train_y)

    # Calculating the scores
    train_score = gnb.score(train_X.toarray(), train_y)
    test_score = gnb.score(test_X.toarray(), test_y)

    # Printing scores
    print("\n=== Testing Scores ===")
    print("Train score: ", train_score)
    print("Test score: ", test_score)
    
    exit()


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

# Obtain the counts of each label in the dataset
distribution = [0] * 11

for label in train_y:
    distribution[label] += 1
for label in validation_y:
    distribution[label] += 1
for label in test_y:
    distribution[label] += 1



plt.figure(figsize=(10, 6))
plt.bar(ORIGINAL_LABELS, distribution)
plt.xlabel('Original Labels', fontsize=15)
plt.ylabel('Occurrences', fontsize=15)
plt.title('Occurrences in Dataset by Original Label', fontsize=20)
plt.savefig('NB/NB_occurrences_original_labels.png')

training_accuracy = []
validation_accuracy = []

for vectorSize in features:
    print(f"Starting feature size: {vectorSize}")
    
    vectorizer = HashingVectorizer(n_features=vectorSize)
    
    # Fit to vectorizer
    train_X = vectorizer.fit_transform(train_data['text'])
    validation_X = vectorizer.transform(validation_data['text'])
    
    # Train Model
    gnb = GaussianNB()
    gnb.fit(train_X.toarray(), train_y)

    # Calculating the scores
    train_score = gnb.score(train_X.toarray(), train_y)
    validation_score = gnb.score(validation_X.toarray(), validation_y)
    test_score = gnb.score(test_X.toarray(), test_y)

    # Printing scores
    print("Train score: ", train_score)
    print("Validation score: ", validation_score)
    print("Test score: ", test_score)

    # Append accurracies
    training_accuracy.append(train_score)
    validation_accuracy.append(validation_score)

    print(f"Finished feature size: {vectorSize}")

print("=== Finished Training ===")
print("Training accuracies: ", training_accuracy)
print("Validation accuracies: ", validation_accuracy)

plt.plot(features, validation_accuracy, marker='o', label='Validation Accuracy')
plt.plot(features, training_accuracy, marker='o', label='Training Accuracy')
plt.ylabel("Accuracy Score")
plt.xlabel("Feature Vector Size")
plt.legend()
plt.title('Accuracy vs. Feature Vector Size')

plt.savefig('NB/NB_accuracy_vs_feature_vector_size.png')

print(f'Features: {features}')
print(f'Training Accuracies: {training_accuracy}')
print(f'Validation Accuracies: {validation_accuracy}')
