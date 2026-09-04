"""Naive Bayes Classifier for implementing whether a tumor is malignant or benign based on the features of the tumor"""

__Course__ = "CSE 574 - Intro To Machine Learning"
__author__ = "Viral Koshti"
__email__ = "viraljag@buffalo.edu"
__UBPersonNumber__ = "50761354"

import csv
import math
import random
from Helpers import SimpleGaussianNaiveBayes


def read_csv(file):
    """Load the dataset from a CSV file and return the features and labels as separate lists"""
    features = [] # Empty features list
    labels = [] # Empty labels list

    # Read the CSV file and skip the header row
    with open(file=file, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:
            # Ignore empty rows
            if not row or not row[0]:
                continue

            # All columns except the last one are input features
            row_features = [float(value) for value in row[:-1]]

            # The last column contains the class label
            label = int(float(row[-1]))

            features.append(row_features)
            labels.append(label)

    return features, labels

def split_data(X, response, test_size=0.25, seed=42):
    """Shuffle the indices instead of the data itself so the labels stay matched with their corresponding samples"""
    indices = list(range(len(X)))

    random.seed(seed)
    random.shuffle(indices)

    test_count = int(round(len(X) * test_size))

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = [X[i] for i in train_indices]
    y_train = [response[i] for i in train_indices]

    X_test = [X[i] for i in test_indices]
    y_test = [response[i] for i in test_indices]

    return X_train, y_train, X_test, y_test


def get_accuracy(y_true, y_pred):
    """Calculate the accuracy of predictions by comparing them to the true labels"""
    correct = sum(
        1 for actual, predicted in zip(y_true, y_pred)
        if actual == predicted
    )
    accuracy = correct / len(y_true)
    return accuracy


if __name__ == "__main__":

    # Load the data and create the train/test split
    features, labels = read_csv("DataP1.csv")

    X_train, y_train, X_test, y_test = split_data(
        features,
        labels
    )

    # Train the model
    model = SimpleGaussianNaiveBayes()
    model.fit(X_train, y_train)

    # Test the model on the samples it has not seen during training
    predictions = model.predict(X_test)
    accuracy = get_accuracy(y_test, predictions)

    print(
        f"Dataset loaded. Training size: {len(X_train)}, "
        f"Test size: {len(X_test)}"
    )
    print(f"Test Accuracy: {accuracy:.2%}")

    # Try the trained model on one new example
    new_case = [
        13.0, 15.0, 85.0, 500.0, 0.1, 0.15, 0.1, 0.05, 0.2, 0.08,
        0.5, 1.5, 4.0, 70.0, 0.01, 0.02, 0.02, 0.01, 0.015, 0.002,
        14.0, 20.0, 90.0, 600.0, 0.2, 0.25, 0.2, 0.1, 0.3, 0.1
    ]

    result = model.predict_one(new_case)

    if result == 0:
         diagnosis = "Malignant"
    else:
        diagnosis = "Benign"

    print(f"\nNew Sample Diagnosis: {diagnosis} (Class {result})")