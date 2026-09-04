"""Naive Bayes Classifier for implementing whether a tumor is malignant or benign based on the features of the tumor"""

__Course__ = "CSE 574 - Intro To Machine Learning"
__author__ = "Viral Koshti"
__email__ = "viraljag@buffalo.edu"
__UBPersonNumber__ = "50761354"

import csv
import math
import random


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

class SimpleGaussianNaiveBayes:

    def __init__(self):
        self.class_labels = []
        self.priors = {}
        self.stats = {}

    def fit(self, X, response):
        total_samples = len(X)
        feature_count = len(X[0])

        # Find the different classes in the training data
        self.class_labels = sorted(set(response))

        for label in self.class_labels:
            # Get all samples that belong to this class
            class_samples = [
                X[i] for i in range(total_samples)
                if response[i] == label
            ]

            class_size = len(class_samples)

            # Probability of seeing this class before looking at the feature values
            self.priors[label] = class_size / total_samples

            feature_stats = []

            for column in range(feature_count):
                values = [sample[column] for sample in class_samples]

                mean = sum(values) / class_size

                if class_size > 1:
                    variance = sum(
                        (value - mean) ** 2 for value in values
                    ) / (class_size - 1)
                else:
                    variance = 0.0

                # Add a very small number so the variance is never zero
                feature_stats.append({
                    "mean": mean,
                    "var": variance + 1e-9
                })

            self.stats[label] = feature_stats

    def _calculate_likelihood(self, value, mean, var):
        # Gaussian probability density function
        normal = 1.0 / math.sqrt(2.0 * math.pi * var)
        exponent = math.exp(
            -((value - mean) ** 2) / (2.0 * var)
        )
        result = normal * exponent
        return result

    def predict_one(self, sample):
        scores = {}

        for label in self.class_labels:
            # Start with the log of the prior probability
            score = math.log(self.priors[label])

            for i, value in enumerate(sample):
                stats = self.stats[label][i]

                likelihood = self._calculate_likelihood(
                    value,
                    stats["mean"],
                    stats["var"]
                )

                # Log probabilities help avoid very small numbers
                score += math.log(max(likelihood, 1e-300))

            scores[label] = score

        # Return whichever class has the highest score
        result = max(scores, key=scores.get)
        return result

    def predict(self, X):
        return [self.predict_one(sample) for sample in X]



def get_accuracy(y_true, y_pred):
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