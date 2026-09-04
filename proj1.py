"""Naive Bayes Classifier for implementing whether a tumor is malignant or benign based on the features of the tumor"""

__Course__ = "CSE 574 - Intro To Machine Learning"
__author__ = "Viral Koshti"
__email__ = "viraljag@buffalo.edu"
__UBPersonNumber__ = "50761354"

import csv
import random

def read_csv(file_path):
    """Load the data from the CSV file and return a list of tuples containing the features and labels."""
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            features = list(map(float, row[:-1]))  # Convert features to float
            label = row[-1]  # Last column is the label
            data.append((features, label))
    return data

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
