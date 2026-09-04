"""Naive Bayes Classifier for implementing whether a tumor is malignant or benign based on the features of the tumor"""

__Course__ = "CSE 574 - Intro To Machine Learning"
__author__ = "Viral Koshti"
__email__ = "viraljag@buffalo.edu"
__UBPersonNumber__ = "50761354"

import csv

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

