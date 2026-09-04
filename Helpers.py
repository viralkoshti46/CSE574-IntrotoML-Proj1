import math


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