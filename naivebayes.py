from abc import ABC, abstractmethod

import numpy as np
import scipy.stats


class NaiveBayesClassifier(ABC):
    """ABC for the Naive Bayes Classifier"""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.array:
        pass


class GaussianNB(NaiveBayesClassifier):

    def fit(self, X, y):
        self.X, self.y = X, y
        self.nobs_, self.nfeat_ = X.shape
        self.classes_ = np.unique(y)
        self.mu_ = np.empty((len(self.classes_), self.nfeat_))
        self.sigma_ = np.empty((len(self.classes_), self.nfeat_))
        for i, cls in enumerate(self.classes_):
            X_where_c = X[np.where(y == cls)]
            self.mu_[i] = X_where_c.mean(axis=0)
            self.sigma_[i] = X_where_c.std(axis=0)
        self.trained = True

    def predict(self, X):
        return [self._classify(sample) for sample in X]

    def _classify(self, sample):
        """ Classifies the sample as the class that results in the largest
        P(Y|X) (posterior).

        Classification using Bayes Rule:

            P(Y|X) = P(X|Y) * P(Y) / P(X)

        or Posterior = Likelihood * Prior / Scaling Factor

        P(Y|X) - The posterior is the probability that sample x is of class y
                 given the feature values of x being distributed according to
                 distribution of y and the prior.
        P(X|Y) - Likelihood of data X given class distribution Y.
                 Gaussian distribution (given by _compute_likelihood)
        P(Y)   - Prior (given by _compute_prior)
        P(X)   - Scales the posterior to make it a proper probability
                 distribution.  This term is ignored since it doesn't affect
                 which class distribution the sample is most likely to belong
                 to.
        """
        posteriors = []
        for i, cls in enumerate(self.classes_):
            posterior = self._compute_prior(cls)
            for feature, mean, sigma in zip(sample, self.mu_[i], self.sigma_[i]):
                likelihood = self._compute_likelihood(mean, sigma, feature)
                posterior *= likelihood
            posteriors.append(posterior)
        return self.classes_[np.argmax(posteriors)]

    def _compute_prior(self, cls):
        return np.mean(self.y == cls)

    def _compute_likelihood(self, mean, sigma, x):
        return scipy.stats.norm(loc=mean, scale=sigma).pdf(x)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    gnb = GaussianNB()
    gnb.fit(X, y)
    yhat = gnb.predict(X)
    print(f"Accuracy = {np.mean(y == yhat)}")
