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
        """Fits the model to the given data. For every class in y computes
        means and standard deviations of the corresponding subset of X.
        """
        self.X, self.y = X, y
        self.nobs_, self.nfeat_ = X.shape
        self.classes_ = np.unique(y)
        self.mu_ = np.empty((len(self.classes_), self.nfeat_))
        self.sigma_ = np.empty((len(self.classes_), self.nfeat_))
        for i, cls in enumerate(self.classes_):
            X_where_c = X[np.where(y == cls)]
            self.mu_[i] = X_where_c.mean(axis=0)
            self.sigma_[i] = X_where_c.std(axis=0)

    def predict(self, X):
        """Classifies each sample in X as the class that results in the largest
        P(Y|X) (posterior).

        Classification using Bayes Rule:

            P(Y|X) = P(X|Y) * P(Y) / P(X)

        or Posterior = Likelihood * Prior / Scaling Factor

        P(Y|X) - The posterior is the probability that sample x is of class y
                 given the feature values of x being distributed according to
                 distribution of y and the prior.
        P(X|Y) - Likelihood of data X given class distribution Y.
                 Gaussian distribution (given by _compute_likelihoods)
        P(Y)   - Prior (given by _compute_priors)
        P(X)   - Scales the posterior to make it a proper probability
                 distribution.  This term is ignored since it doesn't affect
                 which class distribution the sample is most likely to belong
                 to.
        """
        posteriors = []
        if not hasattr(self, 'priors'):
            self.priors = self._compute_priors(self.classes_)
        for i, cls in enumerate(self.classes_):  # loop to small to need vectorization
            likelihoods = self._compute_likelihoods(self.mu_[i], self.sigma_[i], X)
            posterior = self.priors[i] * np.multiply.reduce(likelihoods, axis=1)
            posteriors.append(posterior)
        posteriors = np.stack(posteriors)
        return self.classes_[np.argmax(posteriors, axis=0)]

    def _compute_priors(self, classes):
        return (self.y[:, np.newaxis] == classes).mean(axis=0)

    def _compute_likelihoods(self, means, sigmas, sample):
        return scipy.stats.norm(loc=means, scale=sigmas).pdf(sample)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    gnb = GaussianNB()
    gnb.fit(X, y)
    yhat = gnb.predict(X)
    print(f"Accuracy = {np.mean(y == yhat)}")
