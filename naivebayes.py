from abc import ABC, abstractmethod
from pprint import pprint as print

import numpy as np
import scipy.stats


class NaiveBayesClassifier(ABC):
    """Abstract base class for naive Bayes classifiers"""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the model to the given data.

        Parameters
        ----------
        X : matrix of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array of shape (n_samples,)
            Target values.

        """

    def predict(self, X: np.ndarray) -> np.array:
        """Classifies each sample in X as the class that results in the largest
        P(Y|X) (posterior).

        Classification using Bayes Rule:

            P(Y|X) = P(X|Y) * P(Y) / P(X)

        or Posterior = Likelihood * Prior / Scaling Factor

        P(Y|X) - The posterior is the probability that sample x is of class y
                 given the feature values of x being distributed according to
                 distribution of y and the prior.
        P(X|Y) - Likelihood of data X given class distribution Y.
                 Gaussian distribution (given by _compute_likelihoods).
        P(Y)   - Prior (given by _compute_priorscompute )
        P(X)   - Scales the posterior to make it a proper probability
                 distribution.  This term is ignored since it doesn't affect
                 which class distribution the sample is most likely to belong
                 to.

        Parameters
        ----------
        X : matrix of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.


        Returns
        -------
        yhat : array of shape (n_samples,)
               Predicted target classes.

        """
        scores = []
        if not hasattr(self, "priors"):
            self.log_priors = self._compute_log_priors(self.classes_)
        for i, cls in enumerate(self.classes_):  # TODO vectorize this
            log_likelihoods = self._compute_log_likelihoods(X, cls_indx=i)
            score = self.log_priors[i] + np.sum(log_likelihoods, axis=1)
            scores.append(score)
        scores = np.stack(scores)
        return self.classes_[np.argmax(scores, axis=0)]

    @abstractmethod
    def _compute_log_likelihoods(self, X, cls_indx):
        """Computes the log-likelihood of X using the distribution of the class
        with index cls_indx

        Parameters
        ----------
        X : matrix of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        cls_indx : index of class for which log-likelihood is computed


        Returns
        -------
        log-likelihoods

        """

    def _store_data(self, X, y):
        """

        Parameters
        ----------
        X : matrix of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array of shape (n_samples,)
            Target values.

        """
        self.X, self.y = X, y
        self.n_samples_, self.n_features_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

    def _compute_log_priors(self, classes):
        """

        Parameters
        ----------
        classes : array of classes present in target vector y.


        Returns
        -------
        log-priors

        """
        return np.log((self.y[:, np.newaxis] == classes).mean(axis=0))


class GaussianNB(NaiveBayesClassifier):
    """
    Gaussian naive Bayes classifier for continuous data.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> gnb = GaussianNB()
    >>> gnb.fit(X, y)
    >>> yhat = gnb.predict(X)
    >>> print(f"My accuracy = {np.mean(y == yhat)}")
    0.96
    """
    def fit(self, X, y):
        """Gaussian case: for every class in y computes means and standard
        deviations of the corresponding subset of X.

        Parameters
        ----------
        X : matrix of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array of shape (n_samples,)
            Target values.


        Returns
        -------

        """
        self._store_data(X, y)
        self.mu_ = np.empty((len(self.classes_), self.n_features_))
        self.sigma_ = np.empty((len(self.classes_), self.n_features_))
        for i, cls in enumerate(self.classes_):  # TODO vectorize this
            X_where_c = X[np.where(y == cls)]
            self.mu_[i] = X_where_c.mean(axis=0)
            self.sigma_[i] = X_where_c.std(axis=0)

    def _compute_log_likelihoods(self, X, cls_indx):
        """

        Parameters
        ----------
        X : matrix of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        cls_indx : index of class for which log-likelihood is computed


        Returns
        -------
        log-likelihoods

        """
        loc = self.mu_[cls_indx]
        scale = self.sigma_[cls_indx]
        return np.log(scipy.stats.norm(loc, scale).pdf(X))


class MultinomialNB(NaiveBayesClassifier):
    """
    Gaussian naive Bayes classifier for continuous data.

    """
    def __init__(self, alpha=1):
        self.alpha = alpha
        super().__init__()

    def fit(self, X, y):
        """Multinomial case: for every class c in y and for every
        feature x compute P(x_i | c) where x_i is the ith level of feature x.

        P(x_i | c) = (N_ci + alpha) / (N_c + alpha * n)

        where alpha is the smoothing parameter.

        Parameters
        ----------
        X : matrix of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array of shape (n_samples,)
            Target values.

        """
        self._store_data(X, y)
        self.theta_ = np.empty((self.n_features_, self.n_classes_))
        for c in range(self.n_classes_):
            for i in range(self.n_features_):
                X_where_c = X[np.where(y == c)]
                N_ci = X_where_c[:, i].sum()
                self.theta_[i][c] = N_ci
        for c in range(self.n_classes_):
            N_c = self.theta_[:, c].sum()
            for i in range(self.n_features_):
                self.theta_[i][c] += self.alpha
                self.theta_[i][c] /= (N_c + self.alpha * self.n_samples_)

    def predict(self, X):
        scores = []
        if not hasattr(self, "priors"):
            self.log_priors = self._compute_log_priors(self.classes_)
        for c in range(self.n_classes_):
            log_likelihoods = self._compute_log_likelihoods(X, cls_indx=c)
            scores.append(self.log_priors[c] + log_likelihoods)
        scores = np.stack(scores)
        return self.classes_[np.argmax(scores, axis=0)]

    def _store_data(self, X, y):
        """

        Parameters
        ----------
        X : matrix of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array of shape (n_samples,)
            Target values.

        """
        self.feature_classes = [np.unique(feature) for feature in X.T]
        super()._store_data(X, y)

    def _compute_log_likelihoods(self, X, cls_indx):
        """

        Parameters
        ----------
        X : matrix of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        cls_indx : index of class for which log-likelihood is computed


        Returns
        -------
        log-likelihoods

        """
        likelihoods = [self.theta_[:, cls_indx] @ sample for sample in X]
        return np.log(likelihoods)


def run_gnb():
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    gnb = GaussianNB()
    gnb.fit(X, y)
    yhat = gnb.predict(X)
    print(f"My accuracy = {np.mean(y == yhat)}")

    from sklearn.naive_bayes import GaussianNB as SKGNB

    gnb = SKGNB()
    gnb.fit(X, y)
    yhat = gnb.predict(X)
    print(f"Sklearn's accuracy = {np.mean(y == yhat)}")


def run_mnb():
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    X = X.astype("int32")
    mnb = MultinomialNB()
    mnb.fit(X, y)
    yhat = mnb.predict(X)
    print(f"My accuracy = {np.mean(y == yhat)}")

    from sklearn.naive_bayes import MultinomialNB as SKMNB

    clf = SKMNB()
    clf.fit(X, y)
    yhat = clf.predict(X)
    print(f"Sklearn's accuracy = {np.mean(y == yhat)}")


if __name__ == "__main__":
    #  run_gnb()
    run_mnb()
