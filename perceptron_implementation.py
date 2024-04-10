import numpy as np

class Perceptron(object):
    """Perceptron classifier
    Parameters:
        eta: float - learning rate
        n_iter: int - passes over the training data
        random_state: int - Random number generator seed for random weight initialization

    Attributes:
        w_: 1d-array - weight after fitting
        errors_: list - number of misclassifications (updates) in each epoch (iteration)
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1) # ternary operator

    def fit(self, X, y):
        """
        Fit training data
        Parameters:
        :param X: training vectors, where n_examples is the number of examples and n_features is the number of features
        :param y: target values
        :return: self: object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta  * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
