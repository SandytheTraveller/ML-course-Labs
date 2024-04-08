import numpy as np

class AdalineGD:
    """ADAptive LInear NEuron classifier.

   Parameters
   ------------
   eta : float
     Learning rate (between 0.0 and 1.0)
   n_iter : int
     number of epochs.
   random_state : int
     Random number generator seed for random weight initialization.

   Attributes
   -----------
   w_ : 1d-array
     Weights after fitting.
   b_ : Scalar
     Bias unit after fitting.
   losses_ : list
     Mean squared eror loss function values in each epoch.

   """

    def __init__(self, eta=0.01, epochs=50, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

    def net_input(self, X):
        """Calculate net input"""
        return X.dot(self.w_) + self.b_  # Xw + b

    def activation(self, X):
        """Compute linear activation"""
        return X

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        # weights and bias initialization
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        # Approach 1 - matrix formulation
        for i in range(self.epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)  # phi(Xw+b) = Xw + b
            errors = y - output  # definition of e
            self.w_ += self.eta * X.T.dot(errors) / X.shape[0]  # updating rules for all weights
            self.b_ += self.eta * errors.mean()  # updating the rule
            loss = (errors ** 2).mean()
            self.losses_.append(loss)
        return self

        # Approach 2 - extended learning algorithm with eta
        """
        for i in range(self.epochs): # for each epoch
            net_input = self.net_input(X)
            output = # phi(Xw + b) = Xw + b
            errors = # definition of e
            for j in range(self.w_.shape[0]): # for j in [1, ..., m]
                self.w_[j] + = # updating rule for a single weight
            self.b_ += # updatig rule for bias
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self
        """

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
