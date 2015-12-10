from __future__ import division
import numpy as np
from sklearn.utils import check_X_y, check_array
import random
import plotly.plotly as py
import plotly.graph_objs as go

class SVC(object):
    """Implements a binary classifier SVM using SGD.

    Parameters
    ----------
    reg : float
        The regularization constant

    Attributes
    ----------
    weights_ : [n_features]
        Vector representing the weights of the SVM

    bias_ : float
        Float representing the bias term of the SVM

    n_features_ : int
        The number of features in the dataset.

    n_classes_ : int
        The number of classes in the dataset.
    """

    def __init__(self, reg=1):
        self.reg = reg

        # It's just my habit to suffix variables with '_' if they're
        # instantiated in the "fitting" stage of the algorithm.
        # This is how the scikit-learn maintainers style their code.
        self.weights_ = None
        self.bias_ = 0
        self.n_features_ = None
        self.classes_ = None

    def fit(self, X, y, validation_size=0.25, n_epochs=50, n_steps=100):
        def f(x):
            result = ((self.weights_.dot(x)) + self.bias_)
            # quit()
            return result

        """Trains the support vector machine
        """
        X, y = check_X_y(X, y)

        # Need to make sure y labels are correct
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2

        # Converts the binary labels of y into -1 and 1.
        new_y = np.zeros(len(y))
        new_y[y == self.classes_[0]] = -1
        new_y[y == self.classes_[1]] = 1
        y = new_y

        # Shuffle the training set
        perm = np.random.permutation(len(X))
        X = X[perm]
        y = y[perm]

        # Create the validation set
        val_ind = len(X) * validation_size
        val_X = X[:val_ind]
        val_y = y[:val_ind]
        X = X[val_ind:]
        y = y[val_ind:]
        self.n_features_ = X.shape[1]
        # Initialize weights to between [-1, 1)
        self.weights_ = np.random.random(X.shape[1]) * 2 - 1
        print len(self.weights_)
        self.bias_ = random.random()*2-1
        alpha = 1
        beta = 1

        validation_accuracy = list()
        for cur_epoch in range(n_epochs):
            step = (alpha / (cur_epoch + beta))
            for i in range(10):
                for j in range(10):
                    rand_idx = np.random.choice(np.arange(len(X)))
                    cur_X, cur_y = X[rand_idx], y[rand_idx]
                    if((cur_y * f(cur_X)) >= 1):
                        grad_w = self.reg * self.weights_
                        grad_b = 0
                    else:
                        grad_w = (self.reg * self.weights_) - (cur_y * cur_X)
                        grad_b = -cur_y
                    self.weights_ = self.weights_ - (step * grad_w)
                    self.bias_ = self.bias_ - (step * grad_b)
                total = 0
                correct = 0
                for val_i in range(len(val_X)):
                    total += 1
                    cur_X = val_X[val_i]
                    cur_y = val_y[val_i]
                    # print cur_X
                    if (self.predict(cur_X) == cur_y):
                        correct += 1
                accuracy = (correct * 1.0) / (total * 1.0)
                if(i % 10 == 0):
                    validation_accuracy.append([self.reg, cur_epoch, accuracy])
        matrix = np.matrix(validation_accuracy)
        x = matrix[:, 1]
        x = x.T
        x = np.asarray(x)
        # print x[0]
        y = matrix[:, 2]
        y = y.T
        y = np.asarray(y)
        # print y[0]
        # print x.shape
        
        return (x, y)

        # data = [
        #     go.Scatter(
        #         x = x[0],
        #         y = y[0]
        #         )
        # ]
        # py.image.save_as({'data': data}, '111.png')
        
        # return validation_accuracy

        
        
        # TODO: Write the gradient step update for the SVM
        # It might be helpful to return the prediction accuracy of
        # the SVM on some evaluation set.
        #
        # See sklearn.metrics.accuracy_score

    def predict(self, X):
        """Returns the predictions (-1 or 1) on the feature set X.
        """
        # print X
        X = np.reshape(X, (1, 30))
        # print X
        # print self.weights_
        
        X = check_array(X)
        # print X.shape[1]
        # print self.n_features_

        assert X.shape[1] == self.n_features_
        result = self.weights_.dot(X.T) + self.bias_
        
        
        assert result != 0
        if result > 0:
            return 1
        else:
            return -1