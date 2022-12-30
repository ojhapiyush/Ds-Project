# Logistic regression 
# Gradient descent
# Loss function (cross entropy)
# Optimal learining rate is required

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class logisticreg():
    def __init__(self, lr=0.01, n_itr=1000):
        self.lr = lr
        self.n_iter = n_itr
        self.weights = None
        self.bias = None

    def fit(self, X, Y):
        n_samples,n_features = X.shape
        print(X.shape[0])
        # print(X.shape)
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iter):
            linearpred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linearpred)

            dw = (1/n_samples)*np.dot(X.T,(predictions - Y))
            db = (1/n_samples)*np.sum(predictions-Y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict(self, X):
        linearpred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linearpred)

        for y in y_pred:
            if y <= 0.5:
                y = 0
            else :
                y = 1

        return y_pred


