import numpy as np

# Creating in class template

class Linear_Regression():

    #initiating parameters (hyper parameters: learning_rate and no_of_iteration)
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        # number of training example and number of features
        self.m, self.n = X.shape  # rows and columns of the dataset

        #initiating the weights and bias
        self.w = np.zeros(self.n)   # we can initialize it with 0 as well but often not every problem has only 1 independent colummn
        self.b = 0
        self.X = X
        self.Y = Y

        #Implementing gradient Descent algorithm
        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        Y_prediction = self.predict(self.X)

        # Calculate the variance
        dw = -(2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m
        db = -2 * np.sum(self.Y - Y_prediction)/self.m

        # Updating the weights
        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db

    def predict(self, X):
        return X.dot(self.w) + self.b      #returns the line formula