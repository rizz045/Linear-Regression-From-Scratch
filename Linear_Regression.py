import numpy as np

# Creating in class template

class Linear_Regression():

    #initiating parameters (hyper parameters: learning_rate and no_of_iteration)
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations