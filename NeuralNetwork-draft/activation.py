import numpy as np


def sigmoid(z):
    """
    SIGMOID Compute sigmoid function
    """

    g = np.divide(1.0, 1.0 + np.exp(-z))
    return g

def sigmoidGradient(z):
    """
    SIGMOIDGRADIENT returns the gradient of the sigmoid function evaluated at z
    """

    g = np.multiply(sigmoid(z), (1 - sigmoid(z)))
    return g