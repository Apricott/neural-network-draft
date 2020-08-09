import numpy as np


def sigmoid(z):
    """
    SIGMOID Compute sigmoid functoon
    """


    g = np.divide(1.0, 1.0 + np.exp(-z))
    return g