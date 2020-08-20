import numpy as np


def sigmoid(z):
    """
    SIGMOID Compute sigmoid function
    """

    g = np.divide(1., 1. + np.exp(-z))
    return g

def sigmoidGradient(z):
    """
    SIGMOIDGRADIENT returns the gradient of the sigmoid function evaluated at z
    """

    g = np.multiply(sigmoid(z), (1. - sigmoid(z)))
    return g

def tanh(z):
    """
    TANH Compute hyperbolic tangent function
    """
    
    g = np.tanh(z)
    return g

def tanhGradient(z):
    """
    TANHGRADIENT returns the gradient of the tanh function evaluated at z
    """

    g = - np.multiply(tanh(z), tanh(z)) + 1.
    return g

def softmax(z):
    """
    SOFTMAX returns input normalized into a probability distribution, where all compponents add up to 1
    modified from https://stackoverflow.com/a/39558290
    """
    
    g = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    return g

def tanhReScal(z):
    """
    TANHRESCAL returns tanh input scaled to range [0, 1], suitable for use with logistic regression cost function
    """
    
    g = (z + 1.) / 2.
    return g