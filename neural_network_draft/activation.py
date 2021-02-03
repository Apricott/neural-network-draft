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

def softmax(z, axis=1):
    """
    SOFTMAX returns input normalized into a probability distribution, where all compponents add up to 1
    modified from https://stackoverflow.com/a/47829935
    
    The subtraction of the maximum value is to avoid arithmetic overflow in the case of very large values 
    """
    
    mz = np.amax(z, axis = axis, keepdims = True)
    z_exp = np.exp(z - mz)
    z_sum = np.sum(z_exp, axis = axis, keepdims = True)
    g = z_exp / z_sum

    # If there is 1 in g, convert it to 0.9999999 to avoid problems with the logistic regression cost function
    eps = 0.0000001 * (g == 1)
    g = g - eps
         
    return g

def tanhReScal(z):
    """
    TANHRESCAL returns tanh input scaled to range [0, 1], suitable for use with the logistic regression cost function
    """
    
    g = (z + 1.) / 2.
    return g

def ReLU(z, leak=.01):
    """
    RELU Compute rectified linear unit function
    
    By default, the leaky variant is used to make it suitable for use with the logistic regression cost function
    """

    g1 = z * (z > 0)
    g2 = z * leak * (z <= 0)

    g = g1 + g2

    return g

def ReLUGradient(z, leak=.01):
    """
    RELUGRADIENT returns the gradient of the rectified linear unit function evaluated at z
    """

    g1 = 1. * (z > 0)
    g2 = leak * (z <= 0)

    g = g1 + g2
    
    return g
