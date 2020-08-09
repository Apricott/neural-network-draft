import numpy as np

def lrCostFunction(theta, X, y, lmbd, activ_func):

    # initialize some useful variables
    m = len(y)
    J = 0
    grad = np.zeros(theta.shape)

    pred = activ_func(X @ theta)

    J = 1/m * np.sum(-y.T @ np.log(pred) - (1 - y.T) @ np.log(1 - pred))
    J_reg = lmbd/(2*m) * np.sum(theta[1:] ** 2)
    J = J + J_reg 

    grad = X.T @ (pred - y) / m
    grad_reg = np.concatenate([np.array([[0]]), lmbd/m * theta[1:]]).T
    grad = grad + grad_reg

    grad = grad.reshape(grad.size, order="F")

    return J, grad


def predict():
    """

    """