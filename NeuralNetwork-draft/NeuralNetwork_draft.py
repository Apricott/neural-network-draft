import numpy as np
from scipy.optimize import fmin_ncg
from activation import sigmoid

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


def predict(Theta: list, X: np.ndarray, activation: object = sigmoid) -> np.ndarray:
    """
    Predict output of NN classifier given the list of parameters Theta, values array X and activation function used to train NN classifier
    
    Theta is a list of arrays with weights corresponding to the network layers (excluding the output layer)
    The arrays are sized according to the formula [number of neurons in the next layer(!)] * [number of neurons in the current layer(!)]
    (!) including the bias unit

    > Theta[0]              - input layer weights
    > Theta[1]  (optional)  - first hidden layer weights
    > Theta[>1] (optional)  - subsequent layers' weights

    The array of predicted classes is calculated by selecting the class with the greatest probability resulted from activating all the input and hidden layers 
    (considering the activation function mapps values to the range [0,1] 
    and 
    the number of neurons in the output layer is equal 
        - to the number of classes - in case of multiclass classification
        - 1 or 2 - in case of binary classification
    )

    """
   
    # Initialize some useful variables
    m = X.shape[0]
    p = np.zeros((m, 1))

    # Add bias unit to the X array
    a = np.concatenate([np.ones((m, 1)), X], axis=1)

    for layer, theta in enumerate(Theta, 1):
        # calculate result of activating layer
        a = activation(a @ theta.T)
        
        #  Add bias unit to the resulting array if it's not the last layer
        if layer != len(Theta):
            a = np.concatenate([np.ones((a.shape[0], 1)), a], axis=1)

    # Predicted classes are the indices of elements with the biggest value in every row of resulting array a
    p = np.argmax(a, axis=1)

    return p


def nnCostFunction(nn_params: np.ndarray, layer_sizes: list, num_classes: int, X: np.ndarray, y: np.ndarray, lmbd: float = 0, activation: object = sigmoid) -> float:
    """
    Return cost value for given weights array nn_params and values array X with assigned classes in array y

    nn_params   - 1 dimensional array of weights associated with every neuron in neural network.
                  Flattening of weights arrays is necessary to correctly feed them to the optimization function.

    layer_sizes - list with sizes of every layer in network, including the input and output layers,
                  WITHOUT the bias units. 

    num_classes - integer specifying the number of existig classes

    lmbd        - regularization parameter, omits the bias units
    
    activation  - activation function used to train NN classifier

    """

    # Initialize some useful variables
    J = 0
    J_reg = 0
    m = X.shape[0]
    p = np.zeros((m, 1))

    y = np.eye(num_classes)[y,:].reshape((m, num_classes))

    Theta = []
    split = 0

    # reshape Theta arrays from nn_params array back to their original shape
    for i in range(0, len(layer_sizes) - 1):
        temp = layer_sizes[i + 1] * (layer_sizes[i] + 1) + split
        size = (layer_sizes[i + 1], layer_sizes[i] + 1)
        Theta += [np.reshape(nn_params[split:temp], size)]
        split = temp
    
    # Add bias unit to the X array
    a = np.concatenate([np.ones((m, 1)), X], axis=1)

    # Feedforward the neural network
    for layer, theta in enumerate(Theta, 1):

        # calculate result of activating layer
        a = activation(a @ theta.T)    
        J_reg = J_reg + np.sum(np.sum((theta[:, 1:] ** 2)))

        #  Add bias unit to the resulting array if it's not the last layer
        if layer != len(Theta):
            a = np.concatenate([np.ones((a.shape[0], 1)), a], axis=1)

    J_reg *= lmbd/(2*m)
    J = np.sum(np.sum((-y * np.log(a)) - ((1 - y) * np.log(1 - a)))) / m
    J += J_reg

    return J

