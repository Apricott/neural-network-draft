import numpy as np
import activation
import misc


def predict(Theta: list, X: np.ndarray, fun: object = activation.sigmoid) -> np.ndarray:
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
        a = fun(a @ theta.T)
        
        #  Add bias unit to the resulting array if it's not the last layer
        if layer != len(Theta):
            a = np.concatenate([np.ones((a.shape[0], 1)), a], axis=1)

    # Predicted classes are the indices of elements with the biggest value in every row of resulting array a
    p = np.argmax(a, axis=1)

    return p


def nnCostFunction(nn_params: np.ndarray, layer_sizes: list, num_classes: int, X: np.ndarray, y: np.ndarray, lmbd: float = 0,
                        fun: object = activation.sigmoid, fun_grad: object = activation.sigmoidGradient) -> tuple([float, np.ndarray]):
    """
    Return cost value and gradient for given weights array nn_params and values array X with assigned classes in array y

    nn_params   - 1 dimensional array of weights associated with every neuron of the neural network.
                    "Unrolling" the weights arrays to a vector is necessary to correctly feed them to the optimization function.

    layer_sizes - list with sizes of every layer in the neural network, including the input and output layers,
                    WITHOUT the bias units. 

    num_classes - integer specifying the number of existig classes

    lmbd        - regularization parameter, omits the bias units
    
    activation  - activation function used to train NN classifier

    activationGrad - gradient of the activation function

    """

    # Initialize some useful variables
    J = 0
    J_reg = 0
    Z = []
    A = []
    Grad = []
    m = X.shape[0]
    p = np.zeros((m, 1))

    y = np.eye(num_classes)[y,:].reshape((m, num_classes))
    
    split = 0
    # reshape Theta arrays from nn_params array back to their original shape
    Theta = misc.reshapeTheta(nn_params, layer_sizes)
    
    # Add bias unit to the X array
    a = np.concatenate([np.ones((m, 1)), X], axis=1)
    A += [a]

    # Feedforward the neural network
    for layer, theta in enumerate(Theta, 1):

        z = a @ theta.T
        a = fun(z)    
        J_reg += np.sum(np.sum((theta[:, 1:] ** 2)))
        Z += [z]
        #  Add bias unit to the resulting array if it's not the last layer
        if layer != len(Theta):
            a = np.concatenate([np.ones((a.shape[0], 1)), a], axis=1)
            A += [a]
        

    J_reg *= lmbd/(2*m)
    J = np.sum(np.sum((-y * np.log(a)) - ((1 - y) * np.log(1 - a)))) / m
    J += J_reg
    
    # Backpropagation and Gradients
    error = a - y
    for layer, theta in misc.reverse_enumerate(Theta, 1):
        delta = error.T @ A[layer - 1]   
        theta_grad = (delta + lmbd * Theta[layer - 1]) / m
        # do not regularize bias units
        theta_grad[:, 0] = delta[:, 0] / m      
        Grad += [theta_grad]
        
        if layer != 1:
            error = error @ theta[:,1:] * fun_grad(Z[layer - 2])

    # Gradient has to be "unrolled" to a vector to correctly feed it to the optimization function
    grad = np.concatenate([np.reshape(x, (x.shape[0] * x.shape[1], 1)) for i, x in misc.reverse_enumerate(Grad)])
    grad = grad.flatten()

    return (J, grad)