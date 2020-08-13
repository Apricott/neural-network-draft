import numpy as np


def reverse_enumerate(L: list, stop: int = 0):
   """
   Returns generator iterating L in reversed direction
   yields index and L[index]
   """
   l = len(L)
   for i, n in enumerate(reversed(L)):
       yield l - i - 1 + stop, n

def randInitializeWeights(L_in: int, L_out: int, epsilon: float = 0.12, random_state = None) -> np.ndarray:
    """
    Randomly initialize the weights of a layer with L_in incoming connections and L_out outgoing connections

    Returns an array of size (L_out, L_in + 1), as the first column of the array handles the bias terms
    """

    rng = np.random.default_rng(seed = random_state)
    W = rng.random((L_out, L_in + 1)) * 2 * epsilon - epsilon

    return W

def reshapeTheta(nn_params: np.ndarray, layer_sizes: list) -> list:
    """
    reshape Theta arrays from nn_params array back to their original shape
    """
    Theta = []
    split = 0
    for i in range(0, len(layer_sizes) - 1):
        temp = layer_sizes[i + 1] * (layer_sizes[i] + 1) + split
        size = (layer_sizes[i + 1], layer_sizes[i] + 1)
        Theta += [np.reshape(nn_params[split:temp], size)]
        split = temp

    return Theta