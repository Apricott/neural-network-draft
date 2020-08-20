import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import uniform
from neural_network_draft import neural_network as nn
from neural_network_draft.activation import sigmoidGradient
from neural_network_draft.activation import sigmoid
from neural_network_draft import misc



class NNClassifier:
	"""
	Creates Classifier object used to train neural network and make predictions of classes in data

	Parameters:
	lmbd : float, default=0.0001 
		Regularzation parameter.
	hidden_layer_sizes : list, length=[number of layers]-2, default=[100]
		The ith element represents the number of neurons in the ith layer.
	fun : object, default=activation.sigmoid
		Activation function for the hidden layer. 
	fun_grad : object, default=activation.sigmoidGradient
		Gradient of the activation function used to calculate backpropagation.
	out_layer_activation: object, default=None
		Activation function used for the output layer
	epsilon : float, default=0.12
		Randomly initialized weights will be in the range [0, epsilon)
	method : str, default='Newton-CG'
		Type of solver. For all available types check the documentation of scipy.optimize.minimize function.
	maxiter : int, default=30
		Maximum number of solving function iterations.
	disp : bool, default=False
		Set visibility of solver messages
	random_state : int, RandomState instance, default=None
		Pass an int for reproducible results across multiple calls.

	Attributes:
	Theta : list, length=[number of layers]-1
		The ith element represents the weight matrix corresponding to layer i.
	layer_sizes : list, length=[number of layers]
		Number of neurons in each layer.
	cost : float
		Value of the cost function.

	Methods:
	fit(X, y) :
		Fit the model to data array X and target(s) y.
	predict(X) :	
		Predict classes of data in array X using the multi-layer perceptron classifier.

	"""

	def __init__(self, lmbd: float=0.0001, hidden_layer_sizes: list=[100], fun: object=sigmoid, fun_grad: object=sigmoidGradient, 
			  out_layer_fun: object=None, epsilon: float=0.12, method: str='Newton-CG', maxiter:int=30, disp:bool=True, random_state: float=None):
		self.lmbd = lmbd
		self.hidden_layer_sizes = hidden_layer_sizes
		self.fun = fun
		self.fun_grad = fun_grad
		self.out_layer_fun = out_layer_fun
		self.epsilon = epsilon
		self.method = method
		self.maxiter = maxiter
		self.disp = disp
		self.random_state = random_state

		self.Theta = []
		self.layer_sizes = []
		self.cost = 0

	def fit(self, X, y):
		"""
		Fit the model to data matrix X and target(s) y.

		"""
		if isinstance(X, pd.DataFrame):
			X = X.to_numpy()

		if isinstance(y, pd.DataFrame):
			y = y.to_numpy()

		m, n = X.shape
		self.num_classes = len(np.unique(y))
		self.layer_sizes = [n] + [x for x in self.hidden_layer_sizes] + [self.num_classes]

		self.Theta = [misc.randInitializeWeights(self.layer_sizes[i], self.layer_sizes[i + 1], self.epsilon, self.random_state) for i, x in enumerate(self.layer_sizes[:-1])]
		# "Unroll" weights to a vector to correctly feed them to the optimization function
		self.nn_params = np.concatenate([np.reshape(x, (x.shape[0] * x.shape[1], 1)) for x in self.Theta])

		options = {
			'maxiter': self.maxiter,
			'disp': self.disp
			}

		res = minimize(fun=nn.nnCostFunction, 
				 x0=self.nn_params, 
				 args=(self.layer_sizes, self.num_classes, X, y, self.lmbd, self.fun, self.fun_grad, self.out_layer_fun), 
				 method=self.method,
				 jac=True,
				 options=options)

		self.nn_params = res.x
		self.cost = res.fun
		# reshape Theta arrays from nn_params array back to their original shape
		self.Theta = misc.reshapeTheta(self.nn_params, self.layer_sizes)


	def predict(self, X):
		"""
		Predict classes corresponding to the examples in X array using the multi-layer perceptron classifier

		"""

		if isinstance(X, pd.DataFrame):
			X = X.to_numpy()


		pred = nn.predict(Theta=self.Theta, X=X, fun=self.fun)

		return pred

	