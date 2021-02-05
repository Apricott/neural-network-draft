import numpy as np
import pandas as pd
from copy import copy, deepcopy
from scipy.optimize import minimize
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

	alpha : float, default=1.
		Penalty factor for False Positives, keep in range [1, inf) 
    beta : float, default=1.
		Penalty factor for False Negatives, keep in range [1, inf) 
	threshold : float, default=None
		Prediction probabilities lower than the threshold will be considered as equal to 0.

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
	accuracy_score(self, pred_y, true_y) :
		Return the mean accuracy on the predicted and true labels, the best performance is 1.
	get_params(self, deep=False) :
		Return the flatten vector of weights for all network layers
	get_pred_cost(self, X, y, lmbd, alpha, beta) :
		Calculate cost of prediction

	"""

	def __init__(self, lmbd: float=0.0001, hidden_layer_sizes: list=[100], fun: object=sigmoid, fun_grad: object=sigmoidGradient, 
			  out_layer_fun: object=None, epsilon: float=0.12, alpha:float=1., beta:float=1., threshold:float=None, 
			  method: str='Newton-CG', maxiter: int=30, disp: bool=True, random_state: float=None):

		self._lmbd = lmbd
		self._hidden_layer_sizes = hidden_layer_sizes
		self._fun = fun
		self._fun_grad = fun_grad
		self._out_layer_fun = out_layer_fun
		self._epsilon = epsilon
		self._method = method
		self._maxiter = maxiter
		self._disp = disp
		self._random_state = random_state
		self._alpha = alpha
		self._beta = beta
		self._nn_params = None
		self._cost_fun = nn.nnCostFunction
		self._threshold = threshold 

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
		self._num_classes = len(np.unique(y))
		self.layer_sizes = [n] + [x for x in self._hidden_layer_sizes] + [self._num_classes]

		self.Theta = [misc.randInitializeWeights(self.layer_sizes[i], self.layer_sizes[i + 1], self._epsilon, self._random_state) 
				for i, x in enumerate(self.layer_sizes[:-1])]
		# "Unroll" weights to a vector to correctly feed them to the optimization function
		self._nn_params = np.concatenate([np.reshape(x, (x.shape[0] * x.shape[1], 1)) for x in self.Theta])

		options = {
			'maxiter': self._maxiter,
			'disp': self._disp
			}

		res = minimize(fun=self._cost_fun, 
				 x0=self._nn_params, 
				 args=(self.layer_sizes, self._num_classes, X, y, self._lmbd, self._fun, self._fun_grad, self._out_layer_fun, self._alpha, self._beta), 
				 method=self._method,
				 jac=True,
				 options=options)

		self._nn_params = res.x
		self.cost = res.fun
		# reshape Theta arrays from nn_params array back to their original shape
		self.Theta = misc.reshapeTheta(self._nn_params, self.layer_sizes)


	def predict(self, X) -> np.ndarray:
		"""
		Predict classes corresponding to the examples in X array using the multi-layer perceptron classifier

		"""

		if isinstance(X, pd.DataFrame):
			X = X.to_numpy()

		pred = nn.predict(Theta=self.Theta, X=X, fun=self._fun)

		return pred


	def accuracy_score(self, pred_y, true_y) -> float:
		"""
		Return the mean accuracy on the predicted and true labels, the best performance is 1.

		"""

		accuracy = np.mean([pred_y == true_y])

		return accuracy


	def get_params(self) -> np.ndarray:
		"""
		Return the flatten vector of weights for all network layers
		"""
		params = np.array(self._nn_params, copy=True)
		return params

	def get_pred_cost(self, X, y, lmbd, alpha, beta) -> float:
		"""
		Calculate cost of prediction
		"""
		J, _ = self.cost_fun(self._nn_params, self.layer_sizes, self._num_classes, X, y, 
					   self._lmbd, self._fun, self._fun_grad, self._out_layer_fun, self._alpha, self._beta)

		return J

	def __deepcopy__(self, memo) -> object : # memo is a dict of id's to copies
		"""
		Create deep copy of the classifier
		"""

		id_self = id(self)        # memoization avoids unnecesary recursion
		_copy = memo.get(id_self)
		if _copy is None:
			_copy = type(self)(
				deepcopy(self._lmbd, memo),
				deepcopy(self._hidden_layer_sizes, memo),
				copy(self._fun),
				copy(self._fun_grad),
				copy(self._out_layer_fun, memo),
				deepcopy(self._epsilon, memo),
				deepcopy(self._method, memo),
				deepcopy(self._maxiter, memo),
				deepcopy(self._disp, memo),
				deepcopy(self._random_state, memo),
				deepcopy(self._alpha, memo),
				deepcopy(self._beta, memo))
			memo[id_self] = _copy 
		return _copy