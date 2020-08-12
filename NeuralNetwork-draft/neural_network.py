import numpy as np
import pd as pandas
from scipy.optimize import minimize
import NeuralNetwork_draft 
import activation
import misc



class NNClassifier(lmbd=0.0001, hidden_layer_sizes = tuple(100,), fun = activation.sigmoid, 
				   fun_grad = activation.sigmoidGradient,  epsilon=0.12, method='Newton-CG', random_state=None):

	def __init__():
		self.Theta = []
		self.layer_sizes = []
		self.nnCostFunc = NeuralNetwork_draft.nnCostFunction()
		#self.num_classes = 0

	def fit(self, X, y):
		"""

		"""
		if isinstance(X, pd.DataFrame):
			X = X.to_numpy()

		if isinstance(y, pd.DataFrame):
			y = y.to_numpy()

		m, n = X.shape()
		self.num_classes = y.unique()
		self.layer_sizes = [n] + [x for x in hidden_layer_size] + [1]

		self.Theta = [misc.randInitializeWeights(layer_sizes[x], layer_sizes[X + 1], epsilon, random_state) for x in layer_sizes[:-1]]
		self.nn_params = np.concatenate([np.reshape(x, (x.shape[0] * x.shape[1], 1)) for x in self.Theta])

		res = minimize(fun=self.nnCostFunc, 
				 x0=self.nn_params, 
				 args=(self.layer_sizes, self.num_classes, self.X, self.y, self.lmbd, self.fun, self.fun_grad), 
				 method=self.method,
				 jac=True)

		self.nn_paramas = res.x

		# reshape Theta arrays from nn_params array back to their original shape
		self.Theta = misc.reshapeTheta(nn_params, layer_sizes)


	def predict(self, X):
		"""

		"""



	