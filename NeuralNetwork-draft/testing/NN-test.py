import unittest 
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from neural_network_draft import neural_network, API, activation, misc

"""
Most of these tests are based on Stanford Machine Learning course
https://www.coursera.org/learn/machine-learning

"""

def assertNumpyArraysEqual(this: np.ndarray, that:np.ndarray, atol: float = 1e-08, msg: str =''):
    '''
    modified from http://stackoverflow.com/a/15399475/5459638
    '''
    if this.shape != that.shape:
        print(msg)
        raise AssertionError("Shapes don't match")
    if not np.all(np.abs(this - that) <= atol):
        print(msg)
        raise AssertionError("Elements don't match!")


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.X = pd.read_csv('testing/X.csv', header=None).to_numpy()
        ## classes in this csv are labeled like 10,1,2,...,9 rather than 0 to 9, hence subtraction of 1
        self.y = pd.read_csv('testing/y.csv', header=None).to_numpy() - 1 
        self.Theta1 = pd.read_csv('testing/Theta1_trained.csv', header=None).to_numpy()
        self.Theta2 = pd.read_csv('testing/Theta2_trained.csv', header=None).to_numpy()
        self.nn_params = np.concatenate([np.reshape(self.Theta1, (25 * 401, 1)), np.reshape(self.Theta2, (10 * 26, 1))])
        self.hidden_layer_sizes = [25]
        self.num_classes = 10
        self.lmbd = 1
        self.accuracy = 95.000000 

    def test_fit(self):
        clf = API.NNClassifier(self.lmbd, self.hidden_layer_sizes, activation.sigmoid,
                              activation.sigmoidGradient, epsilon=0.12, method='Newton-CG', maxiter=15, disp=False, random_state=42)
        res = clf.fit(self.X, self.y)
        
        Theta1 = clf.Theta[0]
        Theta2 = clf.Theta[1]
        # Doesn't have to be exactly same as test Theta was calculated using other optimization function, what resulted in slightly different parameters
        assertNumpyArraysEqual(Theta1, self.Theta1, atol=10, msg="Different Theta1!")
        assertNumpyArraysEqual(Theta2, self.Theta2, atol=10, msg="Different Theta2!")

    def test_predict(self):
        clf = API.NNClassifier(self.lmbd, self.hidden_layer_sizes, activation.sigmoid,
                              activation.sigmoidGradient, epsilon=0.12, method='Newton-CG', maxiter=15, disp=False, random_state=42)
        clf.fit(self.X, self.y)

        pred = clf.predict(self.X)
        self.assertAlmostEqual(np.mean([pred == self.y]) * 100, self.accuracy, places=0, msg="Different accuracy!")


class TestMisc(unittest.TestCase):
    
    def test_reverse_enumerate(self):
        lst = [1, 2, 3, 4]
        self.assertListEqual([x for i, x in misc.reverse_enumerate(lst)],[4, 3, 2, 1], "Wrong elements and/or order!")
        self.assertListEqual([i for i, x in misc.reverse_enumerate(lst, 1)],[4, 3, 2, 1], "Wrong index order! (from 1)")
        self.assertListEqual([i for i, x in misc.reverse_enumerate(lst, 0)],[3, 2, 1, 0], "Wrong index order! (from 0)")

    def test_randInitializeWeights(self):
        eps = 0.12
        W = misc.randInitializeWeights(4, 2, epsilon = eps, random_state=1)
        test = np.zeros((2, 5))
        assertNumpyArraysEqual(W, test, eps)


class TestActivationFunctions(unittest.TestCase):

    def test_sigmoid(self):
        self.assertAlmostEqual(activation.sigmoid(0), 0.5, msg="Wrong result! (at 0)")
        self.assertAlmostEqual(activation.sigmoid(100), 1, msg="Wrong result! (at 100)")
        self.assertAlmostEqual(activation.sigmoid(-100), 0, msg="Wrong result! (at -100)")
        assertNumpyArraysEqual(activation.sigmoid(np.array([[0, 100, -100]])), np.array([[0.5, 1, 0]]), msg="Wrong result! (on np.ndarray)")

    def test_sigmoidGradient(self):
        self.assertAlmostEqual(activation.sigmoidGradient(0), 0.25, msg="Wrong result! (at 0)")
        self.assertAlmostEqual(activation.sigmoidGradient(100), 0, msg="Wrong result! (at 100)")
        self.assertAlmostEqual(activation.sigmoidGradient(-100), 0, msg="Wrong result! (at -100)")
        assertNumpyArraysEqual(activation.sigmoidGradient(np.array([[0, 100, -100]])), np.array([[0.25, 0, 0]]), msg="Wrong result! (on np.ndarray)")

    def test_tanh(self):
        self.assertAlmostEqual(activation.tanh(0), 0., msg="Wrong result! (at 0)")
        self.assertAlmostEqual(activation.tanh(100), 1, msg="Wrong result! (at 100)")
        self.assertAlmostEqual(activation.tanh(-100), -1., msg="Wrong result! (at -100)")
        assertNumpyArraysEqual(activation.tanh(np.array([[0, 100, -100]])), np.array([[0, 1., -1.]]), msg="Wrong result! (on np.ndarray)")

    def test_tanhGradient(self):
        self.assertAlmostEqual(activation.tanhGradient(0), 1., msg="Wrong result! (at 0)")
        self.assertAlmostEqual(activation.tanhGradient(100), 0, msg="Wrong result! (at 100)")
        self.assertAlmostEqual(activation.tanhGradient(-100), 0., msg="Wrong result! (at -100)")
        assertNumpyArraysEqual(activation.tanhGradient(np.array([[0, 100, -100]])), np.array([[1, 0., 0.]]), msg="Wrong result! (on np.ndarray)")

    def test_tanhReScal(self):
        self.assertAlmostEqual(activation.tanhReScal(0), 0.5, msg="Wrong result! (at 0)")
        self.assertAlmostEqual(activation.tanhReScal(1), 1, msg="Wrong result! (at 1)")
        self.assertAlmostEqual(activation.tanhReScal(-1), 0., msg="Wrong result! (at -1)")
        assertNumpyArraysEqual(activation.tanhReScal(np.array([[0, 1, -1]])), np.array([[0.5, 1., 0.]]), msg="Wrong result! (on np.ndarray)")

    def test_softmax(self):
        import scipy.special as scsp
        assertNumpyArraysEqual(activation.softmax(np.array([[0, 1, -1]])), scsp.softmax(np.array([[0, 1, -1]])), msg="Wrong result! (on np.ndarray)")
    

class TestNNPrediction(unittest.TestCase):

    def setUp(self):
        self.X = pd.read_csv('testing/X.csv', header=None).to_numpy()
        ## classes in this csv are labeled like 10,1,2,...,9 rather than 0 to 9, hence subtraction of 1
        self.y = pd.read_csv('testing/y.csv', header=None).to_numpy() - 1
        self.Theta1 = pd.read_csv('testing/Theta1.csv', header=None).to_numpy()
        self.Theta2 = pd.read_csv('testing/Theta2.csv', header=None).to_numpy()

    def test_predictions(self):
        pred = neural_network.predict([self.Theta1, self.Theta2], self.X, activation.sigmoid)
        self.assertAlmostEqual(np.mean([pred == self.y]) * 100, 97.5, 1, "Wrong prediction!")


class TestNNCostFunction(unittest.TestCase):

    def setUp(self):
        self.X = pd.read_csv('testing/X.csv', header=None).to_numpy()
        ## classes in this csv are labeled like 10,1,2,...,9 rather than 0 to 9, hence subtraction of 1
        self.y = pd.read_csv('testing/y.csv', header=None).to_numpy() - 1
        self.Theta1 = pd.read_csv('testing/Theta1.csv', header=None).to_numpy()
        self.Theta2 = pd.read_csv('testing/Theta2.csv', header=None).to_numpy()
        self.grad = pd.read_csv('testing/grad.csv', header=None).to_numpy()
        self.grad_reg = pd.read_csv('testing/grad_reg.csv', header=None).to_numpy()
        self.nn_params = np.concatenate([np.reshape(self.Theta1, (25 * 401, 1)), np.reshape(self.Theta2, (10 * 26, 1))])
        self.layer_sizes = [400, 25, 10]
        self.num_classes = 10
        self.lmbd = 0

    def test_cost(self):
        J, _ = neural_network.nnCostFunction(self.nn_params, self.layer_sizes, self.num_classes, self.X, self.y, self.lmbd, activation.sigmoid, activation.sigmoidGradient)
        self.assertAlmostEqual(J, 0.287629, 3, "Wrong cost!")

    def test_costWithRegularization(self):
        self.lmbd = 1
        J, _ = neural_network.nnCostFunction(self.nn_params, self.layer_sizes, self.num_classes, self.X, self.y, self.lmbd, activation.sigmoid, activation.sigmoidGradient)
        self.assertAlmostEqual(J, 0.383770, 3, "Wrong cost! (with regularization)")

    def test_grad(self):
        _, grad = neural_network.nnCostFunction(self.nn_params, self.layer_sizes, self.num_classes, self.X, self.y, self.lmbd, activation.sigmoid, activation.sigmoidGradient)
        grad = grad.reshape((grad.shape[0], 1))
        assertNumpyArraysEqual(grad, self.grad, atol=1e-02, msg="Wrong gradient!")

    def test_gradWithRegularization(self):
        self.lmbd = 1
        _, grad = neural_network.nnCostFunction(self.nn_params, self.layer_sizes, self.num_classes, self.X, self.y, self.lmbd, activation.sigmoid, activation.sigmoidGradient)
        grad = grad.reshape((grad.shape[0], 1))
        assertNumpyArraysEqual(grad, self.grad_reg, atol=1e-02, msg="Wrong gradient! (with regularization)")


if __name__ == '__main__':
    unittest.main()