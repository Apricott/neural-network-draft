import unittest 
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))

#print(os.getcwd())

import NeuralNetwork_draft
import activation 




def assertNumpyArraysEqual(this: np.ndarray, that:np.ndarray, atol: float = 1e-08, msg: str =''):
    '''
    modified from http://stackoverflow.com/a/15399475/5459638
    '''
    if this.shape != that.shape:
        print(msg)
        raise AssertionError("Shapes don't match")
    #if not np.allclose(this, that, atol):
    if not np.all(np.abs(this - that) <= atol):
        print(msg)
        raise AssertionError("Elements don't match!")


class TestActivationFunctions(unittest.TestCase):

    def test_sigmoid(self):
        self.assertAlmostEqual(activation.sigmoid(0), 0.5, msg="Wrong result!")
        self.assertAlmostEqual(activation.sigmoid(100), 1, msg="Wrong result!")
        self.assertAlmostEqual(activation.sigmoid(-100), 0, msg="Wrong result!")
        assertNumpyArraysEqual(activation.sigmoid(np.array([[0, 100, -100]])), np.array([[0.5, 1, 0]]), msg="Wrong result!")

    def test_sigmoidGradient(self):
        self.assertAlmostEqual(activation.sigmoidGradient(0), 0.25, msg="Wrong result!")
        self.assertAlmostEqual(activation.sigmoidGradient(100), 0, msg="Wrong result!")
        self.assertAlmostEqual(activation.sigmoidGradient(-100), 0, msg="Wrong result!")
        assertNumpyArraysEqual(activation.sigmoidGradient(np.array([[0, 100, -100]])), np.array([[0.25, 0, 0]]), msg="Wrong result!")
    

class TestNNPrediction(unittest.TestCase):

    def setUp(self):
        self.X = pd.read_csv('testing/X.csv', header=None).to_numpy()
        ## classes in this csv are indexed from 1 to 10 rather than 0 to 9, hence subtraction of 1
        self.y = (pd.read_csv('testing/y.csv', header=None).to_numpy() - 1).T
        self.Theta1 = pd.read_csv('testing/Theta1.csv', header=None).to_numpy()
        self.Theta2 = pd.read_csv('testing/Theta2.csv', header=None).to_numpy()
        self.pred = NeuralNetwork_draft.predict([self.Theta1, self.Theta2], self.X, activation.sigmoid)

    def test_predictions(self):
        self.assertAlmostEqual(np.mean([self.pred == self.y]) * 100, 97.5, 1, "Wrong prediction!")


class TestNNCostFunctions(unittest.TestCase):

    def setUp(self):
        self.X = pd.read_csv('testing/X.csv', header=None).to_numpy()
        ## classes in this csv are indexed from 1 to 10 rather than 0 to 9, hence subtraction of 1
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
        J, _ = NeuralNetwork_draft.nnCostFunction(self.nn_params, self.layer_sizes, self.num_classes, self.X, self.y, self.lmbd, activation.sigmoid, activation.sigmoidGradient)
        self.assertAlmostEqual(J, 0.287629, 3, "Wrong cost!")

    def test_costWithRegularization(self):
        self.lmbd = 1
        J, _ = NeuralNetwork_draft.nnCostFunction(self.nn_params, self.layer_sizes, self.num_classes, self.X, self.y, self.lmbd, activation.sigmoid, activation.sigmoidGradient)
        self.assertAlmostEqual(J, 0.383770, 3, "Wrong cost!")

    def test_grad(self):
        _, grad = NeuralNetwork_draft.nnCostFunction(self.nn_params, self.layer_sizes, self.num_classes, self.X, self.y, self.lmbd, activation.sigmoid, activation.sigmoidGradient)
        assertNumpyArraysEqual(grad, self.grad, atol=1e-02, msg="Wrong result!")

    def test_gradWithRegularization(self):
        self.lmbd = 1
        _, grad = NeuralNetwork_draft.nnCostFunction(self.nn_params, self.layer_sizes, self.num_classes, self.X, self.y, self.lmbd, activation.sigmoid, activation.sigmoidGradient)
        assertNumpyArraysEqual(grad, self.grad_reg, atol=1e-02, msg="Wrong result!")

if __name__ == '__main__':
    unittest.main()