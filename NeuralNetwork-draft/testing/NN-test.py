import unittest 
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import activation 
import NeuralNetwork_draft



def assertNumpyArraysEqual(this,that,msg=''):
    '''
    modified from http://stackoverflow.com/a/15399475/5459638
    '''
    if this.shape != that.shape:
        print(msg)
        raise AssertionError("Shapes don't match")
    if not np.allclose(this,that):
        print(msg)
        raise AssertionError("Elements don't match!")


class TestActivationFunctions(unittest.TestCase):

    def test_sigmoid(self):
        self.assertAlmostEqual(activation.sigmoid(0), 0.5, msg="Wrong result!")
        self.assertAlmostEqual(activation.sigmoid(100), 1, msg="Wrong result!")
        self.assertAlmostEqual(activation.sigmoid(-100), 0, msg="Wrong result!")
        assertNumpyArraysEqual(activation.sigmoid(np.array([[0, 100, -100]])), np.array([[0.5, 1, 0]]), msg="Wrong result!")
    

class TestNNPrediction(unittest.TestCase):

    def setUp(self):
        self.X = pd.read_csv('NeuralNetwork-draft/testing/X.csv', header=None).to_numpy()
        ## classes in this csv are indexed from 1 to 10 rather than 0 to 9, hence subtraction of 1
        self.y = (pd.read_csv('NeuralNetwork-draft/testing/y.csv', header=None).to_numpy() - 1).T
        self.Theta1 = pd.read_csv('NeuralNetwork-draft/testing/Theta1.csv', header=None).to_numpy()
        self.Theta2 = pd.read_csv('NeuralNetwork-draft/testing/Theta2.csv', header=None).to_numpy()
        self.pred = NeuralNetwork_draft.predict([self.Theta1, self.Theta2], self.X, activation.sigmoid)

    def test_predictions(self):
        self.assertAlmostEqual(np.mean([self.pred == self.y]) * 100, 97.5, 1, "Wrong prediction!")


class TestNNCostFunction(unittest.TestCase):

    def setUp(self):
        self.X = pd.read_csv('NeuralNetwork-draft/testing/X.csv', header=None).to_numpy()
        ## classes in this csv are indexed from 1 to 10 rather than 0 to 9, hence subtraction of 1
        self.y = (pd.read_csv('NeuralNetwork-draft/testing/y.csv', header=None).to_numpy() - 1).T
        self.Theta1 = pd.read_csv('NeuralNetwork-draft/testing/Theta1.csv', header=None).to_numpy()
        self.Theta2 = pd.read_csv('NeuralNetwork-draft/testing/Theta2.csv', header=None).to_numpy()
        self.nn_params = [0, 0]     # <- popraw! maja byc parametry do fmincg
        self.layer_sizes = [400, 25]
        self.num_classes = 10
        self.lmbd = 0

    def test_cost(self):
        J, _ = NeuralNetwork_draft.nnCostFunction(self.nn_params, self.layer_sizes, self.num_classes, self.X, self.y, self.lmbd)
        self.assertAlmostEqual(J, 0.287629, 3, "Wrong cost!")

    def test_costWithRegularization(self):
        self.lmbd = 1
        J, _ = NeuralNetwork_draft.nnCostFunction(self.nn_params, self.layer_sizes, self.num_classes, self.X, self.y, self.lmbd)

if __name__ == '__main__':
    unittest.main()