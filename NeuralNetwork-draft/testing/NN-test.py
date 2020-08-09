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
    
    #def setUp(self):

    def test_sigmoid(self):
        self.assertAlmostEqual(activation.sigmoid(0), 0.5, msg="Wrong result!")
        self.assertAlmostEqual(activation.sigmoid(100), 1, msg="Wrong result!")
        self.assertAlmostEqual(activation.sigmoid(-100), 0, msg="Wrong result!")
        assertNumpyArraysEqual(activation.sigmoid(np.array([[0, 100, -100]])), np.array([[0.5, 1, 0]]), msg="Wrong result!")
    



class TestNNPrediction(unittest.TestCase):

    def setUp(self):
        self.X = pd.read_csv('NeuralNetwork-draft/testing/X.csv').to_numpy()
        self.y = pd.read_csv('NeuralNetwork-draft/testing/y.csv').to_numpy()
        self.Theta1 = pd.read_csv('NeuralNetwork-draft/testing/Theta1.csv').to_numpy()
        self.Theta2 = pd.read_csv('NeuralNetwork-draft/testing/Theta2.csv').to_numpy()
        self.pred = NeuralNetwork_draft.predict(self.Theta1, self.Theta2, self.X)

    def test_predictions(self):
        self.assertAlmostEqual(np.mean([self.pred == self.y]) * 100, 97.5, 1, "prediction is wrong")

  
if __name__ == '__main__':
    unittest.main()