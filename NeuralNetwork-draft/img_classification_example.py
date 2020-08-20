import numpy as np
import pandas as pd
from activation import tanh
from activation import tanhGradient
from activation import tanhReScal
import matplotlib.pyplot as plt
import os
from API import NNClassifier


X = pd.read_csv('testing/X.csv', header=None).to_numpy()
y = pd.read_csv('testing/y.csv', header=None).to_numpy()
## classes in this csv are labeled like 10,1,2,...,9 rather than 0 to 9, hence the mod(y, 10)
y = np.mod(y, 10)

clf = NNClassifier(lmbd=1, hidden_layer_sizes=[25], fun=tanh, fun_grad=tanhGradient, out_layer_fun=tanhReScal,
                   epsilon=0.12, method='Newton-CG', maxiter=15, disp=True, random_state=42)
clf.fit(X, y)

print("{}Classifier cost: {}".format(os.linesep, clf.cost))

pred = clf.predict(X)

print("Accuracy of the classifier: {}".format(np.mean([pred == y]) * 100))
input("{}Press Enter to continue...{}".format(os.linesep, os.linesep))

m, _ = X.shape
rng = np.random.default_rng()
rp = rng.permutation(m)

original_shape = (20,20)
num_classes = 10

for i in rp:
    # Display random exaples and check if classifier got them right
    print("Displaying Example Image")
    ex = X[i:i+1,:]
    im = np.reshape(ex, original_shape, order='F')
    plt.imshow(im*255, cmap='gray_r', vmin=0, vmax=255)

    pred = clf.predict(ex)
    print("Neural Network Classifier Prediction: {}".format(np.mod(pred, num_classes)))
    plt.show()
    input("Press Enter to continue...")

    
