import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from neural_network_draft.activation import ReLU, ReLUGradient, softmax
from neural_network_draft.API import NNClassifier

X = pd.read_csv('testing/X.csv', header=None).to_numpy()
y = pd.read_csv('testing/y.csv', header=None).to_numpy()
## classes in this csv are labeled like 10,1,2,...,9 rather than 0 to 9, hence the mod(y, 10)

y = np.mod(y, 10)

clf = NNClassifier(lmbd=1, hidden_layer_sizes=[25], fun=ReLU, fun_grad=ReLUGradient, out_layer_fun=softmax,
                   epsilon=0.12, alpha=1., beta=1., threshold=None, method='Newton-CG', maxiter=15, disp=True, random_state=42)
clf.fit(X, y)

print("{}Classifier cost: {}".format(os.linesep, clf.cost))

pred = clf.predict(X)
score = clf.accuracy_score(pred, y)

print("Accuracy of the classifier: {}%".format(score*100))
input("{}Press Enter to continue...{}".format(os.linesep, os.linesep))

m, _ = X.shape
rng = np.random.default_rng()
rp = rng.permutation(m)

original_shape = (20,20)

for i in rp:
    # Display random exaples and check if classifier got them right
    print("Displaying Example Image")
    ex = X[i:i+1,:]
    im = np.reshape(ex, original_shape, order='F')
    plt.imshow(im*255, cmap='gray_r', vmin=0, vmax=255)

    pred = clf.predict(ex)
    print("Neural Network Classifier Prediction: {}".format(pred))
    plt.show()
    input("Press Enter to continue...")

    
