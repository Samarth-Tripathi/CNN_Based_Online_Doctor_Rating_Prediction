# add to kfkd.py
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import sigmoid
net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 8064*40),  # 96x96 input pixels per batch
    hidden_num_units=200,  # number of units in hidden layer
    output_nonlinearity=sigmoid,  # output layer uses identity function
    output_num_units=1,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=200,  # we want to train this many epochs
    verbose=1,
    )
from load import load
import numpy as np
X, y = load()
X = X.astype(np.float32)
y = y[:,0].astype(np.float32)

print X.shape, y.shape
net1.fit(X, y)
import pickle as p
with open('mlp_model.pickle', 'wb') as f:
	p.dump(net1, f)


