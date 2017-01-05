# add to kfkd.py
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import softmax
import theano
from numpy import float32
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 8064, 40),
    conv1_num_filters=32, conv1_filter_size=(300, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(200, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(200, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=4,

    #update_learning_rate=0.01,
    #update_momentum=0.9,

    regression=True,
    #batch_iterator_train=BatchIterator(batch_size=128),
    max_epochs=1,
    verbose=1,
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],    
    )
from load import load
import numpy as np

#for file in dir:
#(X, y), (xtest,ytest) = load()  # load 2-d data
X = np.random.rand(5,1,8064,40).astype(np.float32)
y = np.random.rand(5,4).astype(np.float32)
print X.shape
print 'rehaping '
#X = X.reshape(-1,1,40,8064)
print X.shape, y.shape
net2.fit(X, y)



# Training for 1000 epochs will take a while.  We'll pickle the
# trained model so that we can load it back later:
import cPickle as pickle
with open('cnn1.pickle', 'wb') as f:
    pickle.dump(net2, f, -1)
