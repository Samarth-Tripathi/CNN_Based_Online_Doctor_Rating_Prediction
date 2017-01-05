from numpy.lib.stride_tricks import as_strided as ast
import numpy as np 
import os
from sklearn.preprocessing import scale
#from sklearn.utils import shuffle

def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple, 
    even for one-dimensional shapes.
     
    Parameters
        shape - an int, or a tuple of ints
     
    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass
 
    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass
     
    raise TypeError('shape must be an int, or a tuple of ints')

def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
     
    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an 
                  extra dimension for each dimension of the input.
     
    Returns
        an array containing each n-dimensional window from a
    '''
     
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
     
    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)
     
     
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))
     
    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
     
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided
     
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)

windows_size = 40
stride_size = 7
import sys
import pickle
from random import shuffle 
def create():
    print 'preparing dataset...';
    X = [];
    Y = [];
    for directory, dirnames, filenames in os.walk('data_preprocessed_python'):
        for filename in filenames:
            file = os.path.join(directory,filename)
            data = pickle.load(open(file, 'rb'))
            x = data['data']
            y = data['labels']            
            print x.shape,y.shape
	    for trial,outcome in zip(x,y):
                #print trial.shape
		xl = sliding_window(trial, (40,windows_size), ss = (1, stride_size), flatten = True)
                #print xl.shape
		X.extend(xl)
                Y.extend(np.tile(outcome, (len(xl),1)))
#		break
#	    break
#	break
	    #print type(X), type(X[0])
	    #print len(X), X[0].shape, Y.shape
    print 'scaling...'

    print "shuffling..."
    Z = zip(X,Y)
    shuffle(Z)
    X, Y = zip(*Z) 
    #X = np.array(X)   
    #Y = np.asarray(Y)      
    sizef = len(X)/14
    print 'saving', len(X), 'datasets into 15 files of size', sizef 
    for i in range(0,15):
        print 'printint filei ', str(i)
	with open(str(i)+'_data.pic','wb+') as f:
	    pickle.dump((np.asarray(X[i*sizef:(i+1)*sizef], dtype=np.float32),np.asarray(Y[i*sizef:(i+1)*sizef], dtype=np.float32)), f, -1)

def load2d():
    from sklearn.preprocessing import scale
    
    x,y = None, None
    file = '1_data.pic';
    with open(file,'rb') as f:
        x,y = pickle.load(f);
    x = scale(x, axis = 0)
    y = scale(y, axis = 0)
    return x,y
#    for directory, dirnames, filenames in os.walk('processed'):
#	for filename in filenames:
#	    file = os.path.join(directory,filename)
#	    with open(file,'rb') as f:
#                x,y = pickle.load(f)
#		if X is None:
#		    X = x
#		    Y = y
#		else:
#		    X = np.concatenate([X,x])
#		    Y = np.concatenate([Y,y])
#    print 'X shape', X.shape
#    print 'Y shape', Y.shape
#    return X,Y
