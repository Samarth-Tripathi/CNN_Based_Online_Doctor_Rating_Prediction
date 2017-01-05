import pickle as p
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import gzip
ss = None
with open('scale.pickle', 'rb') as f:
	ss = p.load(f)
def scale(whole):
	global ss
	whole = whole.reshape(-1, 40)
	whole = StandardScaler().fit_transform(whole);
	#whole = ss.transform(whole )
	whole = whole.reshape(-1,8064,40)
	return whole

def build():
	datafolder = os.path.join('//home','ubuntu','sud','deap','data_preprocessed_python')
	whole = None #np.empty()
	labels = None
	for file in os.listdir(datafolder):
		file = os.path.join(datafolder, file)
		dat = p.load(open(file,'rb'))
		if whole is None:
			whole = dat['data']
			labels = dat['labels']
		else:
			whole = np.concatenate((whole,dat['data']))
			labels =  np.concatenate((labels, dat['labels']))
			#break
	#procsesing labels for classification
	print whole.shape
	#print whole
	#whole = np.ndarray.tolist(whole)
	#labels = np.ndarray.tolist(labels)
	#trainIndex = int(len(whole)*0.9)
	whole = np.swapaxes(whole, 1,2)
	print 'after swaping', whole.shape
	#whole = whole.reshape(-1, 40*8064)
	whole = scale(whole)
	labels = StandardScaler().fit_transform(labels);
	labels = labels/abs(labels)
	#whole = whole.reshape(-1, 40)
	#ss = StandardScaler().fit(whole )
	#with open('scale.pickle', 'wb') as f:
	#	p.dump(ss,f)
	#whole = ss.transform(whole)
	#whole = whole.reshape(-1,8064,40)
	#trainIndex = int(whole.shape[0]*0.9)
	#np.savez_compressed('whole_data1.pickle',whole,labels)
	#with numpy.open('whole_data1.pickle','wb') as f:
	#	f.write((whole,labels))
		#p.dump([(whole[:trainIndex],labels[:trainIndex]), (whole[trainIndex:], labels[trainIndex:])],f)
	return whole, labels
def load():
	X, Y  = build()
	X = X.reshape(-1, 40*8064)	
	print 'mean', Y.mean(axis=0)
	return X, Y
def load_from_mem():
	data = np.load('whole_data1.pickle')
	return data['arr_0'],data['arr_1']
	with gzipopen('whole_data1.pickle','wb') as f:
		return f.read()
