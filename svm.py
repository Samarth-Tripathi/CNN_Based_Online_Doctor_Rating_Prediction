from load import *
from sklearn import cross_validation
def svm():
	X, Y_ = load()
	#for i in range(0,1):
	#print 'emotion number ',i

	#row_count = X.shape[0];
	#tr_count = int(row_count *0.9);
	#nntrX,nnteX,nntrY,nnteY =cross_validation.train_test_split(X,Y, test_size = 0.1, random_state=0)
	#print nntrX.shape
	#print nntrY.shape
	from sklearn import svm
	for col in range(0,4):
		Y = Y_[:,col]
		for kernel in ('linear', 'rbf'):
			print kernel, col
			clf = svm.SVC(kernel=kernel, gamma=1.0, coef0=1.0, degree=5)
			#clf.fit(nntrX, nntrY)
			scores = cross_validation.cross_val_score(clf, X, Y, cv=10)
			print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
			#k=(np.mean((nnteY) == clf.predict(nnteX)))
			#print k*100

svm()
