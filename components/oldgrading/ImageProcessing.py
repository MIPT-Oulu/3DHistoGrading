#Image processing module, contains bilinear interpolation, lbp and MRELBP

#Import important modules before using
import numpy as np
import scipy.signal
import scipy.ndimage

import sklearn.decomposition as skdec
import sklearn.linear_model as sklin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn import neighbors

#Bilinear interpolation
def imbilinear(im,col,x,row,y):
	#Takes bilinear interpotalion from image
	#Starts from coordinates [y,x], ends at row,col
	x1 = int(np.floor(x))
	x2 = int(np.ceil(x))
	y1 = int(np.floor(y))
	y2 = int(np.ceil(y))
	Q11 = im[y2:y2+row,x1:x1+col]
	Q21 = im[y2:y2+row,x2:x2+col]
	Q12 = im[y1:y1+row,x1:x1+col]
	Q22 = im[y1:y1+row,x2:x2+col]
	R1 = ((x2-x)/(x2-x1+1e-06))*Q11+((x-x1)/(x2-x1+1e-06))*Q21
	R2 = ((x2-x)/(x2-x1+1e-06))*Q12+((x-x1)/(x2-x1+1e-06))*Q22
	P = ((y2-y)/(y2-y1+1e-06))*R1+((y-y1)/(y2-y1+1e-06))*R2
	return P

#Gaussian kernel
def Gauss2D(w,sigma):
	#Generates 2d gaussian kernel
	kernel = np.zeros((w,w))
	#Constant for centering
	r = (w-1)/2
	for ii in range(w):
		for jj in range(w):
			x = -((ii-r)**2+(jj-r)**2)/(2*sigma**2)
			kernel[ii,jj] = np.exp(x)
	#Normalizing the kernel
	kernel = 1/np.sum(kernel)*kernel
	return kernel

#LBP
def lbp(im,r,n):
	#Takes lbp from numpy array
	#Uses n neighbours from radius r
	
	#center pixels
	center = im[r:-r,r:-r]
	row,col = np.shape(center)
	#Differences	
	
	#Empty array for pattern
	pattern = np.zeros((row,col))
	
	#LBP
	pi = np.pi	
	for k in range(n):
		#print(k)
		diff = np.empty((x,y))
		theta = 0+k*(1*2*pi/n)
		x = r+r*np.cos(theta)
		y = r+r*np.sin(theta)		
		#Check if bilinear interpolation is needed
		if abs(x-round(x))<1e-06 and abs(y-round(y))<1e-06:
			x = int(round(x))
			y = int(round(y))
			P = im[y:y+row,x:x+col]	
		else:			
			P = imbilinear(im,col,x,row,y)
		#Difference between neighbour and center pixel		
		diff = P[0:y,0:x] - center
		pattern = pattern+(diff>=0)*2**(k*(diff>=0))	
	
		#Empty histogram
		hist = np.zeros((1,2**n))
		#Binning
		for k in range(2**n):
			hist[0,k] = np.sum(pattern==k)
		hist = 1/np.linalg.norm(hist)*hist
		return hist
		
#MRELBP
def MRELBP(im,N,R,r,w_c,w_r):
	#Takes Median Robust Extended Local Binary Pattern from image im
	#Uses N neighbours from radii R and r, R must be larger than r
	#Median filter uses kernel sizes w_c for center pixels, w_r[0] for larger radius and w_r[1]
	#for smaller radius	
	#Grayscale values are centered at their mean and scales with global standad deviation
	
	#Mean grayscale value and std
	muI = im.mean()
	stdI = im.std()
	
	#Centering and scaling with std
	I = (im-muI)/stdI
	
	#Median filtering
	Ic = scipy.signal.medfilt(I,w_c)	
	#Center pixels
	d = round(R+(w_r[0]-1)/2)
	Ic = Ic[d:-d,d:-d]
	#Subtracting the mean pixel value from center pixels
	Ic = Ic-I.mean()
	#Bining center pixels
	Chist = np.zeros((1,2))
	Chist[0,0] = np.sum(Ic>=0)
	Chist[0,1] = np.sum(Ic<0)	
	
	#Median filtered images for large and small radius
	IL = scipy.signal.medfilt(I,w_r[0])
	d1 = round((w_r[0]-1)/2)
	IL = IL[d1:-d1,d1:-d1]
	IS = scipy.signal.medfilt2d(I,w_r[1])
	d2 = round((w_r[1]-1)/2)
	IS = IS[d2:-d2,d2:-d2]
	
	#Neighbours
	pi = np.pi
	#Empty arrays for the neighbours
	row,col = np.shape(Ic)
	NL = np.zeros((row,col,N))
	NS = np.zeros((row,col,N))
	for k in range(N):
		#Angle to the neighbour
		theta = 0+k*(-1*2*pi/N)
		#Large neighbourhood
		x = R+R*np.cos(theta)
		y = R+R*np.sin(theta)
		if abs(x-round(x)) < 1e-06 and abs(y-round(y)) < 1e-06:
			x = int(round(x))
			y = int(round(y))
			P = IL[y:y+row,x:x+col]
		else:
			P = imbilinear(IL,col,x,row,y)
		NL[:,:,k] = P
		#Small neighbourhood
		x = r+r*np.cos(theta)
		y = r+r*np.sin(theta)
		if abs(x-round(x)) < 1e-06 and abs(y-round(y)) < 1e-06:
			x = int(round(x))
			y = int(round(y))
			P = IS[y:y+row,x:x+col]
		else:
			P = imbilinear(IS,col,x,row,y)
		NS[:,:,k] = P
	#Thresholding
	
	#Thresholding radial neighbourhood
	NR = NL-NS
	
	#Subtraction of means
	#Large neighbourhood
	NLmu = NL.mean(axis=2)		
	#Small neighbouhood
	NSmu = NS.mean(axis=2)
	
	for k in range(N):
		NL[:,:,k] = NL[:,:,k]-NLmu
		NS[:,:,k] = NS[:,:,k]-NSmu	
	
	#Converting to binary images and taking the lbp values
	
	#Initialization of arrays
	lbpIL = np.zeros((row,col))
	lbpIS = np.zeros((row,col))
	lbpIR = np.zeros((row,col))
	
	for k in range(N):
		lbpIL = lbpIL+(NL[:,:,k]>=0)*2**(k*(NL[:,:,k]>=0))
		lbpIS = lbpIS+(NS[:,:,k]>=0)*2**(k*(NS[:,:,k]>=0))
		lbpIR = lbpIR+(NR[:,:,k]>=0)*2**(k*(NR[:,:,k]>=0))
	
	#Binning
	Lhist = np.zeros((1,2**N))
	Shist = np.zeros((1,2**N))
	Rhist = np.zeros((1,2**N))
	for k in range(2**N):
		Lhist[0,k] = np.sum(lbpIL==k)
		Shist[0,k] = np.sum(lbpIS==k)
		Rhist[0,k] = np.sum(lbpIR==k)
	
	#Chist = 1/np.linalg.norm(Chist)*Chist
	#Lhist = 1/np.linalg.norm(Lhist)*Lhist
	#Shist = 1/np.linalg.norm(Shist)*Shist
	#Rhist = 1/np.linalg.norm(Rhist)*Rhist
	return Chist,Lhist,Shist,Rhist

#Mapping	
def getmapping(N):
	#Defines rotation invariant uniform mapping for lbp of N neighbours	
	newMax = N + 2
	table = np.zeros((1,2**N))
	for k in range(2**N):
		#Binary representation of bin number
		binrep = np.binary_repr(k,N)
		#Convert string to list of digits
		i_bin = np.zeros((1,len(binrep)))
		for ii in range(len(binrep)):
			i_bin[0,ii] = int(float(binrep[ii]))
		#Rotation
		j_bin = np.roll(i_bin,-1)
		#uniformity
		numt = np.sum(i_bin!=j_bin)		
		#Binning
		if numt <= 2:
			b = np.binary_repr(k,N)
			c=0
			for ii in range(len(b)):
				c = c+int(float(b[ii]))
			table[0,k] = c
		else:
			table[0,k] = N+1
	#num = newMax
	return table

#Apply mapping to lbp
def maplbp(bin,mapping):
	#Applies mapping to lbp bin
	#Number of bins in output
	N = int(np.max(mapping))
	#Empty array
	outbin = np.zeros((1,N+1))
	for k in range(N+1):
		#RIU indices
		M = mapping==k
		#Extract indices from original bin to new bin
		outbin[0,k] = np.sum(M*bin)
	return outbin

#Scikit PCA
def ScikitPCA(features,ncomp):
	pca = skdec.PCA(n_components=ncomp)
	score = pca.fit(features.T).transform(features.T)
	return score
	
#Principal component analysis
def PCA(features,ncomp):	
	#Feature dimension, x=num variables,N=num observations
	x,N = np.shape(features)		
	#Mean feature
	mean_f = np.mean(features,axis=1)
	#Centering
	centrd = np.zeros((x,N))
	for k in range(N):
		centrd[:,k] = features[:,k]-mean_f
	
	#PCs from covariance matrix if N>=x, svd otherwise
	if x<=N:
		#Covariance matrix
		Cov = np.zeros((x,x))
		f = np.zeros((x,1))
		for k in range(N):		
			f[:,0] = centrd[:,k]
			Cov = Cov+1/N*np.matmul(f,f.T)
			
		#Eigen values
		E,V = np.linalg.eig(Cov)		
		#Sort eigenvalues and vectors to descending order
		idx = np.argsort(E)[::-1]
		V = np.matrix(V[:,idx])
		E = E[idx]
		
		for k in range(ncomp):						
			s = np.matmul(V[:,k].T,centrd).T			
			try:
				score = np.concatenate((score,s),axis=1)
			except NameError:
				score = s
			p = V[:,k]
			try:
				pcomp = np.concatenate((pcomp,p),axis=1)
			except NameError:
				pcomp = p
	else:
		#PCA with SVD
		u,s,v = np.linalg.svd(centrd,compute_uv=1)
		pcomp = v[:,:ncomp]
		score = np.matmul(u,s).T[:,1:ncomp]		
	return pcomp,score
	
#Adaptive grayscale centering
def AGCentering(im,w):
	#Takes mean value in the window and subtracts it from the pixels within the window
	#Scales grayscale values to range [0,1]
	#w must be odd
	
	#Image size
	row,col = np.shape(im)
	#Empty image for output
	new_im = np.zeros((row,col))
	r = int((w-1)/2)
	for ii in range(row-2*r):
		for jj in range(col-2*r):
			y = r+1+ii			
			x = r+1+jj			
			new_im[y-r:y+r,x-r:x+r] = im[y-r:y+r,x-r:x+r]-np.mean(im[y-r:y+r,x-r:x+r])
			
	#Scales grayscale values to range[0 1]
	Gmax = np.max(new_im)
	Gmin = np.min(new_im)
	new_im = (new_im-Gmin)/(Gmax-Gmin)
	return new_im
	
#Local grayscale normalization normalization
def localnorm(im,w,sigma):
	#Centers grayscales with Gaussian weighted mean
	#Gaussian kernel	
	kernel = Gauss2D(w,sigma)
	#Blurring
	blurred = scipy.ndimage.convolve(im,kernel)
	#Normalizing with the blurred image
	new_im = im-blurred
	return new_im

#Local grayscale standardization
def localstandard(im,w1,w2,sigma1,sigma2):
	#Centers grayscales with Gaussian weighted mean
	#Gaussian kernels
	kernel1 = Gauss2D(w1,sigma1)
	kernel2 = Gauss2D(w2,sigma2)		
	#Blurring
	blurred1 = scipy.ndimage.convolve(im,kernel1)
	blurred2 = scipy.ndimage.convolve(im,kernel2)
	#Centering grayscale values
	centered = im-blurred1
	#Standardization
	std = (scipy.ndimage.convolve(centered**2,kernel2))**0.5
	new_im = centered/(std+1e-09)
	return new_im

#Subimage generation
def subimage(im,w,num):
	#Splits the image in num*num subimage of size 2w x 2w
	#Image size
	row,col = np.shape(im)
	imageset = np.zeros((2*w,2*w,num**2))
	#Image counter
	c = 0
	for ky in range(num):
		for kx in range(num):
			stepx = int((col-2*w)/(num-1))
			x1 = kx*stepx
			x2 = x1+2*w
			stepy = int((row-2*w)/(num-1))
			y1 = ky*stepy
			y2 = y1+2*w
			imageset[:,:,c] = im[y1:y2,x1:x2]
			c=c+1
	return imageset
	
#Regression
def regress(features,score):
	pred = []
	#Leave one out split
	loo = LeaveOneOut()	
	for trainidx, testidx in loo.split(features):
		#Indices
		X_train, X_test = features[trainidx], features[testidx]
		X_test -= X_train.mean(0)
		X_train -= X_train.mean(0)
		
		Y_train, Y_test = score[trainidx], score[testidx]		
		#Linear regression		
		regr = sklin.Ridge(alpha=1)
		regr.fit(X_train,Y_train)
		#Predicted score		
		pred.append(regr.predict(X_test))
	
	return np.array(pred)
	

#Logistic regression
def logreg(features,score):
	pred = []
	#Leave one out split
	loo = LeaveOneOut()	
	for trainidx, testidx in loo.split(features):
		#Indices
		X_train, X_test = features[trainidx], features[testidx]
		X_test -= X_train.mean(0)
		X_train -= X_train.mean(0)
		
		Y_train, Y_test = score[trainidx], score[testidx]		
		#Linear regression
		regr = sklin.LogisticRegression(solver='newton-cg',max_iter=1000)
		regr.fit(X_train,Y_train)
		#Predicted score
		P = regr.predict_proba(X_test)
		pred.append(P)
		
	pred = np.array(pred)
	pred = pred[:,:,1]
	return pred.flatten()
#Linear regression with LOO and KFOLD
def KFregress(features,score,fold):
	row,col = np.shape(features)
	fmean = np.mean(features,axis=1)
	for k in range(col):
		features[:,k] = features[:,k] - fmean.T
	pred = np.zeros((row,1))
	#Leave one out split
	loo = LeaveOneOut()	
	for trainidx, testidx in loo.split(features):
		#Indices
		X_train, X_test = features[trainidx], features[testidx]
		Y_train, Y_test = score[trainidx], score[testidx]
		#KFold split
		kpred = np.zeros((int(len(X_train)/fold),1))
		kf = KFold(n_splits = fold)
		c=0
		for ktrain,ktest in kf.split(X_train):
			X_Ktrain = X_train[ktrain]
			Y_Ktrain = Y_train[ktrain]
			#Linear regression
			regr = sklin.LinearRegression()
			regr.fit(X_Ktrain,Y_Ktrain)
			#Predicted score		
			kpred[c,0] = regr.predict(X_test)
			c=c+1
		pred[testidx,0] = np.mean(kpred,axis=0)
	return pred

'''
#LDA
def LDA(features,score):	
	row,col = np.shape(features)
	fmean = np.mean(features,axis=1)
	for k in range(col):
		features[:,k] = features[:,k] - fmean.T
	pred = np.zeros((row,1))
	#Leave one out split
	loo = LeaveOneOut()	
	for trainidx, testidx in loo.split(features):
		#Indices
		X_train, X_test = features[trainidx], features[testidx]
		Y_train, Y_test = score[trainidx], score[testidx]
		clf = LinearDiscriminantAnalysis()
		clf.fit(X_train,Y_train)
		pred[testidx,0] = clf.predict(X_test)
	return pred
'''

#Random forest
def RF(features,score,trees,d):
	score = np.ravel(score)
	row,col = np.shape(features)	
	pred = np.zeros((row,1))
	#Leaveone out split
	loo = LeaveOneOut()
	for trainidx,testidx in loo.split(features):
	#Indices
		X_train, X_test = features[trainidx], features[testidx]
		Y_train, Y_test = score[trainidx], score[testidx]		
		#Random Forest
		rf = RandomForestClassifier(n_estimators=trees,n_jobs = 4,max_depth=d,random_state=42)
		rf.fit(X_train,Y_train)
		#Prediction
		pred[testidx,0] = rf.predict(X_test)
	return pred

def augregression(testf,features,score):
	pred = []
	#Leave one out split	
	N = len(features[:,0])/len(testf[:,0])
	for k in range(len(testf[:,0])):
		#Indices and features
		X_test = testf[k,:]
		X_test = X_test.reshape(1,-1)
		idx = np.linspace(0,N-1,N)+k*N
		X_train = features
		X_train = np.delete(X_train,idx,axis=0)
		#Subtraction of mean
		X_train -= X_train.mean(0)
		X_test -= X_train.mean(0)
		Y_train = score
		Y_train = np.delete(Y_train,idx)				
		#Linear regression
		regr = sklin.Ridge(alpha=1)
		regr.fit(X_train,Y_train)
		#Predicted score		
		pred.append(regr.predict(X_test))
		
	return np.array(pred)
	
def augRF(testf,features,score,trees,d):
	pred = []
	#Leave one out split
	N = len(features[:,0])/len(testf[:,0])	
	for k in range(len(testf[:,0])):
		#Indices and features
		X_test = testf[k,:]
		X_test = X_test.reshape(1,-1)		
		idx = np.linspace(0,N-1,N)+k*N
		X_train = features
		X_train = np.delete(X_train,idx,axis=0)		
		Y_train = score
		Y_train = np.delete(Y_train,idx)				
		#Random forest regression
		rf = RandomForestClassifier(n_estimators=trees,n_jobs = 4,max_depth=d,random_state=42)
		rf.fit(X_train,Y_train)
		#Predicted score		
		pred.append(rf.predict(X_test))
		
	return np.array(pred)
	
def SVM(features,score):
	score = np.ravel(score)
	row,col = np.shape(features)	
	pred = np.zeros((row,1))
	#Leaveone out split
	loo = LeaveOneOut()
	for trainidx,testidx in loo.split(features):
	#Indices
		X_train, X_test = features[trainidx], features[testidx]
		Y_train, Y_test = score[trainidx], score[testidx]		
		#Random Forest
		clf = svm.SVC(random_state=42)
		clf.fit(X_train,Y_train)
		#Prediction
		pred[testidx,0] = clf.predict(X_test)
	return pred
	
def KNN(features,score):
	score = np.ravel(score)
	row,col = np.shape(features)	
	pred = np.zeros((row,1))
	#Leaveone out split
	loo = LeaveOneOut()
	for trainidx,testidx in loo.split(features):
	#Indices
		X_train, X_test = features[trainidx], features[testidx]
		Y_train, Y_test = score[trainidx], score[testidx]		
		#Random Forest
		clf = neighbors.KNeighborsRegressor(n_neighbors=5,weights='distance')
		clf.fit(X_train,Y_train)
		#Prediction
		pred[testidx,0] = clf.predict(X_test)
	return pred
	
def LDA(features,score):
	score = np.ravel(score)
	row,col = np.shape(features)	
	pred = np.zeros((row,1))
	#Leaveone out split
	loo = LeaveOneOut()
	for trainidx,testidx in loo.split(features):
	#Indices
		X_train, X_test = features[trainidx], features[testidx]
		Y_train, Y_test = score[trainidx], score[testidx]		
		#Random Forest
		clf = LinearDiscriminantAnalysis()
		clf.fit(X_train,Y_train)
		#Prediction
		pred[testidx,0] = clf.predict(X_test)
	return pred