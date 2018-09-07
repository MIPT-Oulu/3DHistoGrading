#Regression
import numpy as np
import scipy.signal
import os
import h5py
import time
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
from joblib import Parallel, delayed
import sklearn.metrics as skmet
import sklearn.linear_model as sklin

import ImageProcessing as IP

def load_and_f(path,files):
	#Mapping for lbp
	mapping = IP.getmapping(8)
	for k in range(len(files)):
		#Load file
		file = os.path.join(path,files[k])
		try:
			file = sio.loadmat(file)
			Mz = file['Mz']
			sz = file['sz']			
		except NotImplementedError:
			file = h5py.File(file)
			Mz = file['Mz'][()]
			sz = file['sz'][()]			
		
		#images
		
		#Combine mean and sd images
		image = Mz+sz
		#Grayscale normalization
		image = IP.localstandard(image,23,5,5,1)
		#image = image[20:-20,20:-20]
		#Feature extraction
		dict = {'R':9,'r':3,'wc':5,'wr':(5,5)}		
		f1,f2,f3,f4 = IP.MRELBP(image,8,dict['R'],dict['r'],dict['wc'],dict['wr'])
		
		#Normalization and mapping of the features f2(large neighbourhood lbp) and f4(radial lbp)
		
		#f1 = 1/np.linalg.norm(f1)*f1
		f2 = IP.maplbp(f2,mapping)
		#f2 = 1/np.linalg.norm(f2)*f2
		f3 = IP.maplbp(f3,mapping)
		#f3 = 1/np.linalg.norm(f3)*f3
		f4 = IP.maplbp(f4,mapping)
		#f4 = 1/np.linalg.norm(f4)*f4
		
		#Concatenate features
		f = np.concatenate((f1.T,f2.T,f3.T,f4.T),axis=0)
		try:
			features = np.concatenate((features,f),axis=1)
		except NameError:
			features = f
	
	return features

def parallel_f(path,files,n_jobs): 
	parallelizer = Parallel(n_jobs=n_jobs)
	nlist = []
	N = int(len(files)/n_jobs)
	for k in range(n_jobs):
		nlist.append(files[k*N:(k+1)*N])
		
	iterator = ( delayed(load_and_f)(path,nfiles)
				for nfiles in nlist )
	result = parallelizer(iterator)
	features = np.hstack(result)
	return features, result
	
if __name__=='__main__':
	#Start time
	start_time = time.time()
	#Samples
	impath = r'C:\Users\jfrondel\Desktop\Work\Koodit\BOFKoodia\Segmentation\SurfaceTopology\CleanTopoNew'
	filelist = os.listdir(impath)

	#Grades from excel file

	grades = pd.ExcelFile(r'C:\Users\jfrondel\Desktop\Work\Koodit\BOFKoodia\Segmentation\PTAgreiditjanaytteet.xls')
	grades = pd.read_excel(grades)
	grades = pd.DataFrame.as_matrix(grades)
	grades = grades[:,2:3]
	g = grades[:,0].astype('int')
	#Features
	features,result = parallel_f(impath,filelist,4)	
	#PCA
	score = IP.ScikitPCA(features,10)
	
	pred1 = IP.regress(features.T,g)
	pred2 = IP.logreg(features.T,g>0)
	#pred2 = IP.logreg(features.T,g>0)
	for p in range(len(pred1)):
		if pred1[p]<0:
			pred1[p] = 0
		if pred1[p] > 3:
			pred1[p]=3
	
	#Plotting the prediction
	a = g
	b = np.round(pred1).astype('int')	
	
	#Plotting
	x = score[:,0]
	y = score[:,1]
	fig = plt.figure(figsize=(6,6))
	#plt.grid(True)
	ax1 = fig.add_subplot(111)
	ax1.scatter(score[g<2,0],score[g<2,1],marker='o',color='b',label='Normal')	
	ax1.scatter(score[g>1,0],score[g>1,1],marker='s',color='r',label='OA')
	
	for k in range(len(grades[:,0])):
		txt = filelist[k]
		txt = txt[0:-4]
		txt = txt+str(grades[k,0])		
		if grades[k,0] >= 2:
			ax1.scatter(x[k],y[k],marker='s',color='r')
			#ax1.annotate(txt,xy=(x[k],y[k]),color='r')
		else:
			ax1.scatter(x[k],y[k],marker='o',color='b')
			#ax1.annotate(txt,xy=(x[k],y[k]),color='b')	
	
	for k in range(len(filelist)):
		print(filelist[k],a[k],pred1[k])#,pred3[k])
	C1 = skmet.confusion_matrix(a,b)
	
	MSE1 = skmet.mean_squared_error(a,pred1)
	fpr, tpr, thresholds = skmet.roc_curve(a>0, np.round(pred1)>0, pos_label=1)
	AUC1 = skmet.auc(fpr,tpr)	
	fig0  = plt.figure(figsize=(6,6))
	ax0 = fig0.add_subplot(111)
	ax0.plot(fpr,tpr)
	AUC1 = skmet.roc_auc_score(a>0,pred2)
	
	print(C1)
	
	print(MSE1,AUC1)#,MSE2,MSE3,MSE4)
	
	t = time.time()-start_time
	print("-- %s seconds --" % t)
	#plt.legend()
	#plt.show()
			
	m, b = np.polyfit(a, pred1.flatten(), 1)
	
	R2 = skmet.r2_score(a,pred1.flatten())
	print(R2)
	fig = plt.figure(figsize=(6,6))
	ax2 = fig.add_subplot(111)
	ax2.scatter(a,pred1.flatten())
	ax2.plot(a,m*a,'-',color='r')
	ax2.set_xlabel('Actual grade')
	ax2.set_ylabel('Predicted')
	for k in range(len(grades[:,0])):
		txt = filelist[k]
		txt = txt[0:-4]
		txt = txt+str(grades[k,0])
		ax2.annotate(txt,xy=(a[k],pred1[k]),color='r')
	plt.show()
	
	#Save everythin
	dict = {'g':g,'pred1':pred1,'pred2':pred2}
	sio.savemat(r'c:\users\jfrondel\desktop\regressresults.mat',dict)
	r'''
	#Save everything to excel file
	Data = np.concatenate((g.flatten,pred1.flatten(),pred2.flatten()))
	df1 = pd.DataFrame(Data)
	writer = pd.ExcelWriter(r'c:\users\jfrondel\desktop\output.xlsx')
	df1.to_excel(writer)
	writer.save()
	'''