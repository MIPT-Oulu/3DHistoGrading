import numpy as np
from time import time
import gc

from joblib import Parallel,delayed

from scipy.ndimage import correlate

from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


from Old.LBP_components import Conv_MRELBP



def make_pars(n_pars):
    pars = []

    for k in range(n_pars):
        tmp = {'ks1':0,'sigma1':0,'ks2':0,'sigma2':0,'R':0,'r':0,'wc':0,'wR':0,'wr':0}
        tmp['ks1'] = np.random.randint(1,13)*2+1
        tmp['sigma1'] = np.random.randint(1,tmp['ks1']+1)
        tmp['ks2'] = np.random.randint(1,13)*2+1
        tmp['sigma2'] = np.random.randint(1,tmp['ks2']+1)
        tmp['R'] = np.random.randint(3,28)
        tmp['r'] = np.random.randint(1,tmp['R'])
        tmp['wc'] = np.random.randint(1,8)*2+1
        tmp['wR'] = np.random.randint(1,8)*2+1
        tmp['wr'] = np.random.randint(1,8)*2+1
        pars.append(tmp)

    return pars

def make_2d_gauss(ks,sigma):
    #Mean indices
    c = ks//2
    
    #Exponents
    x = (np.linspace(0,ks-1,ks)-c)**2
    y = (np.linspace(0,ks-1,ks)-c)**2
    
    #Denominator
    denom = np.sqrt(2*np.pi*sigma**2)
    
    #Evaluate gaussians
    ex = 1/denom*np.exp(-0.5*x/sigma**2)
    ey = 1/denom*np.exp(-0.5*y/sigma**2)
    
    #Iterate over kernel size
    kernel = np.zeros((ks,ks))
    for k in range(ks):
        kernel[k,:] = ey[k]*ex
    
    #Normalize so kernel sums to 1
    kernel /= kernel.sum()
    
    return kernel

def local_normalize(image,ks1,sigma1,ks2,sigma2 ):
    #Generate gaussian kernel
    kernel1 = make_2d_gauss(ks1,sigma1)
    kernel2 = make_2d_gauss(ks2,sigma2)
    
    mu = correlate(image,kernel1)
    
    centered = image-mu
    
    sd = correlate(centered**2,kernel2)**0.5
    
    return centered/(sd+1e-9)

def LOO_LR(features,grades,n_splits = -1,mode='ridge'):
    #Get number of splits
    if n_splits == -1:
        n_splits = features.shape[0]
    
    #Make splits
    LOO = LeaveOneOut()
    LOO.get_n_splits(features)
    
    #Iterate over split
    output = []
    for train_idx, test_idx in LOO.split(features):
        f = features[train_idx]-features.mean(0)
        g = grades[train_idx]
        
        #Regression
        if mode == 'ridge':
            model = Ridge(alpha=1,normalize=True,random_state=42)
        elif mode == 'random_forest':
            model = RandomForestRegressor(n_estimators=500 ,n_jobs = 16, max_depth=7,random_state=42)
        model.fit(f,g.reshape(-1,1))
        
        #Make prediction
        output.append(model.predict((features[test_idx]-features.mean(0)).reshape(1,-1)))
        
    gc.collect()

    return np.array(output)

def MSE(preds,targets):
    N = len(preds)
    errors = preds.flatten()-targets.flatten()
    return (errors**2).sum()/N

def get_error(imgs,grades,args,mode='ridge'):
    
    features = []
    
    for img in imgs:
        img = local_normalize(img,args['ks1'],args['sigma1'],args['ks2'],args['sigma2'])
        f = Conv_MRELBP(img,8,args['R'],args['r'],args['wR'],args['wr'],args['wc'])
        features.append(f)
        
    features = np.array(features).squeeze()
    pc = PCA(20,whiten=True,random_state=42)
    pcfeatures = pc.fit(features).transform(features)
        
    preds = LOO_LR(pcfeatures,grades,mode)
    
    mse = MSE(preds,grades)
    
    return mse

def get_feature(img,args,old=False):
    img = local_normalize(img,args['ks1'],args['sigma1'],args['ks2'],args['sigma2'])
    if not old:
        feature = Conv_MRELBP(img,8,args['R'],args['r'],args['wR'],args['wr'],args['wc'])
    else:
        feature = MRELBP(img,8,args['R'],args['r'],args['wc'],(args['wR'],args['wr']))
    return feature

def make_pred(imgs,grades,args,old):
    features = Parallel(n_jobs=16)(delayed(get_feature)(img,args,old) for img in imgs)
    features = np.array(features).squeeze()
    pc = PCA(10,whiten=True,random_state=42)
    pcfeatures = pc.fit(features).transform(features)
        
    preds = LOO_LR(pcfeatures,grades)
    
    return preds

def find_pars_bforce(imgs,grades,n_pars,mode,n_jobs):
    np.random.seed(42)
    pars = make_pars(n_pars)
    min_error = 1e6
    outpars = pars[0]
    for k in range(int(len(pars)/n_jobs)+1):
        k1 = np.min([k*n_jobs,len(pars)+1])
        k2 = np.min([(k+1)*n_jobs,len(pars)])
        start = time()
        if k1<len(pars):
            _pars = pars[k1:k2]
            errors = Parallel(n_jobs=n_jobs)(delayed(get_error)(imgs,grades,p) for p in _pars)
    
            min_idx = np.argmin(np.array(errors))
    
            _min_error = errors[min_idx]
            print("Current minimum error is: {0}".format(_min_error))
            print("Parameters are:")
            print(_pars[min_idx])
            stop = time()
            print("Took {0} seconds".format(int(stop-start)))
            if _min_error < min_error:
                outpars = _pars[min_idx]
                min_error = _min_error
            
    return outpars, min_error
