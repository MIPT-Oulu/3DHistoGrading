import numpy as np
from time import time
import gc

from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from LBPTraining.LBP_components import Conv_MRELBP
from Grading.local_binary_pattern import local_normalize_abs, MRELBP
from Grading.pca_regression import scikit_pca, regress_logo


def make_pars(n_pars):
    """Generate random LBP parameters."""
    pars = []

    for k in range(n_pars):
        tmp = {'ks1': 0, 'sigma1': 0, 'ks2': 0, 'sigma2': 0, 'N': 8, 'R': 0, 'r': 0, 'wc': 0, 'wl': 0, 'ws': 0}
        tmp['ks1'] = np.random.randint(1, 13)*2+1
        tmp['sigma1'] = np.random.randint(1, tmp['ks1']+1)
        tmp['ks2'] = np.random.randint(1, 13)*2+1
        tmp['sigma2'] = np.random.randint(1, tmp['ks2']+1)
        tmp['R'] = np.random.randint(3, 28)
        tmp['r'] = np.random.randint(1, tmp['R'])
        tmp['wc'] = np.random.randint(1, 8)*2+1
        tmp['wl'] = np.random.randint(1, 8)*2+1
        tmp['ws'] = np.random.randint(1, 8)*2+1
        pars.append(tmp)

    return pars


def get_mse(preds, targets):
    n = len(preds)
    errors = preds.flatten() - targets.flatten()
    return (errors ** 2).sum() / n


def get_error(imgs, grades, parameters, args, groups=None):
    
    features = []
    
    for img in imgs:
        img = local_normalize_abs(img, parameters)
        f = MRELBP(img, parameters, normalize=args.hist_normalize)
        features.append(f)
        
    features = np.array(features).squeeze()

    # PCA
    _, score = scikit_pca(features, args.n_components, whitening=True, solver='auto')

    # Groups
    preds, _ = regress_logo(score, grades, groups)
    
    mse = get_mse(preds, grades)
    
    return mse


def find_pars_bforce(imgs, grades, args, groups=None):
    # Unpack parameters
    n_pars = args.n_pars
    n_jobs = args.n_jobs

    np.random.seed(42)
    pars = make_pars(n_pars)
    min_error = 1e6
    outpars = pars[0]
    for k in tqdm(range(int(len(pars)/n_jobs)+1), desc='Optimizing parameters'):
        k1 = np.min([k*n_jobs, len(pars)+1])
        k2 = np.min([(k+1)*n_jobs, len(pars)])
        start = time()
        if k1 < len(pars):
            _pars = pars[k1:k2]
            errors = Parallel(n_jobs=n_jobs)(delayed(get_error)(imgs, grades, p, args, groups) for p in _pars)
    
            min_idx = np.argmin(np.array(errors))
    
            _min_error = errors[min_idx]
            #print("Current minimum error is: {0}".format(_min_error))
            #print("Parameters are:")
            #print(_pars[min_idx])
            stop = time()
            #print("Took {0} seconds".format(int(stop-start)))
            if _min_error < min_error:
                outpars = _pars[min_idx]
                min_error = _min_error
            
    return outpars, min_error
