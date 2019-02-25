import numpy as np
from time import time
import gc

from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import spearmanr
from functools import partial
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval

from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from components.lbptraining.lbp_components import Conv_MRELBP
from components.grading.local_binary_pattern import local_normalize_abs, MRELBP
from components.grading.pca_regression import scikit_pca, regress_logo, regress_loo


def make_pars(n_pars):
    """Generate random LBP parameters."""
    pars = []

    for k in range(n_pars):
        tmp = dict()
        tmp['N'] = 8
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


def make_pars_hyperopt(seed):
    """Generate LBP parameter space for hyperopt."""
    par_set = dict()
    par_set['N'] = hp.choice('N', [8, 16])
    par_set['ks1'] = hp.randint('ks1', 12) * 2 + 3
    par_set['sigma1'] = hp.randint('sigma1', par_set['ks1']) + 1
    par_set['ks2'] = hp.randint('ks2', 12) * 2 + 3
    par_set['sigma2'] = hp.randint('sigma2', par_set['ks2']) + 1
    par_set['R'] = hp.randint('R', 25) + 3
    par_set['r'] = hp.randint('r', par_set['R'] - 2) + 1
    par_set['wc'] = hp.randint('wc', 7) * 2 + 3
    par_set['wl'] = hp.randint('wl', 7) * 2 + 3
    par_set['ws'] = hp.randint('ws', 7) * 2 + 3
    par_set['seed'] = seed

    return par_set


def evaluate(parameters, imgs, grades, args, groups=None, callback=None):
    res = get_error(imgs, grades, parameters, args, groups=groups)
    if callback is not None:
        callback()
    return {'loss': 1 - res, 'status': STATUS_OK}


def get_mse(preds, targets):
    n = len(preds)
    errors = preds.flatten() - targets.flatten()
    return (errors ** 2).sum() / n


def get_corr_loss(preds, targets):
    rho = spearmanr(targets, preds)
    return 1 - rho[0]


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
    if groups is not None:
        preds, _ = regress_logo(score, grades, groups)
    else:
        preds, _ = regress_loo(score, grades)
    
    return get_mse(preds, grades)

    # return get_corr_loss(preds, grades)


def parameter_optimization_loo(imgs, grades, args, groups=None, seed=42):
    """Parameter optimization with Leave-one-out."""
    # Get leave-one-out split
    loo = LeaveOneOut()
    loo.get_n_splits(grades)

    best_pars = []
    trial_list = []
    for train_idx, test_idx in tqdm(loo.split(grades), desc='Optimizing through sets'):
        if groups is not None:
            groups_train = groups[train_idx]
        else:
            groups_train = None

        trials = Trials()
        pbar = tqdm(total=args.n_pars, desc="Hyperopt:")
        param_space = make_pars_hyperopt(seed)
        grades_train = grades[train_idx]
        imgs_train = imgs[train_idx]
        best = fmin(fn=partial(evaluate, imgs=imgs_train, grades=grades_train, args=args,
                               groups=groups_train, callback=lambda: pbar.update()),
                    space=param_space,
                    algo=tpe.suggest,
                    max_evals=args.n_pars,
                    trials=trials,
                    verbose=0,
                    rstate=np.random.RandomState(seed))
        print(best)
        print(space_eval(param_space, best))
        best_pars.append(space_eval(param_space, best))
        trial_list.append(trials)
        pbar.close()

    return best_pars, trials


def parameter_optimization_random(imgs, grades, args, groups=None):
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
