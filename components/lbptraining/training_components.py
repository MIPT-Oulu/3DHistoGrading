import numpy as np

from joblib import Parallel, delayed
from tqdm import tqdm
from functools import partial
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

from components.grading.local_binary_pattern import local_normalize_abs, MRELBP
from components.grading.pca_regression import scikit_pca, regress_logo, regress_loo, standardize


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
    # par_set['N'] = hp.choice('N', [8, 16])
    par_set['N'] = 8
    par_set['ks1'] = hp.randint('ks1', 12) * 2 + 3
    par_set['sigma1'] = hp.randint('sigma1', par_set['ks1']) + 1
    par_set['ks2'] = hp.randint('ks2', 12) * 2 + 3
    par_set['sigma2'] = hp.randint('sigma2', par_set['ks2']) + 1
    par_set['R'] = hp.randint('R', 25) + 3
    par_set['r'] = hp.randint('r', par_set['R'] - 1) + 1
    par_set['wc'] = hp.randint('wc', 7) * 2 + 3
    par_set['wl'] = hp.randint('wl', 7) * 2 + 3
    par_set['ws'] = hp.randint('ws', 7) * 2 + 3
    par_set['seed'] = seed

    return par_set


def fit_model(imgs, grades, parameters, args, loss=mean_squared_error, groups=None):
    """Runs LBP, PCA and regression on given parameters and return error metric."""
    
    # LBP features
    features = []
    for img in imgs:
        img = local_normalize_abs(img, parameters)
        f = MRELBP(img, parameters, normalize=args.normalize_hist)
        features.append(f)
    features = np.array(features).squeeze()
    # Remove zero features
    features = features[~np.all(features == 0, axis=1)]

    # Centering
    mean = np.mean(features, 1)
    features = (features.T - mean).T

    # PCA
    _, score = scikit_pca(features, args.n_components, whitening=True, solver='auto')

    # Groups
    if groups is not None:
        preds, _, _ = regress_logo(score, grades, groups, method=args.regression, standard=False, use_intercept=True)
    else:
        preds, _, _ = regress_loo(score, grades, method=args.regression, standard=False, use_intercept=True)
    
    return loss(preds, grades)


def evaluate(parameters, imgs, grades, args, loss, groups=None):
    try:
        res = fit_model(imgs, grades, parameters, args, loss, groups=groups)
    except TypeError:
        res = 1e6
    # print('Parameters are: {0}'.format(parameters))
    return {'loss': res, 'status': STATUS_OK}  # , 'pars': parameters}


def optimization_hyperopt_loo(imgs, grades, args, loss, groups=None):
    """Parameter optimization with Leave-one-out."""
    # Get leave-one-out split
    loo = LeaveOneOut()
    loo.get_n_splits(grades)

    best_pars = []
    trial_list = []
    error_list = []
    print('\nOptimizing through sets')
    for train_idx, test_idx in tqdm(loo.split(grades), desc='Calculating LOO optimization'):
        # Get training set
        grades_train = grades[train_idx]
        imgs_train = imgs[train_idx]
        if groups is not None:
            groups_train = groups[train_idx]
        else:
            groups_train = None

        try:
            # Initialize, define param space
            trials = Trials()
            param_space = make_pars_hyperopt(args.seed)

            # Optimize
            min_loss = fmin(fn=partial(evaluate, imgs=imgs_train, grades=grades_train, args=args,
                            groups=groups_train, loss=loss),
                            space=param_space,
                            algo=tpe.suggest,
                            max_evals=args.n_pars,
                            trials=trials,
                            verbose=0,
                            rstate=np.random.RandomState(args.seed))

            print(min_loss)
            print(space_eval(param_space, min_loss))
            best_pars.append(space_eval(param_space, min_loss))
            error_list.append(min_loss)
            trial_list.append(trials)
        except TypeError:
            print('Batch failing. Skipping to next one')
            continue

    # Show results
    for i in range(len(best_pars)):
        print('Loss: {0}'.format(error_list[i]))
        print(best_pars[i])

    return best_pars, error_list


def optimization_randomsearch_loo(imgs, grades, args, loss, groups=None):
    # Unpack parameters
    n_pars = args.n_pars
    n_jobs = args.n_jobs
    np.random.seed(args.seed)
    # Create parameter sets
    pars = make_pars(n_pars)

    # Get leave-one-out split
    loo = LeaveOneOut()
    loo.get_n_splits(grades)

    best_pars = []
    error_list = []
    set = 1
    for train_idx, test_idx in loo.split(grades):
        min_error = 1e6
        outpars = pars[0]
        imgs_train = imgs[train_idx]
        grades_train = grades[train_idx]
        groups_train = groups[train_idx]

        for k in tqdm(range(int(len(pars)/n_jobs)+1), desc='Optimizing parameters'):
            k1 = np.min([k*n_jobs, len(pars)+1])
            k2 = np.min([(k+1)*n_jobs, len(pars)])
            if k1 < len(pars):
                _pars = pars[k1:k2]
                errors = Parallel(n_jobs=n_jobs)(delayed(fit_model)
                                                 (imgs_train, grades_train, p, args, loss, groups_train)
                                                 for p in _pars)

                min_idx = np.argmin(np.array(errors))

                _min_error = errors[min_idx]
                if _min_error < min_error:
                    outpars = _pars[min_idx]
                    min_error = _min_error
                    print('Current minimum error is: {0}'.format(min_error))
        print('Best parameters for set {0} are {1}'.format(set, outpars))
        best_pars.append(outpars)
        error_list.append(min_error)
        set += 1

    return best_pars, error_list


def optimization_randomsearch(imgs, grades, args, loss, groups=None):
    # Unpack parameters
    n_pars = args.n_pars
    n_jobs = args.n_jobs
    np.random.seed(args.seed)
    # Create parameter sets
    pars = make_pars(n_pars)

    min_error = 1e6
    outpars = pars[0]

    for k in tqdm(range(int(len(pars)/n_jobs)+1), desc='Optimizing parameters'):
        k1 = np.min([k*n_jobs, len(pars)+1])
        k2 = np.min([(k+1)*n_jobs, len(pars)])
        if k1 < len(pars):
            _pars = pars[k1:k2]
            errors = Parallel(n_jobs=n_jobs)(delayed(fit_model)
                                             (imgs, grades, p, args, loss, groups)
                                             for p in _pars)

            min_idx = np.argmin(np.array(errors))

            _min_error = errors[min_idx]
            if _min_error < min_error:
                outpars = _pars[min_idx]
                min_error = _min_error
                print('Current minimum error is: {0}'.format(min_error))

    return outpars, min_error
