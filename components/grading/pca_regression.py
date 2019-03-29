"""Contains resources for PCA dimensionality reduction and creating regression models."""

import numpy as np

from sklearn.linear_model import Ridge, LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.decomposition import PCA


def regress_loo(features, grades, method='ridge', standard=False, use_intercept=True, groups=None, convert='none'):
    """Calculates linear regression with leave-one-out split and L2 regularization.

    Parameters
    ----------
    features : ndarray
        Input features used in creating regression model.
    grades : ndarray
        Ground truth for the model.
    method : str
        Regression model used. Defaults to ridge regression, but lasso is also possible. Ridge seems to perform better.
    standard : bool
        Choice whether to center features by the mean of training split.
        Defaults to false, since whitened PCA is assumed to be centered.
    use_intercept : bool
        Choice whether to use intercept term on the model.
        If the model does not provide very powerful predictions, it is better to center them by the intercept.
    groups : ndarray
        Patients groups. Used in leave-one-group-out split.
    convert : str
        Possibility to predict exp or log of ground truth. Defaults to no conversion.
    Returns
    -------
    Array of model prdictions, model coefficients and model intercept term.
    """

    # Convert grades
    if convert == 'exp':
        grades = np.exp(grades)
    elif convert == 'log':
        grades = np.log(grades)
    else:
        pass

    predictions = []
    # Get leave-one-out split
    loo = LeaveOneOut()
    loo.get_n_splits(features)
    for train_idx, test_idx in loo.split(features):
        # Train split
        x_train, x_test = features[train_idx], features[test_idx]
        y_train, y_test = grades[train_idx], grades[test_idx]

        # Normalize with mean and std
        if standard:
            x_test -= x_train.mean(0)
            x_train -= x_train.mean(0)

        # Linear regression
        if method == 'ridge':
            model = Ridge(alpha=1, normalize=True, random_state=42, fit_intercept=use_intercept)
        else:
            model = Lasso(alpha=1, normalize=True, random_state=42, fit_intercept=use_intercept)
        model.fit(x_train, y_train)

        # Evaluate on test sample
        predictions.append(model.predict(x_test))

    predictions_flat = []
    for group in predictions:
        for p in group:
            predictions_flat.append(p)

    return np.array(predictions).squeeze(), model.coef_, model.intercept_


def regress_logo(features, grades, groups, method='ridge', standard=False, use_intercept=True, convert='none'):
    """Calculates linear regression with leave-one-group-out split and L2 regularization.

    Parameters
    ----------
    features : ndarray
        Input features used in creating regression model.
    grades : ndarray
        Ground truth for the model.
    method : str
        Regression model used. Defaults to ridge regression, but lasso is also possible. Ridge seems to perform better.
    standard : bool
        Choice whether to center features by the mean of training split.
        Defaults to false, since whitened PCA is assumed to be centered.
    use_intercept : bool
        Choice whether to use intercept term on the model.
        If the model does not provide very powerful predictions, it is better to center them by the intercept.
    groups : ndarray
        Patients groups. Used in leave-one-group-out split.
    convert : str
        Possibility to predict exp or log of ground truth. Defaults to no conversion.
    Returns
    -------
    Array of model prdictions, model coefficients and model intercept term.
    """

    # Convert grades
    if convert == 'exp':
        grades = np.exp(grades)
    elif convert == 'log':
        grades = np.log(grades)
    else:
        pass

    # Lists
    predictions, coefs, intercepts = [], [], []

    # Leave one out split
    logo = LeaveOneGroupOut()
    logo.get_n_splits(features, grades, groups)
    logo.get_n_splits(groups=groups)  # 'groups' is always required

    for train_idx, test_idx in logo.split(features, grades, groups):
        # Indices
        x_train, x_test = features[train_idx], features[test_idx]
        y_train, y_test = grades[train_idx], grades[test_idx]

        # Normalize with mean and std
        if standard:
            x_test -= x_train.mean(0)
            x_train -= x_train.mean(0)

        # Linear regression
        if method == 'ridge':
            model = Ridge(alpha=1, normalize=True, random_state=42, fit_intercept=use_intercept)
        else:
            model = Lasso(alpha=1, normalize=True, random_state=42, fit_intercept=use_intercept)
        model.fit(x_train, y_train)

        # Predicted score
        predictions.append(model.predict(x_test))
        # Save weights
        coefs.append(model.coef_)
        intercepts.append(model.intercept_)

    predictions_flat = []
    for group in predictions:
        for p in group:
            predictions_flat.append(p)

    # Convert grades back
    if convert == 'exp':
        predictions = np.log(np.array(predictions_flat))
    elif convert == 'log':
        predictions = np.exp(np.array(predictions_flat))
    else:
        predictions = np.array(predictions_flat)

    return predictions, np.mean(np.array(coefs), axis=0), np.mean(np.array(intercepts), axis=0)


def logistic_loo(features, grades, standard=False, seed=42, use_intercept=False, groups=None):
    """Calculates logistic regression with leave-one-out split.

    Parameters
    ----------
    features : ndarray
        Input features used in creating regression model.
    grades : ndarray
        Ground truth for the model.
    standard : bool
        Choice whether to center features by the mean of training split.
        Defaults to false, since whitened PCA is assumed to be centered.
    seed : int
        Random seed used in the model.
    use_intercept : bool
        Choice whether to use intercept term on the model.
        If the model does not provide very powerful predictions, it is better to center them by the intercept.
    groups : ndarray
        Patients groups. Used in leave-one-group-out split.
    Returns
    -------
    Array of model prdictions, model coefficients and model intercept term.
    """
    predictions = []
    # Leave one out split
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(features):
        # Indices
        x_train, x_test = features[train_idx], features[test_idx]
        y_train, y_test = grades[train_idx], grades[test_idx]

        # Normalize with mean and std
        if standard:
            x_test -= x_train.mean(0)
            x_train -= x_train.mean(0)

        # Linear regression
        model = LogisticRegression(solver='newton-cg', max_iter=1000, random_state=seed, fit_intercept=use_intercept)
        model.fit(x_train, y_train)

        # Predicted score
        p = model.predict_proba(x_test)
        predictions.append(p)

    predictions_flat = []
    for group in predictions:
        for p in group:
            predictions_flat.append(p)

    return np.array(predictions_flat)[:, 1], model.coef_, model.intercept_


def logistic_logo(features, grades, groups, standard=False, seed=42, use_intercept=False):
    """Calculates logistic regression with leave-one-group-out split and L2 regularization.

    Parameters
    ----------
    features : ndarray
        Input features used in creating regression model.
    grades : ndarray
        Ground truth for the model.
    standard : bool
        Choice whether to center features by the mean of training split.
        Defaults to false, since whitened PCA is assumed to be centered.
    seed : int
        Random seed used in the model.
    use_intercept : bool
        Choice whether to use intercept term on the model.
        If the model does not provide very powerful predictions, it is better to center them by the intercept.
    groups : ndarray
        Patients groups. Used in leave-one-group-out split.
    Returns
    -------
    Array of model prdictions, model coefficients and model intercept term.
    """

    # Lists
    predictions, coefs, intercepts = [], [], []
    # Leave one out split
    logo = LeaveOneGroupOut()
    logo.get_n_splits(features, grades, groups)
    logo.get_n_splits(groups=groups)  # 'groups' is always required

    for train_idx, test_idx in logo.split(features, grades, groups):
        # Indices
        x_train, x_test = features[train_idx], features[test_idx]
        y_train, y_test = grades[train_idx], grades[test_idx]

        # Normalize with mean and std
        if standard:
            x_test -= x_train.mean(0)
            x_train -= x_train.mean(0)

        # Linear regression
        model = LogisticRegression(solver='newton-cg', max_iter=1000, random_state=seed, fit_intercept=use_intercept)
        model.fit(x_train, y_train)

        # Predicted score
        p = model.predict_proba(x_test)
        predictions.append(p)
        # Save weights
        coefs.append(model.coef_)
        intercepts.append(model.intercept_)

    predictions_flat = []
    for group in predictions:
        for p in group:
            predictions_flat.append(p)

    # print('Logistic model score: {0}'.format(model.score(features, targets)))
    # print('Intercept: {0}'.format(model.intercept_))
    return np.array(predictions_flat)[:, 1], np.mean(np.array(coefs), axis=0), np.mean(np.array(intercepts), axis=0)


def regress(data_x, data_y, split, method='ridge', standard=False):
    """Calculates linear regression model by dividing data into train and test sets."""
    # Train and test split
    x_train = data_x[:split]
    x_test = data_x[split:]

    y_train = data_y[:split]
    y_test = data_y[split:]

    # Normalize with mean and std
    if standard:
        x_test -= x_train.mean(0)
        x_train -= x_train.mean(0)

    # Linear regression
    if method == 'ridge':
        model = Ridge(alpha=1, normalize=True, random_state=42)
    else:
        model = Lasso(alpha=1, normalize=True, random_state=42)
    model.fit(x_train, y_train)

    # Predicted score
    predictions = model.predict(x_test)

    # Mean squared error
    mse = mean_squared_error(y_test, predictions)

    # Explained variance
    r2 = r2_score(y_test, predictions)

    return np.array(predictions), model.coef_, mse, r2


def scikit_pca(features, n_components, whitening=False, solver='full', seed=42):
    """Calculates dimensionality reduction for input features to given number of PCA components.

    Parameters
    ----------
    features : ndarray
        Input features requiring dimensionality reduction.
    n_components : int or float
        Number of output PCA components. If >= 1, this is the number of PCa components.
        If < 1, this is the explained variance of output PCA components and number is calculated automatically.
    whitening : bool
        Choice whether to whiten the output PCA components.
    seed : int
        Random seed used in the PCA.
    solver : str
        Solver for singular value decomposition. Defaults to full solve, possibility for auto, arpack or randomized.
    Returns
    -------
    PCA object containing all calculated properties, features with dimensionality reduction.
    """
    pca = PCA(n_components=n_components, svd_solver=solver, whiten=whitening, random_state=seed)
    score = pca.fit(features).transform(features)
    return pca, score


def standardize(array, axis=0):
    """Standardization by mean and standard deviation.

    Parameters
    ----------
    array : ndarray
        Input array to be standardized.
    axis : int
        Axis of standardization.
    Returns
    -------
    Standardized array.
    """
    mean = np.mean(array, axis=axis)
    std = np.std(array, axis=axis)
    try:
        res = (array - mean) / std
    except ValueError:
        res = ((array.T - mean) / std).T
    return res


def get_pca(features, n_components):
    """Calculates principal components using covariance matrix or singular value decomposition. Alternate method.

    Parameters
    ----------
    features : ndarray
        Input features requiring dimensionality reduction.
    n_components : int
        Number of output PCA components.

    Returns
    -------
    Eigenvectors, dimensionality reduced features.
    """
    # Feature dimension, x=num variables,n=num observations
    x, n = np.shape(features)
    # Mean feature
    mean_f = np.mean(features, axis=1)
    # Centering
    centered = np.zeros((x, n))
    for k in range(n):
        centered[:, k] = features[:, k]-mean_f

    # PCs from covariance matrix if n>=x, svd otherwise
    if n >= x:
        # Covariance matrix
        cov = np.zeros((x, x))
        f = np.zeros((x, 1))
        for k in range(n):
            f[:, 0] = centered[:, k]
            cov = cov+1/n*np.matmul(f, f.T)

        # Eigenvalues
        e, v = np.linalg.eig(cov)
        # Sort eigenvalues and vectors to descending order
        idx = np.argsort(e)[::-1]
        v = np.matrix(v[:, idx])

        for k in range(n_components):
            s = np.matmul(v[:, k].T, centered).T
            try:
                score = np.concatenate((score, s), axis=1)
            except NameError:
                score = s
            p = v[:, k]
            try:
                n_components = np.concatenate((n_components, p), axis=1)
            except NameError:
                n_components = p
    else:
        # PCA with SVD
        u, s, v = np.linalg.svd(centered, compute_uv=1)
        n_components = v[:, :n_components]
        score = np.matmul(u, s).T[:, 1:n_components]
    return n_components, score
