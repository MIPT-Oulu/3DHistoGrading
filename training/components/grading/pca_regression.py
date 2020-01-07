"""Contains resources for PCA dimensionality reduction and creating regression models."""

import numpy as np
import shap
import matplotlib.pyplot as plt


from joblib import dump
from pathlib import Path
from time import strftime

from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression, Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.decomposition import PCA

from components.utilities.load_write import load_binary_weights


def regress_loo(features, grades, method='ridge', standard=False, use_intercept=True, groups=None, convert='none', alpha=1.0):
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
            model = Ridge(alpha=alpha, normalize=True, random_state=42, fit_intercept=use_intercept)
        else:
            model = Lasso(alpha=alpha, normalize=True, random_state=42, fit_intercept=use_intercept)
        model.fit(x_train, y_train)

        # Evaluate on test sample
        predictions.append(model.predict(x_test))

    predictions_flat = []
    for group in predictions:
        for p in group:
            predictions_flat.append(p)

    return np.array(predictions).squeeze(), model.coef_, model.intercept_


def regress_logo(features, grades, groups, method='ridge', standard=False, use_intercept=True, convert='none', alpha=1.0):
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
    alpha : float
        Regularization coefficient. c^-1
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
            model = Ridge(alpha=alpha, normalize=True, random_state=42, fit_intercept=use_intercept)
        elif method == 'lasso':
            model = Lasso(alpha=alpha, normalize=True, random_state=42, fit_intercept=use_intercept)
        else:
            model = LinearRegression(normalize=True, fit_intercept=use_intercept, n_jobs=-1)
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
    Array of model predictions, model coefficients and model intercept term.
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
        predictions.extend(p[:, 1])  # Add the positive predictions to list
        # Save weights
        coefs.append(model.coef_)
        intercepts.append(model.intercept_)

    # Average coefficients
    coefs = np.mean(np.array(coefs), axis=0).squeeze()
    intercepts = np.mean(np.array(intercepts), axis=0).squeeze()

    return np.array(predictions), coefs, intercepts


def rforest_logo(features, grades, groups, standard=False, seed=42, n_trees=50, tree_depth=None, savepath=None, zone=''):
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
    n_trees : int
        Number of trees in the Random Forest
    tree_depth : int
        Maximum depth of the individual tree.
    groups : ndarray
        Patients groups. Used in leave-one-group-out split.
    savepath : str
        Path to save the model.
    zone : str
        Zone that is graded.
    Returns
    -------
    Array of model predictions, model coefficients and model intercept term.
    """

    # Lists
    predictions, coefs, intercepts, models = [], [], [], []
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
        model = RandomForestClassifier(n_estimators=n_trees, random_state=seed, max_depth=tree_depth, n_jobs=12)
        model.fit(x_train, y_train)

        # Predicted score
        p = model.predict_proba(x_test)
        predictions.append(p)

        # Save weights
        coefs.append(model.feature_importances_)  # Importance of PCA components is returned
        intercepts.append(0.0)  # No intercept in RF
        models.append(model)

    predictions_flat = []
    for group in predictions:
        for p in group:
            predictions_flat.append(p)

    if savepath is not None:
        Path(savepath + '/models/').mkdir(exist_ok=True)
        filename = savepath + '/models/' + strftime(f'RF_model_{zone}_%Y_%m_%d_%H_%M_%S.sav')
        dump(models, filename)

    return np.array(predictions_flat)[:, 1], np.mean(np.array(coefs), axis=0).squeeze(), np.mean(np.array(intercepts), axis=0).squeeze()


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


def pca_regress_pipeline_log(features, grades, groups, n_components=0.9, solver='full', whitening=True, standard=False,
                             seed=42, mod_coefs=True, alpha=0.1, grade_name='', savepath=None):

    feature_names = ['Center +', 'Center -', 'Large U-1', 'Large U-2', 'Large U-3', 'Large U-4', 'Large U-5', 'Large U-6',
                     'Large U-7', 'Large N-U', 'Small U-1', 'Small U-2', 'Small U-3', 'Small U-4', 'Small U-5', 'Small U-6', 'Small U-7',
                     'Small N-U', 'Radial U-0', 'Radial U-1', 'Radial U-2', 'Radial U-3', 'Radial U-4', 'Radial U-5', 'Radial U-6',
                     'Radial U-7', 'Radial U-8', 'Radial N-U']
    grades_log = grades

    # Fit PCA to full data
    pca = PCA(n_components=n_components, svd_solver=solver, whiten=whitening, random_state=seed)
    pca.fit(features)

    # Leave one out split
    logo = LeaveOneGroupOut()
    logo.get_n_splits(features, grades_log, groups)
    logo.get_n_splits(groups=groups)  # 'groups' is always required
    all_shap_values, all_shap_values_lin = [], []
    for train_idx, test_idx in logo.split(features, grades_log, groups):
        # Indices
        x_train, x_test = features[train_idx], features[test_idx]
        y_train, y_test = grades_log[train_idx], grades_log[test_idx]

        # Normalize with mean and std
        if standard:
            x_test -= x_train.mean(0)
            x_train -= x_train.mean(0)

        # Logistic regression
        model = LogisticRegression(solver='newton-cg', max_iter=1000, random_state=seed, fit_intercept=False)
        model.fit(pca.transform(x_train), y_train > 1)

        model_lin = Ridge(alpha=alpha, normalize=True, random_state=seed, fit_intercept=True)
        model_lin.fit(pca.transform(x_train), y_train)

        # Predicted score (for logistic regression)
        p = model.predict_proba(pca.transform(x_test))
        p_lin = model_lin.predict(pca.transform(x_test))

        # Merge PCA into the linear model
        if mod_coefs:
            coef = (pca.components_.T / pca.singular_values_) @ model.coef_.T * np.sqrt(pca.n_samples_ - 1)
            coef_lin = (pca.components_.T / pca.singular_values_) @ model_lin.coef_.T * np.sqrt(pca.n_samples_ - 1)

            # Update models
            model.coef_ = coef.T
            model_lin.coef_ = coef_lin.T

            p2_lin = model_lin.predict(x_test)
            p2 = model.predict_proba(x_test)

            # Inference
            p_inf = (x_test @ coef).squeeze()
            p_inf = (1 + np.exp(-p_inf)) ** -1

            eps = 1.0e-10
            assert np.sum(np.abs(p - p2)) < eps, 'LOGReg results are not equal'
            assert np.sum(np.abs(p_inf - p[:, 1])) < eps, 'LOGReg results are not equal'
            assert np.sum(np.abs(p_lin - p2_lin)) < eps, 'LINReg results are not equal'
        else:  # Otherwise run PCA
            x_train = pca.transform(x_train)
            x_test = pca.transform(x_test)

        # Interpretability

        # Logistic regression
        explainer = shap.LinearExplainer(model, x_train, feature_dependence='correlation', nsamples=x_train.shape[0])
        shap_values = explainer.shap_values(x_test)

        # Linear regression
        explainer_lin = shap.LinearExplainer(model_lin, x_train, feature_dependence='correlation',
                                             nsamples=x_train.shape[0])
        shap_values_lin = explainer_lin.shap_values(x_test)

        # Append prediction
        all_shap_values.append(shap_values)
        all_shap_values_lin.append(shap_values_lin)

    # Combine shap values and plot the summary
    all_shap_values = np.vstack(all_shap_values)
    all_shap_values_lin = np.vstack(all_shap_values_lin)

    # Inverse PCA for the model without PCA
    if not mod_coefs:
        all_shap_values = pca.inverse_transform(all_shap_values)
        all_shap_values_lin = pca.inverse_transform(all_shap_values_lin)

    # Force plot
    # shap.force_plot(explainer.expected_value, all_shap_values, features)
    # plt.show()

    # Summary plots
    shap.summary_plot(all_shap_values, features, show=False, feature_names=feature_names)
    # plt.title(f'Logistic Regression ({grade_name})')
    if savepath is not None:
        plt.savefig(f'{savepath}{grade_name}_logistic_cov.png', transparent=False, bbox_inches='tight')
        plt.show()
    else:
        plt.show()
    shap.summary_plot(all_shap_values_lin, features, show=False, feature_names=feature_names)
    # plt.title(f'Linear Ridge Regression ({grade_name})')
    if savepath is not None:
        plt.savefig(f'{savepath}{grade_name}_linear_cov.png', transparent=False, bbox_inches='tight')
        plt.show()
    else:
        plt.show()


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


def evaluate_model(features, args, model_path):
    """Evaluates features on saved model.

    Parameters
    ----------
    features : ndarray
        MRELBP features used for predictions.
    args : Namespace
        standardization = Centering or centering and standardization of features.
    model_path : str
        Path to saved model.
    Returns
    -------
    Linear predictions, logistic predicitons, PCA components.
    """
    # Load model

    _, n_comp, eigen_vectors, sv_scaled, weights, weights_log, mean_feature, [intercept_lin, intercept_log] \
        = load_binary_weights(model_path)

    # Standardize features
    if args.standardization == 'centering':
        mean = np.mean(features, 1)
        features = features.T - mean
        # features = features.T - mean_feature
    else:
        features = standardize(features.T, axis=0)

    # PCA
    if args.use_PCA:
        score = np.matmul(features, eigen_vectors / sv_scaled)
    else:
        score = features

    # Regression
    pred_linear = np.matmul(score, weights) + intercept_lin
    pred_logistic = np.matmul(score, weights_log) + intercept_log
    pred_logistic = (1 + np.exp(-pred_logistic)) ** -1

    return pred_linear, pred_logistic, score