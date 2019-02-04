import numpy as np

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.decomposition import PCA


def regress_loo(features, sgrades):
    """Calculates linear regression with leave-one-out split."""
    predictions = []
    # Get leave-one-out split
    loo = LeaveOneOut()
    loo.get_n_splits(features)
    for train_idx, test_idx in loo.split(features):
        # Train split
        f = features[train_idx] - features.mean(0)
        g = sgrades[train_idx]

        # Linear regression
        model = Ridge(alpha=1, normalize=True, random_state=42)
        model.fit(f, g.reshape(-1, 1))

        # Evaluate on test sample
        p = model.predict((features[test_idx] - features.mean(0)).reshape(1, -1))
        predictions.append(p)
    return np.array(predictions).squeeze(), model.coef_


def regress_logo(features, score, groups=None):
    """Calculates linear regression with leave-one-group-out split."""
    predictions = []
    if groups is None:
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
                           9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
                           15, 16, 16, 17, 17, 18, 19, 19])
    # Leave one out split
    logo = LeaveOneGroupOut()
    logo.get_n_splits(features, score, groups)
    logo.get_n_splits(groups=groups)  # 'groups' is always required

    for train_idx, test_idx in logo.split(features, score, groups):
        # Indices
        x_train, x_test = features[train_idx], features[test_idx]
        x_test -= x_train.mean(0)
        x_train -= x_train.mean(0)

        y_train, y_test = score[train_idx], score[test_idx]
        # Linear regression
        model = Ridge(alpha=1, normalize=True, random_state=42)
        model.fit(x_train, y_train)
        # Predicted score
        predictions.append(model.predict(x_test))

    predictions_flat = []
    for group in predictions:
        for p in group:
            predictions_flat.append(p)

    return np.array(predictions_flat), model.coef_


def regress(data_x, data_y, split):
    """Calculates linear regression model by dividing data into train and test sets."""
    # Train and test split
    data_x_train = data_x[:split]
    data_x_test = data_x[split:]

    data_y_train = data_y[:split]
    data_y_test = data_y[split:]

    # Linear regression
    model = Ridge(alpha=1, normalize=True, random_state=42)
    model.fit(data_x_train, data_y_train)
    # Predicted score
    predictions = model.predict(data_x_test)

    # Mean squared error
    mse = mean_squared_error(data_y_test, predictions)

    # Explained variance
    r2 = r2_score(data_y_test, predictions)

    return np.array(predictions), model.coef_, mse, r2


# Logistic regression
def logistic_loo(features, score):
    """Calculates logistic regression with leave-one-out split."""
    predictions = []
    # Leave one out split
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(features):
        # Indices
        x_train, x_test = features[train_idx], features[test_idx]
        x_test -= x_train.mean(0)
        x_train -= x_train.mean(0)

        y_train, y_test = score[train_idx], score[test_idx]
        # Linear regression
        model = LogisticRegression(solver='newton-cg', max_iter=1000)
        model.fit(x_train, y_train)
        # Predicted score
        p = model.predict_proba(x_test)
        predictions.append(p)

    predictions = np.array(predictions)
    predictions = predictions[:, :, 1]
    return predictions.flatten()


def logistic_logo(features, score, groups=None):
    """Calculates logistic regression with leave-one-group-out split."""
    predictions = []
    if groups is None:
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
                           9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
                           15, 16, 16, 17, 17, 18, 19, 19])
    # Leave one out split
    logo = LeaveOneGroupOut()
    logo.get_n_splits(features, score, groups)
    logo.get_n_splits(groups=groups)  # 'groups' is always required

    for train_idx, test_idx in logo.split(features, score, groups):
        # Indices
        x_train, x_test = features[train_idx], features[test_idx]
        x_test -= x_train.mean(0)
        x_train -= x_train.mean(0)

        y_train, y_test = score[train_idx], score[test_idx]
        # Linear regression
        model = LogisticRegression(solver='newton-cg', max_iter=1000)
        model.fit(x_train, y_train)
        # Predicted score
        p = model.predict_proba(x_test)
        predictions.append(p)

    # predictions = np.array(predictions)
    # predictions = predictions[:,:,1]

    predictions_flat = []
    for group in predictions:
        for p in group:
            predictions_flat.append(p)

    return np.array(predictions_flat)[:, 1]


def regress_old(features, score):
    """Calculates linear regression with leave-one-out split. Obsolete."""
    predictions = []
    # Leave one out split
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(features):
        # Indices
        x_train, x_test = features[train_idx], features[test_idx]
        x_test -= x_train.mean(0)
        x_train -= x_train.mean(0)

        y_train, y_test = score[train_idx], score[test_idx]
        # Linear regression
        model = Ridge(alpha=1, normalize=True, random_state=42)
        model.fit(x_train, y_train)
        # Predicted score
        predictions.append(model.predict(x_test))

    return np.array(predictions), model.coef_


def scikit_pca(features, n_components, whitening=False, solver='full'):
    """Calculates PCA components using Scikit implementation."""
    pca = PCA(n_components=n_components, svd_solver=solver, whiten=whitening, random_state=42)
    score = pca.fit(features).transform(features)
    return pca, score


def get_pca(features, n_components):
    """Calculates principal components using covariance matrix or singular value decomposition."""
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
