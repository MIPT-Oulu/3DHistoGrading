import numpy as np

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut


def regress_loo(features, sgrades):
    """Calculates linear regression with leave-one-out split."""
    # Evaluate surface
    loo = LeaveOneOut()
    loo.get_n_splits(features)
    predictions = []
    for train_idx, test_idx in loo.split(features):
        # Train split
        f = features[train_idx] - features.mean(0)
        g = sgrades[train_idx]

        # Linear regression
        ridge_model = Ridge(alpha=1, normalize=True, random_state=42)
        ridge_model.fit(f, g.reshape(-1, 1))

        # Evaluate on test sample
        p = ridge_model.predict((features[test_idx] - features.mean(0)).reshape(1, -1))
        predictions.append(p)
    return np.array(predictions).squeeze(), ridge_model.coef_


def regress_logo(features, score, groups=None):
    """Calculates linear regression with leave-one-group-out split."""
    pred = []
    if groups is None:
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
                           9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
                           15, 16, 16, 17, 17, 18, 19, 19])
    # Leave one out split
    logo = LeaveOneGroupOut()
    logo.get_n_splits(features, score, groups)
    logo.get_n_splits(groups=groups)  # 'groups' is always required

    for trainidx, testidx in logo.split(features, score, groups):
        # Indices
        x_train, x_test = features[trainidx], features[testidx]
        x_test -= x_train.mean(0)
        x_train -= x_train.mean(0)

        y_train, y_test = score[trainidx], score[testidx]
        # Linear regression
        regr = Ridge(alpha=1, normalize=True, random_state=42)
        regr.fit(x_train, y_train)
        # Predicted score
        pred.append(regr.predict(x_test))

    predflat = []
    for group in pred:
        for p in group:
            predflat.append(p)

    return np.array(predflat), regr.coef_


def regress(data_x, data_y, split):
    """Calculates linear regression model by dividing data into train and test sets."""
    pred = []
    # Train and test split
    data_x_train = data_x[:split]
    data_x_test = data_x[split:]

    data_y_train = data_y[:split]
    data_y_test = data_y[split:]

    # Linear regression
    regr = Ridge(alpha=1, normalize=True, random_state=42)
    regr.fit(data_x_train, data_y_train)
    # Predicted score
    pred = regr.predict(data_x_test)

    # Mean squared error
    mse = mean_squared_error(data_y_test, pred)

    # Explained variance
    r2 = r2_score(data_y_test, pred)

    return np.array(pred), regr.coef_, mse, r2


# Logistic regression
def logreg_loo(features, score):
    """Calculates logistic regression with leave-one-out split."""
    pred = []
    # Leave one out split
    loo = LeaveOneOut()
    for trainidx, testidx in loo.split(features):
        # Indices
        x_train, x_test = features[trainidx], features[testidx]
        x_test -= x_train.mean(0)
        x_train -= x_train.mean(0)

        y_train, y_test = score[trainidx], score[testidx]
        # Linear regression
        regr = LogisticRegression(solver='newton-cg', max_iter=1000)
        regr.fit(x_train, y_train)
        # Predicted score
        p = regr.predict_proba(x_test)
        pred.append(p)

    pred = np.array(pred)
    pred = pred[:, :, 1]
    return pred.flatten()


def logreg_logo(features, score, groups=None):
    """Calculates logistic regression with leave-one-group-out split."""
    pred = []
    if groups is None:
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
                           9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
                           15, 16, 16, 17, 17, 18, 19, 19])
    # Leave one out split
    logo = LeaveOneGroupOut()
    logo.get_n_splits(features, score, groups)
    logo.get_n_splits(groups=groups)  # 'groups' is always required

    for trainidx, testidx in logo.split(features, score, groups):
        # Indices
        x_train, x_test = features[trainidx], features[testidx]
        x_test -= x_train.mean(0)
        x_train -= x_train.mean(0)

        y_train, y_test = score[trainidx], score[testidx]
        # Linear regression
        regr = LogisticRegression(solver='newton-cg', max_iter=1000)
        regr.fit(x_train, y_train)
        # Predicted score
        p = regr.predict_proba(x_test)
        pred.append(p)

    # pred = np.array(pred)
    # pred = pred[:,:,1]

    predflat = []
    for group in pred:
        for p in group:
            predflat.append(p)

    return np.array(predflat)[:, 1]


def regress_old(features, score):
    """Calculates linear regression with leave-one-out split. Obsolete."""
    pred = []
    # Leave one out split
    loo = LeaveOneOut()
    logo = LeaveOneGroupOut()
    for trainidx, testidx in loo.split(features):
        # Indices
        x_train, x_test = features[trainidx], features[testidx]
        x_test -= x_train.mean(0)
        x_train -= x_train.mean(0)

        y_train, y_test = score[trainidx], score[testidx]
        # Linear regression
        regr = Ridge(alpha=1, normalize=True, random_state=42)
        regr.fit(x_train, y_train)
        # Predicted score
        pred.append(regr.predict(x_test))

    return np.array(pred), regr.coef_
