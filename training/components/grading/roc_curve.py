import numpy as np
import matplotlib.pyplot as plt

from scipy import interp
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from tqdm.auto import tqdm


def roc_curve_multi(preds, targets, lim, aucs=None, ci_l=None, ci_h=None, savepath=None, title=None):
    """ROC curve for three logistic regression predictions.

    Parameters
    ----------
    targets : numpy.array
        Ground truth
    preds : numpy.array
        Predictions
    savepath: str
        Where to save the figure with ROC curve
    lim : int
        Limit used for logistic regression.
    title : str
        Title for the plot.
    """

    fpr_surf, tpr_surf, _ = roc_curve(targets[0] > lim, preds[0])
    fpr_deep, tpr_deep, _ = roc_curve(targets[1] > lim, preds[1])
    fpr_calc, tpr_calc, _ = roc_curve(targets[2] > lim, preds[2])

    auc_surf = roc_auc_score(targets[0] > lim, preds[0])
    auc_deep = roc_auc_score(targets[1] > lim, preds[1])
    auc_calc = roc_auc_score(targets[2] > lim, preds[2])

    # Plot figure
    plt.figure(figsize=(11, 11))
    red = (225 / 225, 126 / 225, 49 / 225)
    green = (128 / 225, 160 / 225, 60 / 225)
    blue = (132 / 225, 102 / 225, 179 / 225)
    plt.plot(fpr_surf, tpr_surf, color=blue, linewidth=5)
    plt.plot(fpr_deep, tpr_deep, color=green, linewidth=5)
    plt.plot(fpr_calc, tpr_calc, color=red, linewidth=5)
    plt.plot([0, 1], [0, 1], '--', color='black')
    if aucs is None or ci_h is None or ci_l is None:
        plt.legend(['surface, AUC: {:0.2f}'.format(auc_surf),
                    'deep, AUC: {:0.2f}'.format(auc_deep),
                    'calcified, AUC: {:0.2f}'.format(auc_calc)], loc='lower right', fontsize=30)
    # Confidence intervals
    else:
        plt.legend(['Surface, AUC: {:0.2f}, ({:1.2f}, {:2.2f})'.format(aucs[0], ci_l[0], ci_h[0]),
                    'Deep, AUC: {:0.2f}, ({:1.2f}, {:2.2f})'.format(aucs[1], ci_l[1], ci_h[1]),
                    'Calcified, AUC: {:0.2f}, ({:1.2f}, {:2.2f})'.format(aucs[2], ci_l[2], ci_h[2])],
                   loc='lower right', fontsize=22)
    plt.ylabel('True Positive Rate', fontsize=36)
    plt.xlabel('False Positive Rate', fontsize=36)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    if title is not None:
        plt.title(title)
    plt.grid()
    plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def plot_vois(x, y, labels, savepath=None, location='lower right', axis_labels=None, baselines=None):
    colors = [(132 / 225, 102 / 225, 179 / 225),
              (128 / 225, 160 / 225, 60 / 225),
              (225 / 225, 126 / 225, 49 / 225)]

    # Plot figure
    plt.figure(figsize=(11, 11))
    for voi in range(3):
        plt.plot(x[voi], y[voi], color=colors[voi], linewidth=5)
    plt.legend(labels, loc=location, fontsize=20)

    if baselines is not None:
        for voi in range(3):
            plt.plot([0 - np.random.uniform(0, 0.1), 1], [baselines[voi], baselines[voi]], '--', color=colors[voi], alpha=0.5, linewidth=4)

    if axis_labels is not None:
        plt.xlabel(axis_labels[0], fontsize=36)
        plt.ylabel(axis_labels[1], fontsize=36)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def roc_curve_single(preds, targets, lim, savepath=None, title=None):
    """Plots ROC curve for given logistic regression prediction and ground truth.

    Parameters
    ----------
    targets : numpy.array
        Ground truth
    preds : numpy.array
        Predictions
    savepath: str
        Where to save the figure with ROC curve
    lim : int
        Limit used for logistic regression.
    title : str
        Title for the plot.
    """

    fpr, tpr, _ = roc_curve(targets > lim, preds)
    auc = roc_auc_score(targets > lim, preds)

    # Plot figure
    plt.figure(figsize=(11, 11))
    blue = (132 / 225, 102 / 225, 179 / 225)
    plt.plot(fpr, tpr, color=blue, linewidth=5)
    plt.plot([0, 1], [0, 1], '--', color='black')
    plt.legend(['AUC: {:0.2f}'.format(auc)], loc='lower right', fontsize=30)
    plt.ylabel('True Positive Rate', fontsize=36)
    plt.xlabel('False Positive Rate', fontsize=36)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    if title is not None:
        plt.title(title)
    plt.grid()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def roc_curve_bootstrap(y, preds, savepath=None, n_bootstrap=1000, seed=42, lim=None):
    """Evaluates ROC curve using bootstrapping

    Also reports confidence intervals and prints them.

    Parameters
    ----------
    y : numpy.array
        Ground truth
    preds : numpy.array
        Predictions
    savepath: str
        Where to save the figure with ROC curve
    n_bootstrap:
        Number of bootstrap samples to draw
    seed : int
        Random seed
    lim : int
        Limit used for logistic regression. If given, it is displayed in the plot.
    Returns
    -------
    Area under ROC curve, bootstrapping confidence intervals.
    """
    auc = roc_auc_score(y, preds)
    print('No bootstrapping: auc = {0}'.format(auc))
    np.random.seed(seed)
    aucs = []
    tprs = []
    base_fpr = np.linspace(0, 1, 1001)
    k = 0
    for _ in tqdm(range(n_bootstrap), total=n_bootstrap, desc='Bootstrap'):
        ind = np.random.choice(y.shape[0], y.shape[0])
        if y[ind].sum() == 0:
            continue
        try:
            aucs.append(roc_auc_score(y[ind], preds[ind]))
        except ValueError:
            k += 1
            continue
        fpr, tpr, _ = roc_curve(y[ind], preds[ind])
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
    if k > 0:
        print('{0} exceptions occurred. Check grade distribution'.format(k))
    auc = np.mean(aucs)
    print('Bootstrapping: auc = {0}'.format(auc))
    tprs = np.array(tprs)
    mean_tprs = np.mean(tprs, 0)
    std = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    CI_l, CI_h = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

    plt.figure(figsize=(8, 8))
    plt.title(f'AUC {np.round(auc, 2):.2f} 95% CI [{np.round(CI_l, 2):.2f}-{np.round(CI_h, 2):.2f}]')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)
    plt.plot(base_fpr, mean_tprs, 'r-')
    plt.plot([0, 1], [0, 1], '--', color='black')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    if lim is not None:
        text_string = 'Grade > {0}'.format(lim)
        plt.text(0.75, 0.25, text_string, fontsize=14, verticalalignment='top')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    print('AUC:', np.round(auc, 5))
    print(f'CI [{CI_l:.5f}, {CI_h:.5f}]')
    return auc, CI_l, CI_h


def calc_curve_bootstrap(curve, metric, y, preds, n_bootstrap, seed, stratified=True, alpha=95):
    """
    Method adapted from Aleksei Tiulpin, university of Oulu.
    Source: https://github.com/MIPT-Oulu/OAProgression

    Parameters
    ----------
    curve : function
        Function, which computes the curve.
    metric : fucntion
        Metric to compute, e.g. AUC for ROC curve or AP for PR curve
    y : numpy.array
        Ground truth
    preds : numpy.array
        Predictions
    n_bootstrap:
        Number of bootstrap samples to draw
    seed : int
        Random seed
    stratified : bool
        Whether to do a stratified bootstrapping
    alpha : float
        Confidence intervals width
    """

    np.random.seed(seed)
    metric_vals = []
    ind_pos = np.where(y == 1)[0]
    ind_neg = np.where(y == 0)[0]

    for _ in tqdm(range(n_bootstrap)):
        if stratified:
            ind_pos_bs = np.random.choice(ind_pos, ind_pos.shape[0])
            ind_neg_bs = np.random.choice(ind_neg, ind_neg.shape[0])
            ind = np.hstack((ind_pos_bs, ind_neg_bs))
        else:
            ind = np.random.choice(y.shape[0], y.shape[0])

        if y[ind].sum() == 0:
            continue
        metric_vals.append(metric(y[ind], preds[ind]))

    metric_val = np.mean(metric_vals)
    x_curve_vals, y_curve_vals, _ = curve(y, preds)
    ci_l = np.percentile(metric_vals, (100 - alpha) // 2)
    ci_h = np.percentile(metric_vals, alpha + (100 - alpha) // 2)

    print('Values for bootstrapped metric: {0}, [{1}, {2}]'.format(metric_val, ci_l, ci_h))

    return metric_val, ci_l, ci_h, x_curve_vals, y_curve_vals


def display_bootstraps(x_vals, y_vals, aucs, aucs_l, aucs_h, title=None, savepath=None):
    """
    Displays result of three bootstrapped ROC curves.
    See calc_curve_bootstrap.
    """
    # Check for three predictions
    if len(x_vals) != 3:
        raise Exception('Function optimized for three predictions!')

    # Plot figure
    plt.figure(figsize=(11, 11))
    red = (225 / 225, 126 / 225, 49 / 225)
    green = (128 / 225, 160 / 225, 60 / 225)
    blue = (132 / 225, 102 / 225, 179 / 225)
    plt.plot(x_vals[0], y_vals[0], color=blue, linewidth=5)
    plt.plot(x_vals[1], y_vals[1], color=green, linewidth=5)
    plt.plot(x_vals[2], y_vals[2], color=red, linewidth=5)
    plt.plot([0, 1], [0, 1], '--', color='black')
    plt.legend(['surface, AUC: {:0.3f}, [{:1.3f}, {:2.3f}]'.format(aucs[0], aucs_l[0], aucs_h[0]),
                'deep, AUC: {:0.3f}, [{:1.3f}, {:2.3f}]'.format(aucs[1], aucs_l[1], aucs_h[1]),
                'calcified, AUC: {:0.3f}, [{:1.3f}, {:2.3f}]'.format(aucs[2], aucs_l[2], aucs_h[2])],
               loc='lower right', fontsize=30)
    plt.ylabel('True Positive Rate', fontsize=36)
    plt.xlabel('False Positive Rate', fontsize=36)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    if title is not None:
        plt.title(title)
    plt.grid()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def mse_bootstrap(y, preds, savepath=None, n_bootstrap=1000, seed=42):
    """Calculates mean standard error, spearman rho, and wilcoxon p and confidence intervals using bootstrapping.
    Needs to be checked."""
    np.random.seed(seed)
    mses = []
    rhos = []
    wilcs = []
    for _ in tqdm(range(n_bootstrap), total=n_bootstrap, desc='Bootstrap:'):
        ind = np.random.choice(y.shape[0], y.shape[0])
        if y[ind].sum() == 0:
            continue
        rho = spearmanr(y[ind], preds[ind])
        wilc = wilcoxon(y[ind], preds[ind])
        MSE1 = mean_squared_error(y[ind], preds[ind])
        mses.append(MSE1)
        rhos.append(rho[0])
        wilcs.append(wilc[1])

    mse_m = np.mean(mses)
    rho_m = np.mean(rhos)
    wilc_m = np.mean(wilcs)

    CI_l_mse, CI_h_mse = np.percentile(mses, 2.5), np.percentile(mses, 97.5)
    CI_l_rho, CI_h_rho = np.percentile(rhos, 2.5), np.percentile(rhos, 97.5)
    CI_l_wilc, CI_h_wilc = np.percentile(wilcs, 2.5), np.percentile(wilcs, 97.5)

    print('MSE: {0}'.format(mse_m))
    print(f'CI [{CI_l_mse:.5f}, {CI_h_mse:.5f}]')
    print('Spearman: {0}'.format(rho_m))
    print(f'CI [{CI_l_rho:.5f}, {CI_h_rho:.5f}]')
    print('Wilcoxon: {0}'.format(wilc_m))
    print(f'CI [{CI_l_wilc:.5f}, {CI_h_wilc:.5f}]')
    return mse_m, CI_l_mse, CI_h_mse