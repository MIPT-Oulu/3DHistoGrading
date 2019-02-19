import numpy as np
import matplotlib.pyplot as plt

from scipy import interp
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from tqdm.auto import tqdm


def roc_curve_multi(preds, targets, lim, savepath=None, seed=42):
    """ROC curve for three predictions."""

    fpr_surf, tpr_surf, _ = roc_curve(targets[0] > lim, preds[0])
    fpr_deep, tpr_deep, _ = roc_curve(targets[1] > lim, preds[1])
    fpr_calc, tpr_calc, _ = roc_curve(targets[2] > lim, preds[2])

    auc_surf = roc_auc_score(targets[0] > lim, preds[0])
    auc_deep = roc_auc_score(targets[1] > lim, preds[1])
    auc_calc = roc_auc_score(targets[2] > lim, preds[2])

    # Plot figure
    plt.figure(figsize=(11, 11))
    #red = (217 / 225, 95 / 225, 2 / 225)
    red = (225 / 225, 126 / 225, 49 / 225)
    #green = (217 / 225, 95 / 225, 2 / 225)
    green = (128 / 225, 160 / 225, 60 / 225)
    #blue = (117 / 225, 112 / 225, 179 / 225)
    blue = (132 / 225, 102 / 225, 179 / 225)
    plt.plot(fpr_surf, tpr_surf, color=blue, linewidth=3)
    plt.plot(fpr_deep, tpr_deep, color=green, linewidth=3)
    plt.plot(fpr_calc, tpr_calc, color=red, linewidth=3)
    plt.plot([0, 1], [0, 1], '--', color='black')
    plt.legend(['surface, AUC: {:0.3f}'.format(auc_surf),
                'deep, AUC: {:0.3f}'.format(auc_deep),
                'calcified, AUC: {:0.3f}'.format(auc_calc)], loc='lower right', fontsize=30)
    plt.ylabel('True Positive Rate', fontsize=36)
    plt.xlabel('False Positive Rate', fontsize=36)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.grid()
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


def mse_bootstrap(y, preds, savepath=None, n_bootstrap=1000, seed=42):
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


def roc_multi_bootstrap(y, preds, y2, preds2, y3, preds3, savepath=None, n_bootstrap=1000, seed=42):
    # 1
    np.random.seed(seed)
    aucs = []
    tprs = []
    base_fpr = np.linspace(0, 1, 1001)
    for _ in tqdm(range(n_bootstrap), total=n_bootstrap, desc='Bootstrap:'):
        ind = np.random.choice(y.shape[0], y.shape[0])
        if y[ind].sum() == 0:
            continue
        aucs.append(roc_auc_score(y[ind], preds[ind]))
        fpr, tpr, _ = roc_curve(y[ind], preds[ind])
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
    auc = np.mean(aucs)
    tprs = np.array(tprs)
    mean_tprs = np.mean(tprs, 0)
    std = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    CI_l, CI_h = np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

    # 2
    aucs2 = []
    tprs2 = []
    for _ in tqdm(range(n_bootstrap), total=n_bootstrap, desc='Bootstrap:'):
        ind = np.random.choice(y2.shape[0], y2.shape[0])
        if y2[ind].sum() == 0:
            continue
        aucs2.append(roc_auc_score(y2[ind], preds2[ind]))
        fpr2, tpr2, _ = roc_curve(y2[ind], preds2[ind])
        tpr2 = interp(base_fpr, fpr2, tpr2)
        tpr2[0] = 0.0
        tprs2.append(tpr2)
    auc2 = np.mean(aucs2)
    tprs2 = np.array(tprs2)
    mean_tprs2 = np.mean(tprs2, 0)
    std2 = np.std(tprs2, axis=0)
    tprs_upper2 = np.minimum(mean_tprs2 + std2, 1)
    tprs_lower2 = mean_tprs2 - std2
    CI_l2, CI_h2 = np.percentile(aucs2, 2.5), np.percentile(aucs2, 97.5)

    # 3
    aucs3 = []
    tprs3 = []
    for _ in tqdm(range(n_bootstrap), total=n_bootstrap, desc='Bootstrap:'):
        ind = np.random.choice(y3.shape[0], y3.shape[0])
        if y3[ind].sum() == 0:
            continue
        aucs3.append(roc_auc_score(y3[ind], preds3[ind]))
        fpr3, tpr3, _ = roc_curve(y3[ind], preds3[ind])
        tpr3 = interp(base_fpr, fpr3, tpr3)
        tpr3[0] = 0.0
        tprs3.append(tpr3)
    auc3 = np.mean(aucs3)
    tprs3 = np.array(tprs3)
    mean_tprs3 = np.mean(tprs3, 0)
    std3 = np.std(tprs3, axis=0)
    tprs_upper3 = np.minimum(mean_tprs3 + std3, 1)
    tprs_lower3 = mean_tprs3 - std3
    CI_l3, CI_h3 = np.percentile(aucs3, 2.5), np.percentile(aucs3, 97.5)

    # ROC without bootstrapping
    # sfpr,stpr,_ = roc_curve(y, pred, pos_label=1)
    # dfpr,dtpr,_ = roc_curve(y2, pred2, pos_label=1)
    # cfpr,ctpr,_ = roc_curve(y3, pred3, pos_label=1)

    plt.figure(figsize=(8, 8))
    # plt.title(f'AUC {np.round(auc, 2):.2f} 95% CI [{np.round(CI_l, 2):.2f}-{np.round(CI_h, 2):.2f}]')
    # plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='red', alpha=0.1)
    # plt.plot(base_fpr, mean_tprs, 'r-')
    plt.plot(fpr, tpr, 'r-')
    # plt.fill_between(base_fpr, tprs_lower2, tprs_upper2, color='green', alpha=0.1)
    # plt.plot(base_fpr, mean_tprs2, 'g-')
    plt.plot(fpr2, tpr2, 'g-')
    # plt.fill_between(base_fpr, tprs_lower3, tprs_upper3, color='blue', alpha=0.1)
    # plt.plot(base_fpr, mean_tprs3, 'b-')
    plt.plot(fpr3, tpr3, 'b-')
    plt.plot([0, 1], [0, 1], '--', color='black')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.tight_layout()
    plt.legend(['Surface, AUC: {:0.2f}'.format(auc),
                'Deep Zone, AUC: {:0.2f}'.format(auc2),
                'Calcified, AUC: {:0.2f}'.format(auc3)], loc='lower right')
    plt.rcParams.update({'font.size': 20})
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    print('AUC:', np.round(auc, 5))
    print(f'CI [{CI_l:.5f}, {CI_h:.5f}]')
    return auc, CI_l, CI_h