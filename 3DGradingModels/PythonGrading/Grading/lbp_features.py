from scipy import interp
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm

from Grading.grading import *
#from grading_old import regress

from scipy.stats import spearmanr, wilcoxon

import sklearn.metrics as skmet


def pipeline_lbp(impath, savepath, save, pars, dtype='dat'):
    # Start time
    start_time = time.time()
    # Calculate MRELBP from dataset
    # Parameters
    # mapping = getmapping(dict['N']) # mapping

    # Save parameters
    writer = pd.ExcelWriter(save + r'\LBP_parameters.xlsx')
    print(pars)
    df1 = pd.DataFrame(pars, index=[0])
    df1.to_excel(writer)
    writer.save()

    files = os.listdir(impath)
    files.sort()
    if dtype == 'h5':
        images = Loadh5(impath, files)

    features = None  # Reset feature array

    for k in tqdm(range(len(files)), desc='Calculating LBP features'):
        # Load file
        if dtype == 'dat':
            if k > len(files) / 2 - 1:
                break
            file = os.path.join(impath, files[2 * k])
            try:
                Mz = loadbinary(file, np.float64)
            except:
                continue
            file = os.path.join(impath, files[2 * k + 1])
            try:
                sz = loadbinary(file, np.float64)
            except:
                continue
        elif dtype == 'mat':
            file = os.path.join(impath, files[k])
            try:
                file = sio.loadmat(file)
                Mz = file['Mz']
                sz = file['sz']
            except NotImplementedError:
                file = h5py.File(file)
                Mz = file['Mz'][()]
                sz = file['sz'][()]

        # Combine mean and sd images
        if dtype == 'h5':
            image = images[k]
            if np.shape(image)[0] != 400:
                image = image[24:-24, 24:-24]
        else:
            image = Mz + sz
        # Grayscale normalization
        # image = local_normalize(image,dict['ks1'],dict['sigma1'],dict['ks2'],dict['sigma2'])
        image = localstandard(image, pars['ks1'], pars['sigma1'], pars['ks2'], pars['sigma2'])
        plt.imshow(image)
        plt.show()
        print(image)
        # LBP
        hist, lbpIL, lbpIS, lbpIR = MRELBP(image, pars['N'], pars['R'], pars['r'], pars['wc'], (pars['wl'], pars['ws']))
        # hist = Conv_MRELBP(image,dict['N'],dict['R'],dict['r'],dict['wr'][0],dict['wr'][1] ,dict['wc'])
        if hist.shape[0] == 1:
            hist = hist.T
        # print(hist2.shape)
        # print(np.sum(abs(hist2-hist)))
        try:
            features = np.concatenate((features, hist), axis=1)
        except ValueError:
            features = hist
        # Save images
        # if dtype == 'dat':
        #    cv2.imwrite(savepath + '\\' + files[2 * k][:-9] + '.png', lbpIS)
        # else:
        #    cv2.imwrite(savepath + '\\' + files[k][:-9] + '.png', lbpIS)

        # Plot LBP images
        # plt.imshow(lbpIS); plt.show()
        # plt.imshow(lbpIL); plt.show()
        # plt.imshow(lbpIR); plt.show()

    # Save features
    writer = pd.ExcelWriter(save + r'\LBP_features_python.xlsx')
    df1 = pd.DataFrame(features)
    df1.to_excel(writer, sheet_name='LBP_features')
    writer.save()

    t = time.time() - start_time
    print('Elapsed time: {0}s'.format(t))
    return features


def pipeline_load(featurepath, gpath, save, choice, comps, modelpath):
    # Start time
    start_time = time.time()
    # Load grades to array
    grades = pd.read_excel(gpath, 'Sheet1')
    grades = pd.DataFrame(grades).values
    fnames = grades[:, 0].astype('str')
    g = list(grades[:, choice].astype('int'))
    g = np.array(g)
    # print('Max grade: {0}, min grade: {1}'.format(max(g), min(g)))

    # Load features
    features = pd.read_excel(featurepath, 'LBP_features')
    features = pd.DataFrame(features).values.astype('int')
    mean = np.mean(features, 1)  # mean feature
    # if features.shape[1] != 36:
    #    features = features.T

    # PCA
    # PCA parameters: whitening, svd solver (auto/full)
    # pca, score = ScikitPCA(features.T, comps, True, 'auto')
    pca, score = ScikitPCA(features.T, comps, True, 'auto')
    # pca, score = PCA(features,10)

    # Regression
    pred1, weights = regress_group(score, g)
    # pred1 = regress_new(score, g)
    pred2 = logreg_group(score, g > 1)
    for p in range(len(pred1)):
        if pred1[p] < 0:
            pred1[p] = 0
        if pred1[p] > max(g):
            pred1[p] = max(g)
    # Plotting PCA
    b = np.round(pred1).astype('int')

    # Reference for pretrained PCA
    _, _, eigenvec, _, weightref, m = loadbinaryweights(save + modelpath)
    dataadjust = features.T - mean
    print('Mean vector')
    print(mean)
    print('Weights')
    print(weights)
    # dataadjust = features.T
    # pca2, score2 = ScikitPCA(dataadjust, comps)
    # print(np.sum(np.abs(score2.flatten() - score.flatten())))
    print('n_samples')
    print(dataadjust.shape[0])
    pcaref = np.matmul(dataadjust,
                       pca.components_.T * np.sqrt(dataadjust.shape[0] - 1) / pca.singular_values_.T)
    print('pcaref')
    print(pcaref[0, :])
    # pcaref = np.matmul(dataadjust, eigenvec)
    # reference = np.matmul(pcaref, weightref)
    reference = np.matmul(pcaref, weights)
    print('prediction')
    print(reference)
    print('Sum of differences to actual grades (pretrained)')
    print(np.sum(np.abs((reference + 1.5).flatten() - g)))
    # print(reference)

    # ROC curve
    C1 = skmet.confusion_matrix(g, b)
    MSE1 = skmet.mean_squared_error(g, pred1)
    fpr, tpr, thresholds = skmet.roc_curve(g > 0, np.round(pred1) > 0, pos_label=1)
    AUC1 = skmet.auc(fpr, tpr)
    AUC2 = skmet.roc_auc_score(g > 1, pred2)
    m, b = np.polyfit(g, pred1.flatten(), 1)
    R2 = skmet.r2_score(g, pred1.flatten())
    # fig0  = plt.figure(figsize=(6,6))
    # ax0 = fig0.add_subplot(111)
    # ax0.plot(fpr,tpr)
    mse_bootstrap(g, pred1)

    # Save prediction
    stats = np.zeros(len(g))
    stats[0] = MSE1
    stats[1] = AUC1
    stats[2] = AUC2
    tuples = list(zip(fnames, g, pred1, abs(g - pred1), pred2, stats))
    writer = pd.ExcelWriter(save + r'\prediction_python.xlsx')
    df1 = pd.DataFrame(tuples, columns=['Sample', 'Actual grade', 'Prediction', 'Difference', 'Logistic prediction',
                                        'MSE, AUC1, AUC2'])
    df1.to_excel(writer, sheet_name='Prediction')
    writer.save()

    # Save calculated weights
    writebinaryweights(save + '\\' + featurepath[-12:-8] + '_weights.dat'
                       , comps, pca.components_, pca.singular_values_ / np.sqrt(dataadjust.shape[0] - 1)
                       , weights, mean)

    # Spearman corr
    rho = spearmanr(g, pred1)
    # Wilcoxon p
    wilc = wilcoxon(g, pred1)
    print('Spearman: {0}, p: {1}, Wilcoxon p: {2}'.format(rho[0], rho[1], wilc[1]))

    # print('Confusion matrix')
    # print(C1)
    print('Mean squared error, Area under curve 1 and 2')
    print(MSE1, AUC1, AUC2)  # ,MSE2,MSE3,MSE4)
    # print('R2 score')
    # print(R2)
    # print('Sample, grade, prediction')
    # for k in range(len(fnames)):
    #    print(fnames[k],a[k],pred1[k])#,pred3[k])

    # x = score[:,0]
    # y = score[:,1]
    # fig = plt.figure(figsize=(6,6))
    # ax1 = fig.add_subplot(111)
    # ax1.scatter(score[g<2,0],score[g<2,1],marker='o',color='b',label='Normal')	
    # ax1.scatter(score[g>1,0],score[g>1,1],marker='s',color='r',label='OA')
    # for k in range(len(g)):
    #    txt = fnames[k][0:-4]+str(g[k])
    #    if g[k] >= 2:
    #        ax1.scatter(x[k],y[k],marker='s',color='r')
    #    else:
    #        ax1.scatter(x[k],y[k],marker='o',color='b')

    # Scatter plot actual vs prediction
    fig = plt.figure(figsize=(6, 6))
    ax2 = fig.add_subplot(111)
    ax2.scatter(g, pred1.flatten())
    ax2.plot(g, m * g, '-', color='r')
    ax2.set_xlabel('Actual grade')
    ax2.set_ylabel('Predicted')
    for k in range(len(g)):
        txt = fnames[k]
        txt = txt + str(g[k])
        ax2.annotate(txt, xy=(g[k], pred1[k]), color='r')
    plt.show()
    return g, pred2, MSE1


def roc_curve_bootstrap(y, preds, savepath=None, n_bootstrap=1000, seed=42):
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
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
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
        MSE1 = skmet.mean_squared_error(y[ind], preds[ind])
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


def roc_multi(y, preds, y2, preds2, y3, preds3, savepath=None, n_bootstrap=1000, seed=42):
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

    ## ROC without bootstrapping
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
    plt.show()
    plt.close()

    print('AUC:', np.round(auc, 5))
    print(f'CI [{CI_l:.5f}, {CI_h:.5f}]')
    return auc, CI_l, CI_h


# def roc_multi():
#    sfpr,stpr,_ = roc_curve(sgrades>1,surfp_log,pos_label=1)
#    dfpr,dtpr,_ = roc_curve(dgrades>1,deepp_log,pos_label=1)
#    cfpr,ctpr,_ = roc_curve(cgrades>1,calcp_log,pos_label=1)
#    sscore = roc_auc_score(sgrades>1,surfp_log)
#    dscore = roc_auc_score(dgrades>1,deepp_log)
#    cscore = roc_auc_score(cgrades>1,calcp_log)
#
#    plt.figure(figsize=(11,11))
#    plt.plot(sfpr,stpr,color='r')
#    plt.plot(dfpr,dtpr,color='g')
#    plt.plot(cfpr,ctpr,color='b')
#    plt.legend(['surface, AUC: {:0.3f}'.format(sscore),
#                'deep, AUC: {:0.3f}'.format(dscore),
#                'calcified, AUC: {:0.3f}'.format(cscore)],loc='lower right')
#    plt.ylabel('True Positive Rate')
#    plt.xlabel('False Positive Rate')
#    savepath = r'Z:\3DHistoData\Grading\ROC.png'
#    plt.savefig(savepath, bbox_inches='tight')
#    plt.show()

def Loadh5(impath, flist):
    # Image loading
    images = []

    for file in flist:
        h5 = h5py.File(os.path.join(impath, file), 'r')
        ims = h5['sum'][:]
        h5.close()
        images.append(ims)
    return images


featurepath1 = r'Z:\3DHistoData\Grading\LBP_features_surfnew.xlsx'
featurepath2 = r'Z:\3DHistoData\Grading\LBP_features_deepnew.xlsx'
featurepath3 = r'Z:\3DHistoData\Grading\LBP_features_calcnew.xlsx'
#featurepath3 = r'Z:\3DHistoData\Grading\LBP_features_calc_deeppar.xlsx'
gpath = r'Z:\3DHistoData\Grading\PTAgreiditjanaytteet.xls'
save = r'Z:\3DHistoData\Grading'

surf = 2
deepECM = 8
ccECM = 9

# Check this !!! #
ncomp = 20

print('Surface')
grade1, pred21, mse = pipeline_load(featurepath1, gpath, save, surf, ncomp, r'\surf_weights.dat')
print('Deep')
grade2, pred22, mse2 = pipeline_load(featurepath2, gpath, save, deepECM, ncomp, r'\deep_weights.dat')
print('Calcified')
grade3, pred23, mse3 = pipeline_load(featurepath3, gpath, save, ccECM, ncomp, r'\calc_weights.dat')
#print('{0}, {1}, {2}'.format(mse, mse2, mse3))
roc_curve_bootstrap(grade1>1, pred21)
roc_curve_bootstrap(grade2>1, pred22)
roc_curve_bootstrap(grade3>1, pred23)
roc_multi(grade1>1, pred21, grade2>1, pred22, grade3>1, pred23, r'Z:\3DHistoData\Grading\ROC.png')