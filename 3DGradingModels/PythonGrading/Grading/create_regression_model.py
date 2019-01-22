import time
import pandas as pd

from Grading.grading import *
from Grading.roc_curve import *
# from grading_old import regress

from scipy.stats import spearmanr, wilcoxon


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


if __name__ == '__main__':
    featurepath1 = r'Z:\3DHistoData\Grading\LBP_features_surfnew.xlsx'
    featurepath2 = r'Z:\3DHistoData\Grading\LBP_features_deepnew.xlsx'
    featurepath3 = r'Z:\3DHistoData\Grading\LBP_features_calcnew.xlsx'
    # featurepath3 = r'Z:\3DHistoData\Grading\LBP_features_calc_deeppar.xlsx'
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

    # Receiver operating characteristics curve
    roc_curve_bootstrap(grade1 > 1, pred21)
    roc_curve_bootstrap(grade2 > 1, pred22)
    roc_curve_bootstrap(grade3 > 1, pred23)
    roc_multi(grade1 > 1, pred21, grade2 > 1, pred22, grade3 > 1, pred23, r'Z:\3DHistoData\Grading\ROC.png')