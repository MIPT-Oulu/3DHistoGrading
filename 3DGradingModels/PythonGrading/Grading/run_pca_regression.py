import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Grading.pca_regression import scikit_pca, regress_logo, logistic_logo
from Grading.roc_curve import mse_bootstrap, roc_curve_bootstrap, roc_multi
from Utilities.load_write import load_binary_weights, write_binary_weights

from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_curve, roc_auc_score, auc, r2_score


def pipeline_load(feature_path, grades_path, save_path, choice, comps, model_path):
    # TODO implement grading for Insaf and Isokerays series

    # Load grades to array
    grades_array = pd.read_excel(grades_path, 'Sheet1')
    grades_array = pd.DataFrame(grades_array).values
    fnames = grades_array[:, 0].astype('str')
    grades = list(grades_array[:, choice].astype('int'))
    grades = np.array(grades)

    # Load features
    features = pd.read_excel(feature_path, 'LBP_features')
    features = pd.DataFrame(features).values.astype('int')
    mean = np.mean(features, 1)  # mean feature
    # if features.shape[1] != 36:
    #    features = features.T

    # PCA
    pca, score = scikit_pca(features.T, comps, whitening=True, solver='auto')
    # pca, score = PCA(features,10)

    # Regression
    pred_linear, weights = regress_logo(score, grades)
    # pred_linear = regress_new(score, g)
    pred_logistic = logistic_logo(score, grades > 1)
    for p in range(len(pred_linear)):
        if pred_linear[p] < 0:
            pred_linear[p] = 0
        if pred_linear[p] > max(grades):
            pred_linear[p] = max(grades)
    # Plotting PCA
    b = np.round(pred_linear).astype('int')

    # Reference for pretrained PCA
    _, _, eigenvec, _, weightref, m = load_binary_weights(save_path + model_path)
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
    print(np.sum(np.abs((reference + 1.5).flatten() - grades)))
    # print(reference)

    # ROC curve
    # c1 = confusion_matrix(g, b)
    mse1 = mean_squared_error(grades, pred_linear)
    fpr, tpr, thresholds = roc_curve(grades > 0, np.round(pred_linear) > 0, pos_label=1)
    auc1 = auc(fpr, tpr)
    auc2 = roc_auc_score(grades > 1, pred_logistic)
    m, b = np.polyfit(grades, pred_linear.flatten(), 1)
    # r2 = r2_score(g, pred_linear.flatten())
    # fig0  = plt.figure(figsize=(6,6))
    # ax0 = fig0.add_subplot(111)
    # ax0.plot(fpr,tpr)
    mse_bootstrap(grades, pred_linear)

    # Save prediction
    stats = np.zeros(len(grades))
    stats[0] = mse1
    stats[1] = auc1
    stats[2] = auc2
    tuples = list(zip(fnames, grades, pred_linear, abs(grades - pred_linear), pred_logistic, stats))
    writer = pd.ExcelWriter(save_path + r'\prediction_python.xlsx')
    df1 = pd.DataFrame(tuples, columns=['Sample', 'Actual grade', 'Prediction', 'Difference', 'Logistic prediction',
                                        'MSE, auc1, auc2'])
    df1.to_excel(writer, sheet_name='Prediction')
    writer.save()

    # Save calculated weights
    write_binary_weights(save_path + '\\' + feature_path[-12:-8] + '_weights.dat',
                         comps,
                         pca.components_,
                         pca.singular_values_ / np.sqrt(dataadjust.shape[0] - 1),
                         weights,
                         mean)

    # Spearman corr
    rho = spearmanr(grades, pred_linear)
    # Wilcoxon p
    wilc = wilcoxon(grades, pred_linear)
    print('Spearman: {0}, p: {1}, Wilcoxon p: {2}'.format(rho[0], rho[1], wilc[1]))

    # print('Confusion matrix')
    # print(c1)
    print('Mean squared error, Area under curve 1 and 2')
    print(mse1, auc1, auc2)  # ,MSE2,MSE3,MSE4)
    # print('r2 score')
    # print(r2)
    # print('Sample, grade, prediction')
    # for k in range(len(fnames)):
    #    print(fnames[k],a[k],pred_linear[k])#,pred3[k])

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
    ax2.scatter(grades, pred_linear.flatten())
    ax2.plot(grades, m * grades, '-', color='r')
    ax2.set_xlabel('Actual grade')
    ax2.set_ylabel('Predicted')
    for k in range(len(grades)):
        txt = fnames[k]
        txt = txt + str(grades[k])
        ax2.annotate(txt, xy=(grades[k], pred_linear[k]), color='r')
    plt.show()
    return grades, pred_logistic, mse1


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