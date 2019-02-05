import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Grading.pca_regression import scikit_pca, regress_logo, logistic_logo
from Grading.roc_curve import mse_bootstrap, roc_curve_bootstrap, roc_multi
from Grading.torch_regression import torch_regression
from Utilities.load_write import load_binary_weights, write_binary_weights, load_excel

from argparse import ArgumentParser
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_curve, roc_auc_score, auc, r2_score


def pipeline_load(args, voi_idx=0, show_results=True, check_samples=True, torch=False):

    # Load grades to array
    grades, hdr_grades = load_excel(args.grade_path, titles=args.grades_used)
    grades = grades[voi_idx, :]

    # Duplicate grades for subvolumes
    grades = duplicate_vector(grades, args.n_subvolumes)
    hdr_grades = duplicate_vector(hdr_grades, args.n_subvolumes)

    # Load features
    features, hdr_features = load_excel(args.voi_paths[voi_idx])
    # Mean feature
    mean = np.mean(features, 1)

    print(grades.shape)
    print(features.shape)
    ## Check matching samples
    #if check_samples:
    #    print('Loaded grades (g) and features (f)')
    #    for i in range(grades.shape[0]):
    #        print('g, {0}, \tf {1}'.format(hdr_grades[i], hdr_features[i]))

    # PCA
    pca, score = scikit_pca(features.T, args.n_components, whitening=True, solver='auto')

    # Patient groups
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8,
                       9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
                       15, 16, 16, 17, 18, 19, 19])

    # Linear regression
    if torch:
        split = 20
        pred_linear, weights = torch_regression(score[:split], score[split:], grades[:split], grades[split:])
    else:
        pred_linear, weights = regress_logo(score, grades, groups)
    for p in range(len(pred_linear)):
        if pred_linear[p] < 0:
            pred_linear[p] = 0
        if pred_linear[p] > max(grades):
            pred_linear[p] = max(grades)
    # Logistic regression
    pred_logistic = logistic_logo(score, grades > 1, groups)

    # Reference for pretrained PCA
    try:
        reference_regress(features, grades, mean, args, pca, weights, args.grades_used[voi_idx] + '_weights.dat')
    except:
        print('Reference model not found!')

    # ROC curves
    fpr, tpr, thresholds = roc_curve(grades > 0, np.round(pred_linear) > 0, pos_label=1)
    auc_linear = auc(fpr, tpr)
    auc_logistic = roc_auc_score(grades > 1, pred_logistic)

    # Spearman corr
    rho = spearmanr(grades, pred_linear)
    # Wilcoxon p
    wilc = wilcoxon(grades, pred_linear)
    # R^2 value
    r2 = r2_score(grades, pred_linear.flatten())
    # Mean squared error
    mse_linear = mean_squared_error(grades, pred_linear)
    mse_bootstrap(grades, pred_linear)
    # c1 = confusion_matrix(grades, np.round(pred_linear).astype('int'))

    # Save prediction
    stats = np.zeros(len(grades))
    stats[0] = mse_linear
    stats[1] = auc_linear
    stats[2] = auc_logistic
    stats[3] = r2
    tuples = list(zip(hdr_grades, grades, pred_linear, abs(grades - pred_linear), pred_logistic, stats))
    writer = pd.ExcelWriter(args.save_path + r'\prediction_' + args.grades_used[voi_idx] + '.xlsx')
    df1 = pd.DataFrame(tuples, columns=['Sample', 'Actual grade', 'Prediction', 'Difference', 'Logistic prediction',
                                        'MSE, auc_linear, auc_logistic, r^2'])
    df1.to_excel(writer, sheet_name='Prediction')
    writer.save()

    # Save calculated weights
    dataadjust = features.T - mean
    write_binary_weights(args.save_path + '\\' + args.grades_used[voi_idx] + '_weights.dat',
                         args.n_components,
                         pca.components_,
                         pca.singular_values_ / np.sqrt(dataadjust.shape[0] - 1),
                         weights,
                         mean)

    # Display results
    if show_results:
        # Stats
        print('Mean squared error, Area under curve (linear and logistic)')
        print(mse_linear, auc_linear, auc_logistic)
        print('Spearman: {0}, p: {1}, Wilcoxon p: {2}, r^2: {3}'.format(rho[0], rho[1], wilc[1], r2))

        # PCA scatter plot
        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_subplot(111)
        ax1.scatter(score[grades < 2, 0], score[grades < 2, 1], marker='o', color='b', label='Normal')
        ax1.scatter(score[grades > 1, 0], score[grades > 1, 1], marker='s', color='r', label='OA')
        for k in range(len(grades)):
            if grades[k] >= 2:
                ax1.scatter(score[k, 0], score[k, 1], marker='s', color='r')
            else:
                ax1.scatter(score[k, 0], score[k, 1], marker='o', color='b')

        # Scatter plot actual vs prediction
        m, b = np.polyfit(grades, pred_linear.flatten(), 1)
        fig = plt.figure(figsize=(6, 6))
        ax2 = fig.add_subplot(111)
        ax2.scatter(grades, pred_linear.flatten())
        ax2.plot(grades, m * grades + b, '-', color='r')
        ax2.set_xlabel('Actual grade')
        ax2.set_ylabel('Predicted')
        for k in range(len(grades)):
            txt = hdr_grades[k]
            txt = txt + str(grades[k])
            ax2.annotate(txt, xy=(grades[k], pred_linear[k]), color='r')
        plt.show()
    return grades, pred_logistic, mse_linear


def reference_regress(features, grades, mean, args, pca, weights, model):
    _, _, eigenvec, _, weightref, m = load_binary_weights(args.save_path + '\\' + model)
    dataadjust = features.T - mean
    # print('Mean vector')
    # print(mean)
    # print('Weights')
    # print(weights)
    # dataadjust = features.T
    # pca2, score2 = ScikitPCA(dataadjust, comps)
    # print(np.sum(np.abs(score2.flatten() - score.flatten())))
    pcaref = np.matmul(dataadjust,
                       pca.components_.T * np.sqrt(dataadjust.shape[0] - 1) / pca.singular_values_.T)
    # print('pcaref')
    # print(pcaref[0, :])
    # pcaref = np.matmul(dataadjust, eigenvec)
    # reference = np.matmul(pcaref, weightref)
    reference = np.matmul(pcaref, weights)
    # print('prediction')
    # print(reference)
    print('Sum of differences to actual grades (pretrained)')
    print(np.sum(np.abs((reference + 1.5).flatten() - grades)))


def duplicate_vector(vector, n):
    new_vector = []
    for i in range(len(vector)):
        for j in range(n):
            new_vector.append(vector[i])

    if isinstance(vector[0], type('str')):
        return new_vector
    else:
        return np.array(new_vector)


if __name__ == '__main__':
    # Arguments
    path = r'Y:\3DHistoData\Grading\LBP\2mm\\'
    parser = ArgumentParser()
    parser.add_argument('--voi_paths', type=str, default=
                        [path + 'LBP_features_surface.xlsx', path + 'LBP_features_deep.xlsx', path + 'LBP_features_calcified.xlsx'])
    parser.add_argument('--grade_path', type=str, default=r'Y:\3DHistoData\Grading\trimmed_grades_2mm.xlsx')
    parser.add_argument('--grades_used', type=dict, default=['surf_sub', 'deep_mat', 'calc_mat'])
    parser.add_argument('--save_path', type=str, default=r'Y:\3DHistoData\Grading\LBP\2mm')
    parser.add_argument('--n_components', type=int, default=20)
    parser.add_argument('--n_subvolumes', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=12)
    arguments = parser.parse_args()

    # PCA and regression pipeline
    gradelist = []
    preds = []
    mses = []
    for zone in range(len(arguments.grades_used)):
        grade, pred, mse = pipeline_load(arguments, zone)
        gradelist.append(grade)
        preds.append(pred)
        mses.append(mse)

    # Receiver operating characteristics curve
    roc_curve_bootstrap(gradelist[0] > 1, preds[0])
    roc_curve_bootstrap(gradelist[1] > 1, preds[1])
    roc_curve_bootstrap(gradelist[2] > 1, preds[2])
    # roc_multi(gradelist[0] > 1, preds[0], gradelist[1] > 1, preds[1], gradelist[2] > 1, preds[2], r'Y:\3DHistoData\Grading\ROC.png')
