import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import components.grading.args_grading as arg

from time import time
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_curve, roc_auc_score, auc, r2_score

from components.grading.pca_regression import scikit_pca, regress_logo, regress_loo, logistic_logo, logistic_loo
from components.grading.roc_curve import roc_curve_single, roc_curve_multi
from components.grading.torch_regression import torch_regression
from components.utilities.load_write import load_binary_weights, write_binary_weights, load_excel
from components.utilities.misc import duplicate_vector


def pipeline_prediction(args, grade_name, pat_groups=None, check_samples=False):
    print(grade_name)
    # Load grades to array
    grades, hdr_grades = load_excel(args.grade_path, titles=[grade_name])

    # Duplicate grades for subvolumes
    grades = duplicate_vector(grades.squeeze(), args.n_subvolumes)
    hdr_grades = duplicate_vector(hdr_grades, args.n_subvolumes)

    # Load features
    features, hdr_features = load_excel(args.feature_path + grade_name + '_' + args.str_components + '.xlsx')
    # Mean feature
    mean = np.mean(features, 1)

    # Check matching samples
    if check_samples:
        print('Loaded grades (g) and features (f)')
        for i in range(grades.shape[0]):
            print('g, {0}, \tf {1}'.format(hdr_grades[i], hdr_features[i]))

    # PCA
    pca, score = scikit_pca(features.T, args.n_components, whitening=True, solver='auto')
    print(score.shape)

    # Linear and logistic regression
    lim = (np.min(grades) + np.max(grades)) // 2
    if args.split == 'max_pool':
        split = 20
        pred_linear, weights = torch_regression(score[:split], score[split:], grades[:split], grades[split:])
        pred_logistic = logistic_loo(score, grades > lim)
    elif args.split == 'train_test':
        return
    elif args.split == 'logo' and pat_groups is not None:
        pred_linear, weights = regress_logo(score, grades, pat_groups)
        try:
            pred_logistic = logistic_logo(score, grades > lim, pat_groups)
        except ValueError:
            print('Error on groups. Check grade distribution.')
            pred_logistic = logistic_loo(score, grades > lim)
    elif args.split == 'loo' or pat_groups is None:
        pred_linear, weights = regress_loo(score, grades)
        pred_logistic = logistic_loo(score, grades > lim)
    else:
        raise Exception('No valid regression method selected (see arguments)!')

    # Handle edge cases
    for p in range(len(pred_linear)):
        if pred_linear[p] < 0:
            pred_linear[p] = 0
        if pred_linear[p] > max(grades):
            pred_linear[p] = max(grades)

    # Reference for pretrained PCA
    try:
        reference_regress(features, grades, mean, args, pca, weights, grade_name + '_weights.dat')
    except:
        print('Reference model not found!')

    # ROC curves
    fpr, tpr, thresholds = roc_curve(grades > 0, np.round(pred_linear) > 0, pos_label=1)
    auc_linear = auc(fpr, tpr)
    auc_logistic = roc_auc_score(grades > lim, pred_logistic)

    # Spearman corr
    rho = spearmanr(grades, pred_linear)
    # Wilcoxon p
    wilc = wilcoxon(grades, pred_linear)
    # R^2 value
    r2 = r2_score(grades, pred_linear.flatten())
    # Mean squared error
    mse_linear = mean_squared_error(grades, pred_linear)
    #mse_boot, l_mse, h_mse = mse_bootstrap(grades, pred_linear)
    # c1 = confusion_matrix(grades, np.round(pred_linear).astype('int'))

    # Save prediction
    stats = np.zeros(len(grades))
    stats[0] = mse_linear
    stats[1] = auc_linear
    stats[2] = auc_logistic
    stats[3] = r2
    tuples = list(zip(hdr_grades, grades, pred_linear, abs(grades - pred_linear), pred_logistic, stats))
    writer = pd.ExcelWriter(args.save_path + r'\prediction_' + grade_name + '.xlsx')
    df1 = pd.DataFrame(tuples, columns=['Sample', 'Actual grade', 'Prediction', 'Difference', 'Logistic prediction',
                                        'MSE, auc_linear, auc_logistic, r^2'])
    df1.to_excel(writer, sheet_name='Prediction')
    writer.save()

    # Save calculated weights
    dataadjust = features.T - mean
    write_binary_weights(args.save_path + '\\' + grade_name + '_weights.dat',
                         score.shape[1],
                         pca.components_,
                         pca.singular_values_ / np.sqrt(dataadjust.shape[0] - 1),
                         weights.flatten(),
                         mean)

    # Display results
    print('Mean squared error, Area under curve (linear and logistic)')
    print(mse_linear, auc_linear, auc_logistic)
    print(r'Spearman: {0}, p: {1}, Wilcoxon p: {2}, r2: {3}'.format(rho[0], rho[1], wilc[1], r2))
    text_string = 'MSE: {0:.2f}\nSpearman: {1:.2f}\nWilcoxon: {2:.2f}\n$R^2$: {3:.2f}' \
        .format(mse_linear, rho[0], wilc[1], r2)
    save_fig = args.save_path + '\\linear_' + grade_name + '_' + args.str_components + '_' + args.split
    # Draw plot
    plot_linear(grades, pred_linear, text_string, grade_name, savepath=save_fig)

    return grades, pred_logistic, mse_linear


def reference_regress(features, grades, mean, args, pca, weights, model):
    _, _, eigenvec, _, weightref, m = load_binary_weights(args.save_path + '\\' + model)
    dataadjust = features.T - mean
    pcaref = np.matmul(dataadjust,
                       pca.components_.T * np.sqrt(dataadjust.shape[0] - 1) / pca.singular_values_.T)
    reference = np.matmul(pcaref, weights)
    print('Sum of differences to actual grades (pretrained)')
    print(np.sum(np.abs((reference + 1.5).flatten() - grades)))


def plot_linear(grades, pred_linear, text_string, title, savepath=None, annotate=False, headers=None):
    # Scatter plot actual vs prediction
    m, b = np.polyfit(grades, pred_linear.flatten(), 1)
    fig = plt.figure(figsize=(6, 6))
    ax2 = fig.add_subplot(111)
    ax2.scatter(grades, pred_linear.flatten(), linewidths=7, color=(132 / 225, 102 / 225, 179 / 225))
    ax2.plot(grades, m * grades + b, '--', color='black')
    ax2.set_xlabel('Actual grade', fontsize=24)
    ax2.set_ylabel('Predicted', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title(title)

    ax2.text(0.05, 0.95, text_string, transform=ax2.transAxes, fontsize=14, verticalalignment='top')
    if annotate and headers is not None:
        for k in range(len(grades)):
            txt = headers[k]
            ax2.annotate(txt, xy=(grades[k], pred_linear[k]), color='r')
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # Arguments
    choice = '2mm'
    datapath = r'X:\3DHistoData'
    arguments = arg.return_args(datapath, choice, pars=arg.set_90p_2m, grade_list=arg.grades_cut)
    # LOGO for 2mm samples
    if choice == '2mm':
        arguments.split = 'logo'
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
                           15, 16, 16, 17, 18, 19, 19])  # 2mm, 34 patients
    else:
        groups = None

    # Start time
    start_time = time()

    # PCA and regression pipeline
    gradelist = []
    preds = []
    mses = []
    # Loop for surface, deep and calcified analysis
    for title in arguments.grades_used:
        grade, pred, mse = pipeline_prediction(arguments, title, pat_groups=groups)
        gradelist.append(grade)
        preds.append(pred)
        mses.append(mse)

    # Receiver operating characteristics curve
    if len(gradelist) == 3:
        split = arguments.split
        lim = 1
        save_path = arguments.save_path + '\\roc_' + arguments.str_components + '_' + split
        roc_curve_multi(preds, gradelist, lim, savepath=save_path)
    else:
        split = arguments.split
        for i in range(len(arguments.grades_used)):
            lim = (np.min(gradelist[i]) + np.max(gradelist[i])) // 2
            grade_used = arguments.grades_used[i]
            save_path = arguments.save_path + '\\roc_' + grade_used + '_' + arguments.str_components + '_' + split
            roc_curve_single(preds[i], gradelist[i], lim, savepath=save_path)

    # Display spent time
    t = time() - start_time
    print('Elapsed time: {0}s'.format(t))
