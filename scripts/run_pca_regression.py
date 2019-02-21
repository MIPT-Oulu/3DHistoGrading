import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from time import time
from argparse import ArgumentParser
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_curve, roc_auc_score, auc, r2_score

from components.grading.pca_regression import scikit_pca, regress_logo, regress_loo, logistic_logo, logistic_loo
from components.grading.roc_curve import mse_bootstrap, roc_curve_bootstrap, roc_curve_multi
from components.grading.torch_regression import torch_regression
from components.utilities.load_write import load_binary_weights, write_binary_weights, load_excel
from components.utilities.misc import duplicate_vector


def pipeline_prediction(args, grade_name, pat_groups=None, show_results=True, check_samples=False):
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
    if show_results:
        # Stats
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
    path = r'X:\3DHistoData\Grading\LBP\\' + choice + '\\'
    parser = ArgumentParser()
    parser.add_argument('--feature_path', type=str, default=path + '\\Features_')
    parser.add_argument('--grades_used', type=str,
                        default=['surf_sub',
                                 'deep_mat',
                              #   'deep_cell',
                              #   'deep_sub',
                                 'calc_mat',
                               #  'calc_vasc',
                               #  'calc_sub'
                                 ])
    parser.add_argument('--split', type=str, choices=['loo', 'logo', 'train_test', 'max_pool'], default='logo')
    parser.add_argument('--save_path', type=str, default=path)
    parser.add_argument('--n_components', type=int, default=0.9)
    parser.add_argument('--str_components', type=str, default='90')
    parser.add_argument('--n_jobs', type=int, default=12)

    if choice == 'Insaf':
        n_samples = 34
        groups = duplicate_vector(np.linspace(1, n_samples, num=n_samples), 2)
        parser.add_argument('--n_subvolumes', type=int, default=1)
        parser.add_argument('--grade_path', type=str,
                            default=r'X:\3DHistoData\Grading\trimmed_grades_' + choice + '.xlsx')
    elif choice == '2mm':
        # Patient groups
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
                           15, 16, 16, 17, 18, 19, 19])  # 2mm, 34 patients
        # groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
        #                   15, 16, 16, 17, 17, 18, 19, 19])  # 2mm, 36 patients
        parser.add_argument('--n_subvolumes', type=int, default=1)
        parser.add_argument('--grade_path', type=str,
                            default=r'X:\3DHistoData\Grading\trimmed_grades_' + choice + '.xlsx')
#        parser.add_argument('--grade_path', type=str, default=r'Y:\3DHistoData\Grading\ERCGrades.xlsx')
    else:
        n_samples = 14
        groups = duplicate_vector(np.linspace(1, n_samples, num=n_samples), 9)
        parser.add_argument('--n_subvolumes', type=int, default=9)
        parser.add_argument('--grade_path', type=str,
                            default=r'X:\3DHistoData\Grading\trimmed_grades_' + choice + '.xlsx')
    arguments = parser.parse_args()

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

   # # Receiver operating characteristics curve
   # split = arguments.split
   # #save_path = arguments.save_path
   # for i in range(len(arguments.grades_used)):
   #     lim = (np.min(gradelist[i]) + np.max(gradelist[i])) // 2
   #     grade_used = arguments.grades_used[i]
   #     print(grade_used)
   #     roc_curve_bootstrap(gradelist[i] > lim, preds[i], savepath=
   #     save_path + '\\roc_' + grade_used + '_' + arguments.str_components + '_' + split, lim=lim)

    split = arguments.split
    save_root = arguments.save_path
    lim = 1
    save_path = save_root + '\\roc_' + arguments.str_components + '_' + split
    roc_curve_multi(preds, gradelist, lim, save_path)


    # Display spent time
    t = time() - start_time
    print('Elapsed time: {0}s'.format(t))
