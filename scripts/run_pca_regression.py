import numpy as np
import os
import matplotlib.pyplot as plt
import components.grading.args_grading as arg

from time import time
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_curve, roc_auc_score, auc, r2_score

from components.grading.pca_regression import scikit_pca, regress_logo, regress_loo, logistic_logo, logistic_loo, standardize
from components.grading.roc_curve import roc_curve_single, roc_curve_multi
from components.grading.torch_regression import torch_regression
from components.utilities.load_write import load_binary_weights, write_binary_weights, load_excel
from components.utilities.misc import duplicate_vector


def pipeline_prediction(args, grade_name, pat_groups=None, check_samples=False):
    # Load grades to array
    grades, hdr_grades = load_excel(args.grade_path, titles=[grade_name])

    # Duplicate grades for subvolumes
    grades = duplicate_vector(grades.squeeze(), args.n_subvolumes)
    hdr_grades = duplicate_vector(hdr_grades, args.n_subvolumes)
    # Sort grades based on alphabetical order
    grades = np.array([grade for _, grade in sorted(zip(hdr_grades, grades), key=lambda var: var[0])])

    # Load features
    features, hdr_features = load_excel(args.feature_path + grade_name + '_' + args.str_components + '.xlsx')
    # Mean feature
    mean = np.mean(features, 1)
    # Standardize features
    features = standardize(features, axis=0)

    # Check matching samples
    if check_samples:
        print('Loaded grades (g) and features (f)')
        for i in range(grades.shape[0]):
            print('g, {0}, \tf {1}\t g_s {2}'.format(hdr_grades[i], hdr_features[i], grades[i]))

    # Train regression models
    if args.train_regression:
        print('\nTraining regression model on: {0}'.format(grade_name))

        # Limit for logistic regression
        bound = (np.min(grades) + np.max(grades)) // 2
        if bound != 1:
            print('Limit is set to {0}'.format(bound))

        # PCA
        pca, score = scikit_pca(features.T, args.n_components, whitening=True, solver='auto')
        score = features.T
        # Linear and logistic regression
        if args.split == 'max_pool':
            split = 20
            pred_linear, weights = torch_regression(score[:split], score[split:], grades[:split], grades[split:])
            pred_logistic, weights_log, intercept_log = logistic_loo(score, grades > bound)

        elif args.split == 'logo' and pat_groups is not None:
            pred_linear, weights, intercept_lin = regress_logo(score, grades, pat_groups, method=args.regression)
            try:
                pred_logistic, weights_log, intercept_log = logistic_logo(score, grades > bound, pat_groups)
            except ValueError:
                print('Error on groups. Check grade distribution.')
                pred_logistic, weights_log, intercept_log = logistic_loo(score, grades > bound)

        elif args.split == 'loo' or pat_groups is None:
            pred_linear, weights, intercept_lin = regress_loo(score, grades, method=args.regression)
            pred_logistic, weights_log, intercept_log = logistic_loo(score, grades > bound)

        else:
            raise Exception('No valid regression method selected (see arguments)!')
        # Save calculated weights
        model_path = os.path.dirname(args.save_path)
        write_binary_weights(model_path + '/' + grade_name + '_weights.dat',
                             score.shape[1],
                             pca.components_,
                             pca.singular_values_ / np.sqrt(features.shape[1] - 1),
                             weights.flatten(),
                             weights_log.flatten(),
                             mean,
                             [intercept_lin, intercept_log])
    # Use pretrained models
    else:
        print('\nEvaluating with saved model weights on: {0}\n'.format(grade_name))
        # Load model
        model_path = os.path.dirname(args.save_path)
        _, n_comp, eigen_vectors, sv_scaled, weights, weights_log, mean_feature, [intercept_lin, intercept_log] \
            = load_binary_weights(model_path + '/' + grade_name + '_weights.dat')

        # PCA
        data_adjust = standardize(features, axis=0).T
        score = np.matmul(data_adjust, eigen_vectors / sv_scaled)

        # Regression
        pred_linear = np.matmul(score, weights) + intercept_lin
        pred_logistic = np.matmul(score, weights_log) + intercept_log

    # Handle edge cases
    for p in range(len(pred_linear)):
        if pred_linear[p] < 0:
            pred_linear[p] = 0
        if pred_linear[p] > max(grades):
            pred_linear[p] = max(grades)

    # Reference for pretrained PCA
    # reference_regress(features, args, score, grade_name + '_weights.dat', pred_linear, pred_logistic)

    # AUCs
    # auc_linear = auc(fpr, tpr)
    # auc_logistic = roc_auc_score(grades > lim, pred_logistic)

    # Spearman corr
    rho = spearmanr(grades, pred_linear)
    # Wilcoxon p
    wilc = wilcoxon(grades, pred_linear)
    # R^2 value
    r2 = r2_score(grades, pred_linear.flatten())
    # Mean squared error
    mse_linear = mean_squared_error(grades, pred_linear)

    # # Save prediction
    # stats = np.zeros(len(grades))
    # stats[0] = mse_linear
    # stats[1] = auc_linear
    # stats[2] = auc_logistic
    # stats[3] = r2
    # tuples = list(zip(hdr_grades, grades, pred_linear, abs(grades - pred_linear), pred_logistic, stats))
    # writer = pd.ExcelWriter(args.save_path + r'\prediction_' + grade_name + '.xlsx')
    # df1 = pd.DataFrame(tuples, columns=['Sample', 'Actual grade', 'Prediction', 'Difference', 'Logistic prediction',
    #                                     'MSE, auc_linear, auc_logistic, r^2'])
    # df1.to_excel(writer, sheet_name='Prediction')
    # writer.save()

    # Display results
    text_string = 'MSE: {0:.2f}\nSpearman: {1:.2f}\nWilcoxon: {2:.2f}\n$R^2$: {3:.2f}' \
        .format(mse_linear, rho[0], wilc[1], r2)
    save_fig = args.save_path + '\\linear_' + grade_name + '_' + args.str_components + '_' + args.split
    # Draw plot
    plot_linear(grades, pred_linear, text_string, grade_name, savepath=save_fig)

    return grades, pred_logistic, mse_linear


def reference_regress(features, args, pca_components, model, linear, logistic):
    _, _, eigenvec, singular_values, weight_lin, weight_log, m, std = load_binary_weights(args.save_path + '\\' + model)
    dataadjust = features.T - m
    pcaref = np.matmul(dataadjust,
                       eigenvec / singular_values.T)
    linear_ref = np.matmul(pcaref, weight_lin)
    log_ref = np.matmul(pcaref, weight_log)
    print('Difference between pretrained and trained method')
    pcaerr = np.sum(np.abs(pcaref - pca_components))
    linerr = np.sum(np.abs(linear_ref - linear))
    logerr = np.sum(np.abs(log_ref - logistic))
    print('Error on PCA: {0}'.format(pcaerr))
    print('Error on Linear regression: {0}'.format(linerr))
    print('Error on Logistic regression: {0}'.format(logerr))


def plot_linear(grades, pred_linear, text_string, plt_title, savepath=None, annotate=False, headers=None):
    # Scatter plot actual vs prediction
    [slope, intercept] = np.polyfit(grades, pred_linear.flatten(), 1)
    fig = plt.figure(figsize=(6, 6))
    ax2 = fig.add_subplot(111)
    ax2.scatter(grades, pred_linear.flatten(), linewidths=7, color=(132 / 225, 102 / 225, 179 / 225))
    ax2.plot(grades, slope * grades + intercept, '--', color='black')
    ax2.set_xlabel('Actual grade', fontsize=24)
    ax2.set_ylabel('Predicted', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title(plt_title)

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
    datapath = r'/run/user/1003/gvfs/smb-share:server=nili,share=dios2$/3DHistoData'
    arguments = arg.return_args(datapath, choice, pars=arg.set_90p_2m, grade_list=arg.grades_cut)
    arguments.train_regression = True
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
