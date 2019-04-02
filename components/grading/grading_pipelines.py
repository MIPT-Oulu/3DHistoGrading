"""Contains pipelines for calculating MRELBP features and training/evaluating regression models."""

import numpy as np
import matplotlib.pyplot as plt
import os

from time import time
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.signal import medfilt2d
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_curve, roc_auc_score, auc, r2_score, \
    precision_recall_fscore_support, f1_score, accuracy_score
from scipy.stats import spearmanr, wilcoxon

from components.grading.local_binary_pattern import local_normalize_abs as local_standard, MRELBP, Conv_MRELBP
from components.utilities.load_write import save_excel, load_vois_h5
from components.grading.pca_regression import scikit_pca, regress_logo, regress_loo, logistic_logo, logistic_loo, standardize
from components.utilities.load_write import load_binary_weights, write_binary_weights, load_excel
from components.utilities.misc import plot_array_3d, plot_array_2d, plot_array_3d_animation, print_images, auto_corner_crop, plot_histograms


def pipeline_lbp(args, files, parameters, grade_used):
    """Calculates LBP features from input image (mean + standard deviation).

    Performs local contrast normalization, calculates MRELBP and saves features to .xlsx file.

    Supports parallelization for decreased processing times.

    Parameters
    ----------
    args : Namespace
        All grading arguments parsed into a namespace:
        n_subvolumes = Amount of subvolumes input image is splitted into.
        n_jobs = Number of parallel workers.
        save_images = Choice whether to save LBP and input images.
        auto_crop = Choice whether to automatically crop deep and calcified input images.
        convolution = Choice whether to use MRELBP pipeline with or without convolution.
        normalize_hist = Choice whether to normalize MRELBP histograms by sum.
        convert_grades = Choice whether to predict optionally exp or log of grades.
        save_path = Path to save images and features.

    files : list
        List of input image datasets (as .h5)
    parameters : dict
        MRELBP parameters used. See MRELBP and local_standard
    grade_used : str
        Title of the predicted grade (should be given on first row of the Excel file).
    """
    # Start time
    start_time = time()

    for vol in range(args.n_subvolumes):
        if args.n_subvolumes > 1:
            print('Loading images from subvolume {0}'.format(vol))
            files_input = files[vol]
        else:
            print('Loading images...')
            files_input = files
        # Load and normalize images
        images_norm = (Parallel(n_jobs=args.n_jobs)(delayed(load_voi)  # Initialize
                       (args, files_input[i], grade_used, parameters, save_images=args.save_images, autocrop=args.auto_crop)
                                                         for i in range(len(files_input))))  # Iterable

        # Calculate features
        if args.convolution:
            features = (Parallel(n_jobs=args.n_jobs)(delayed(Conv_MRELBP)  # Initialize
                        (images_norm[i], parameters,  # LBP parameters
                         normalize=args.normalize_hist,
                         savepath=args.save_path + '/Images/LBP/',
                         sample=files_input[i][:-3] + '_' + grade_used)  # Save paths
                          for i in tqdm(range(len(files_input)), desc='Calculating LBP features')))  # Iterable
        else:
            features = (Parallel(n_jobs=args.n_jobs)(delayed(MRELBP)  # Initialize
                        (images_norm[i], parameters,  # LBP parameters
                         normalize=args.normalize_hist,
                         savepath=args.save_path + '/Images/LBP/',
                         sample=files_input[i][:-3] + '_' + grade_used,  # Save paths
                         save_images=args.save_images)
                          for i in tqdm(range(len(files_input)), desc='Calculating LBP features')))  # Iterable

        # Convert to array
        features = np.array(features).squeeze()

        # Save features
        if args.n_subvolumes > 1:
            save = args.save_path + '/Features/' + grade_used + '_' + str(vol) + '.xlsx'
        else:
            save = args.save_path + '/Features/' + grade_used + '.xlsx'
        save_excel(features.T, save, files_input)

    # Display spent time
    t = time() - start_time
    print('Elapsed time: {0}s'.format(t))


def pipeline_prediction(args, grade_name, pat_groups=None, check_samples=False, combiner=np.mean):
    """Gets predictions from saved MRELBP features.

    1. Loads features and ground truth from .xlsx file

    2. Sort samples alphabetically and remove zero features. Optional centering for features.

    3. PCA dimensionality reduction.

    4. Linear and logistic regression.

    5. Create result plots.

    Parameters
    ----------
    args : Namespace
        All grading arguments parsed into a namespace:
        n_subvolumes = Amount of subvolumes input image is splitted into.
        grade_path = Path to ground truth.
        feature_path = Path to MRELBP features.
        save_path = Path to save results.
        train_regression = Choice whether to train a new model or evaluate on an existing one.
        standardization = Choice whether to center features before PCA.
        split = Cross-validation split used in training the model.
        logistic_limit = Limit used to make logistic prediction.
        convert_grades = Choice whether to predict optionally exp or log of grades.

    grade_name : str
        Title of the predicted grade (should be given on first row of the Excel file).
    pat_groups : ndarray (1-dimensional)
        patient groups for training with leave-one-group-out -split.
    check_samples : bool
        Choice whether to print all names of ground truth and features.
        Used to make sure that features and ground truth match (debugging)
    combiner : function
        Method to combine predictions of multiple subimages. Defaults to mean of predictions.
        Other possibilities: np.max, np.median
    Returns
    -------
    Ground truth, logistic predictions (for ROC curves), mean standard error.
    """

    # Load grades to array
    grades, hdr_grades = load_excel(args.grade_path, titles=[grade_name])

    # Sort grades based on alphabetical order
    grades = np.array([grade for _, grade in sorted(zip(hdr_grades, grades.squeeze()), key=lambda var: var[0])])

    # Limit for logistic regression
    bound = args.logistic_limit

    # Load features from subvolumes
    subvolumes = args.n_subvolumes > 1
    if subvolumes:
        feature_list, means = [], []
        for vol in range(args.n_subvolumes):
            features, hdr_features = load_excel(args.feature_path + '/' + grade_name + '_' + str(vol) + '.xlsx')
            # Remove zero features
            features = features[~np.all(features == 0, axis=1)]
            feature_list.append(features)
            # Mean feature
            mean_sub = np.mean(features, 1)
            means.append(mean_sub)
        mean = np.mean(means, axis=0)
    # Load features without subvolumes
    else:
        features, hdr_features = load_excel(args.feature_path + grade_name + '.xlsx')
        # Remove zero features
        features = features[~np.all(features == 0, axis=1)]
        # Mean feature
        mean = np.mean(features, 1)

        # Check matching samples
        if check_samples:
            print('Loaded grades (g) and features (f)')
            for i in range(grades.shape[0]):
                print('g, {0}, \tf {1}\t g_s {2}'.format(hdr_grades[i], hdr_features[i], grades[i]))

    # Train regression models
    if args.train_regression:
        print('\nTraining regression model on: {0}'.format(grade_name))

        if bound != 1:
            print('Limit is set to {0}'.format(bound))
        # Define split
        if args.split == 'logo' and pat_groups is not None:
            lin_regressor = regress_logo
            log_regressor = logistic_logo
        elif args.split == 'loo' or pat_groups is None:
            lin_regressor = regress_loo
            log_regressor = logistic_loo
        else:
            raise Exception('No valid cross-validation split selected (see arguments)!')

        # PCA and Regression
        if subvolumes:
            preds_lin, preds_log, scores, eigenvecs_list, singulars_list = [], [], [], [], []
            ints_lin, ints_log, w_lin, w_log = [], [], [], []
            for vol in range(args.n_subvolumes):
                # Standardize features
                if args.standardization == 'centering':
                    features = feature_list[vol].T - mean
                else:
                    features = standardize(feature_list[vol], axis=0).T

                # PCA
                pca, score_sub = scikit_pca(features, args.n_components, whitening=True, solver='auto')
                # Use same number of components for each subimage
                if vol == 0:
                    args.n_components = score_sub.shape[1]

                # Regression
                pred_linear_sub, weights, intercept_lin = lin_regressor(score_sub, grades, groups=pat_groups,
                                                                    method=args.regression, convert=args.convert_grades)
                pred_logistic_sub, weights_log, intercept_log = log_regressor(score_sub, grades > bound, groups=pat_groups)

                # Append to lists
                preds_lin.append(pred_linear_sub)
                preds_log.append(pred_logistic_sub)
                scores.append(score_sub)
                eigenvecs_list.append(pca.components_)
                singulars_list.append(pca.singular_values_)
                ints_lin.append(intercept_lin)
                ints_log.append(intercept_log)
                w_lin.append(weights)
                w_log.append(weights_log)

            # Combine lists (e.g. average, max)
            pred_linear = combiner(np.array(preds_lin), axis=0)
            pred_logistic = combiner(np.array(preds_log), axis=0)
            score = combiner(np.array(scores), axis=0)
            eigenvectors = combiner(np.array(eigenvecs_list), axis=0)
            singular_values = combiner(np.array(singulars_list), axis=0) / np.sqrt(features.shape[1] - 1)
            intercept_lin = combiner(np.array(ints_lin), axis=0)
            intercept_log = combiner(np.array(ints_log), axis=0)
            weights = combiner(np.array(w_lin), axis=0)
            weights_log = combiner(np.array(w_log), axis=0)
        else:
            # Standardize features
            if args.standardization == 'centering':
                features = features.T - mean
            else:
                features = standardize(features, axis=0).T

            # PCA
            pca, score = scikit_pca(features, args.n_components, whitening=True, solver='auto')
            eigenvectors = pca.components_
            singular_values = pca.singular_values_ / np.sqrt(features.shape[1] - 1)

            # Regression
            pred_linear, weights, intercept_lin = lin_regressor(score, grades, groups=pat_groups,
                                                                method=args.regression, convert=args.convert_grades)
            pred_logistic, weights_log, intercept_log = log_regressor(score, grades > bound, groups=pat_groups)

        # Save calculated weights
        print(intercept_log, intercept_lin)
        model_root = os.path.dirname(args.save_path)
        write_binary_weights(model_root + '/' + grade_name + '_weights.dat',
                             score.shape[1],
                             eigenvectors,
                             singular_values,
                             weights.flatten(),
                             weights_log.flatten(),
                             mean,
                             [intercept_lin, intercept_log])

    # Use pretrained models
    else:
        print('\nEvaluating with saved model weights on: {0}\n'.format(grade_name))
        model_root = os.path.dirname(args.save_path)
        if args.n_subvolumes > 1:
            preds_lin, preds_log, scores = [], [], []
            for vol in range(args.n_subvolumes):
                pred_linear_sub, pred_logistic_sub, score_sub = evaluate_model(feature_list[vol], args,
                                                                model_root + '/' + grade_name + '_weights.dat')
                preds_lin.append(pred_linear_sub)
                preds_log.append(pred_logistic_sub)
                scores.append(score_sub)

            pred_linear = combiner(np.array(preds_lin), axis=0)
            pred_logistic = combiner(np.array(preds_log), axis=0)
            score = combiner(np.array(scores), axis=0)
        else:
            pred_linear, pred_logistic, score = evaluate_model(features, args, model_root + '/' + grade_name + '_weights.dat')



    # Reference for pretrained PCA
    # reference_regress(features, args, score, grade_name + '_weights.dat', pred_linear, pred_logistic)

    # Logistic statistics
    auc_logistic = roc_auc_score(grades > bound, pred_logistic)
    prec, recall, _, support = precision_recall_fscore_support(grades > bound, pred_logistic > args.log_pred_threshold, average='binary')
    f1 = f1_score(grades > bound, pred_logistic > args.log_pred_threshold)
    accuracy = accuracy_score(grades > bound, pred_logistic > args.log_pred_threshold)
    conf_matrix = confusion_matrix(grades > bound, pred_logistic > args.log_pred_threshold)

    # Spearman corr
    rho, pval = spearmanr(grades, pred_linear)
    # Wilcoxon p
    wilc = wilcoxon(grades, pred_linear)
    # R^2 value
    r2 = r2_score(grades, pred_linear.flatten())
    # Mean squared error
    mse_linear = mean_squared_error(grades, pred_linear)

    # Handle edge cases
    for p in range(len(pred_linear)):
        if pred_linear[p] < 0:
            pred_linear[p] = 0
        if pred_linear[p] > max(grades):
            pred_linear[p] = max(grades)

    # Save prediction
    stats = np.zeros(len(grades))
    stats[0] = mse_linear
    stats[2] = auc_logistic
    stats[3] = r2
    tuples = list(zip(hdr_grades, grades, pred_linear, abs(grades - pred_linear), pred_logistic, stats))
    writer = pd.ExcelWriter(args.save_path + r'\prediction_' + grade_name + '.xlsx')
    df1 = pd.DataFrame(tuples, columns=['Sample', 'Actual grade', 'Prediction', 'Difference', 'Logistic prediction',
                                        'MSE, auc_logistic, r^2'])
    df1.to_excel(writer, sheet_name='Prediction')
    writer.save()

    # Display results
    text_string = 'MSE: {0:.2f}\nSpearman, p: {1:.2f}, {2:.4f}\nWilcoxon sum, p: {3:.2f}, {4:.2f}\n$R^2$: {5:.2f}' \
        .format(mse_linear, rho, pval, wilc[0], wilc[1], r2)
    logistic_results = 'AUC: {0:.3f}\nPrecision: {1:.3f}\nRecall/sensitivity: {2:.3f}\nAccuracy: {3:.3f}\nf1 {4:.3f}' \
        .format(auc_logistic, prec, recall, accuracy, f1)
    print(text_string, '\n', logistic_results)
    print('Number of components: ', score.shape[1])
    save_lin = args.save_path + '/linear_' + grade_name + '_' + args.split
    # Draw linear plot
    plot_linear(grades, pred_linear, text_string, plt_title=grade_name, savepath=save_lin)

    # Plot PCA components
    save_pca = args.save_path + '/pca_' + grade_name + '_' + args.split
    save_pca_ani = args.save_path + '/pca_animation_' + grade_name + '_' + args.split
    if score.shape[1] == 3:
        plot_array_3d(score, savepath=save_pca, plt_title=grade_name, grades=grades)
        #plot_array_3d_animation(score, save_pca_ani, plt_title=grade_name, grades=grades)
    elif score.shape[1] == 2:
        plot_array_2d(score, savepath=save_pca, plt_title=grade_name, grades=grades)

    # Plot grade distributions
    plot_histograms(grades, plt_title=grade_name, savepath=args.save_path + '//distribution_' + grade_name)
    return grades, pred_logistic, conf_matrix


def evaluate_model(features, args, model_path):
    """Evaluates features on saved model.

    Parameters
    ----------
    features : ndarray
        MRELBP features used for predictions.
    args : Namespace
        standardization = Centering or centering and standardization of features.
    model_path : str
        Path to saved model.
    Returns
    -------
    Linear predictions, logistic predicitons, PCA components.
    """
    # Load model

    _, n_comp, eigen_vectors, sv_scaled, weights, weights_log, mean_feature, [intercept_lin, intercept_log] \
        = load_binary_weights(model_path)

    # Standardize features
    if args.standardization == 'centering':
        mean = np.mean(features, 1)
        features = features.T - mean
        # features = features.T - mean_feature
    else:
        features = standardize(features, axis=0).T

    # PCA
    score = np.matmul(features, eigen_vectors / sv_scaled)

    # Regression
    pred_linear = np.matmul(score, weights) + intercept_lin
    pred_logistic = np.matmul(score, weights_log) + intercept_log

    return pred_linear, pred_logistic, score


def reference_regress(features, args, pca_components, model, linear, logistic):
    """Shows differences between model evaluation and training."""
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


def load_voi(args, file, grade, par, save_images=False, autocrop=True):
    """Loads mean+std images and performs automatic artefact crop and grayscale normalization."""
    path = args.image_path
    save = args.save_path
    # Load images
    image_surf, image_deep, image_calc = load_vois_h5(path, file)

    # Select VOI
    if grade[:4] == 'surf':
        image = image_surf[:]
    elif grade[:4] == 'deep':
        if autocrop:
            image, cropped = auto_corner_crop(image_deep)
            if cropped:
                # print_crop(image_deep, image, file[:-3] + ' deep zone')
                print('Automatically cropped sample {0}, deep zone from shape: ({1}, {2}) to: ({3}, {4})'
                      .format(file[:-3], image_deep.shape[0], image_deep.shape[1], image.shape[0], image.shape[1]))
        else:
            image = image_deep[:]
    elif grade[:4] == 'calc':
        if autocrop:
            image, cropped = auto_corner_crop(image_calc)
            if cropped:
                # print_crop(image_calc, image, file[:-3] + ' calcified zone')
                print('Automatically cropped sample {0}, calcified zone from shape: ({1}, {2}) to: ({3}, {4})'
                      .format(file[:-3], image_calc.shape[0], image_calc.shape[1], image.shape[0], image.shape[1]))
        else:
            image = image_calc[:]
    else:
        raise Exception('Check selected zone!')

    # Median filtering for noisy images
    if args.median_filter:
        image = medfilt2d(image, 3)
    # Normalize
    image_norm = local_standard(image, par)
    # Save image
    if save_images:
        titles_norm = ['Mean + Std', '', 'Normalized']
        print_images((image, image, image_norm),
                     subtitles=titles_norm, title=file + ' Input',
                     save_path=save + '/Images/Input/', sample=file[:-3] + '_' + grade + '.png')
    return image_norm


def plot_linear(grades, pred_linear, text_string, plt_title, savepath=None, annotate=False, headers=None):
    """Plots linear predictions against ground truth."""
    # Choose color
    if plt_title[:4] == 'deep':
        color = (128 / 225, 160 / 225, 60 / 225)
        plt_title = 'Deep zone'
    elif plt_title[:4] == 'calc':
        color = (225 / 225, 126 / 225, 49 / 225)
        plt_title = 'Calcified zone'
    else:
        color = (132 / 225, 102 / 225, 179 / 225)
        plt_title = 'Surface zone'

    # Scatter plot actual vs prediction
    [slope, intercept] = np.polyfit(grades, pred_linear.flatten(), 1)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(grades, pred_linear.flatten(), linewidths=7, color=color)
    ax.plot(grades, slope * grades + intercept, '--', color='black')
    ax.set_xlabel('Actual grade', fontsize=24)
    ax.set_ylabel('Predicted', fontsize=24)
    ax.xaxis.set_ticks(np.arange(0, 3 + 1, step=1.0))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim([-0.1, 3.1])
    plt.ylim([-0.1, 3.1])
    plt.title(plt_title, fontsize=24)

    ax.text(0.05, 0.95, text_string, transform=ax.transAxes, fontsize=14, verticalalignment='top')
    if annotate and headers is not None:
        for k in range(len(grades)):
            txt = headers[k]
            ax.annotate(txt, xy=(grades[k], pred_linear[k]), color='r')
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def print_crop(image, image_crop, title=None, savepath=None):
    """Prints original and cropped images in 2 subplots."""
    fig = plt.figure(dpi=500)
    ax1 = fig.add_subplot(121)
    cax1 = ax1.imshow(image, cmap='gray')
    if not isinstance(image, np.bool_):
        cbar1 = fig.colorbar(cax1, ticks=[np.min(image), np.max(image)], orientation='horizontal')
        cbar1.solids.set_edgecolor("face")
    ax1 = fig.add_subplot(122)
    cax1 = ax1.imshow(image_crop, cmap='gray')
    if not isinstance(image_crop, np.bool_):
        cbar1 = fig.colorbar(cax1, ticks=[np.min(image_crop), np.max(image_crop)], orientation='horizontal')
        cbar1.solids.set_edgecolor("face")

    # Give plot a title
    if title is not None:
        fig.suptitle(title)

    # Save images
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", transparent=True)
    plt.show()
