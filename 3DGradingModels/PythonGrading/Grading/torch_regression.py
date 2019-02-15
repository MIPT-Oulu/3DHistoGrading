import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn

from time import time
from torch.autograd import Variable
from argparse import ArgumentParser
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score, auc, r2_score
from scipy.stats import spearmanr, wilcoxon

from Utilities.misc import duplicate_vector
from Utilities.load_write import load_excel
from Grading.roc_curve import roc_curve_bootstrap, mse_bootstrap
from Grading.pca_regression import scikit_pca, logistic_loo

# TODO Implement linear regression on Pytorch


class LinearRegression(nn.Module):
    def __init__(self, n_comp=15):

        # Super class constructor
        super(LinearRegression, self).__init__()
        # define Linear model
        self.linear = nn.Linear(n_comp, 1, bias=True)  # PCA comps in and prediction out

    def forward(self, x):
        # Forward pass
        pred_y = self.linear(x)
        return pred_y


def torch_regression(x_train, x_val, y_train, y_val, learning_rate=1e-5, savepath=None):
    # Model
    comps = x_train.shape[1]
    model = LinearRegression(n_comp=comps)

    # Variables
    x_train_tensor = Variable(torch.from_numpy(x_train.astype(np.float32)))
    y_train_tensor = Variable(torch.from_numpy(y_train.astype(np.float32)))

    # Loss, optimizer
    criterion = nn.MSELoss()  # Mean Squared Loss
    # criterion = nn.MultiLabelMarginLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Stochastic Gradient Descent

    # Train
    epochs = 5000
    for epoch in range(epochs):

        # Forward pass: Calculate prediction
        pred_y = model.forward(x_train_tensor)

        # Calculate loss
        loss = criterion(pred_y, y_train_tensor)

        # Zero gradients, backward pass, update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print('epochs {}, loss {}'.format(epochs, loss.item()))

    # Test
    x_val_tensor = Variable(torch.from_numpy(x_val.astype(np.float32)))
    predicted = model.forward(x_val_tensor).data.numpy()  # Convert prediction to numpy
    weights = model.linear.weight.data

    plt.plot(y_train, pred_y.data.numpy(), 'go', label='Training', alpha=.5)
    plt.plot(y_val, predicted, 'ro', label='Validation', alpha=0.5)
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    print(model.state_dict())
    # return np.concatenate((predicted.flatten(), pred_y.data.numpy().flatten())), weights.data.numpy()  # Return both preds
    return pred_y.data.numpy().flatten(), weights.data.numpy()


def pipeline(args, grade_name, train_group, test_group):
    # Check save path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Load grades to array
    grades_train, hdr_train_grade = load_excel(args.train_path, titles=[grade_name])
    grades_test, hdr_test_grade = load_excel(args.test_path, titles=[grade_name])

    # N of subvolumes
    n_test = args.n_subvolumes_test
    n_train = args.n_subvolumes_train

    # Duplicate grades for subvolumes
    grades_train = duplicate_vector(grades_train.squeeze(), n_train, reshape=True)
    hdr_train_grade = duplicate_vector(hdr_train_grade, n_train, reshape=True)
    grades_test = duplicate_vector(grades_test.squeeze(), n_test, reshape=True)
    hdr_test_grade = duplicate_vector(hdr_test_grade, n_test, reshape=True)

    # Load features
    f_train, hdr_train_f = load_excel(args.train_f_path + grade_name + '_' + str(args.n_components) + '.xlsx')
    f_test, hdr_test_f = load_excel(args.test_f_path + grade_name + '_' + str(args.n_components) + '.xlsx')

    # PCA
    pca_train, score_train = scikit_pca(f_train.T, args.n_components, whitening=True, solver='auto')
    pca_test, score_test = scikit_pca(f_test.T, args.n_components, whitening=True, solver='auto')

    # Reshape PCA score (sample, PCA component) --> (sample, subvolume, PCA component)
    dims = score_train.shape
    dims_test = score_test.shape
    score_train = np.reshape(score_train, (dims[0] // n_train, n_train, dims[1]))
    hdr_train_f = np.reshape(hdr_train_f, (dims[0] // n_train, n_train))
    score_test = np.reshape(score_test, (dims_test[0] // n_test, n_test, dims_test[1]))
    hdr_test_f = np.reshape(hdr_test_f, (dims_test[0] // n_test, n_test))

    # Linear and logistic regression
    pred_linear, weights = torch_regression(score_train, score_test, grades_train, grades_test,
                                            savepath=args.save_path + '\\torch_' + grade_name + '_' + str(args.n_components))

    # Combined grades
    # grades = np.concatenate((grades_train, grades_test))
    # hdr_grades = np.concatenate((hdr_train_grade,  hdr_test_grade))
    grades = grades_train
    hdr_grades = hdr_train_grade
    pred_linear = pred_linear.T

    # ROC curves
    fpr, tpr, thresholds = roc_curve(grades > 0, np.round(pred_linear) > 0, pos_label=1)
    auc_linear = auc(fpr, tpr)

    # Spearman corr
    rho = spearmanr(grades, pred_linear)
    # Wilcoxon p
    wilc = wilcoxon(grades, pred_linear)
    # R^2 value
    r2 = r2_score(grades, pred_linear.flatten())
    # Mean squared error
    mse_linear = mean_squared_error(grades, pred_linear)
    mse_boot, l_mse, h_mse = mse_bootstrap(grades, pred_linear)

    # Stats
    print('Mean squared error, Area under curve (linear)')
    print(mse_linear, auc_linear)
    print(r'Spearman: {0}, p: {1}, Wilcoxon p: {2}, r2: {3}'.format(rho[0], rho[1], wilc[1], r2))

    # Scatter plot actual vs prediction
    m, b = np.polyfit(grades, pred_linear.flatten(), 1)
    fig = plt.figure(figsize=(6, 6))
    ax2 = fig.add_subplot(111)
    ax2.scatter(grades, pred_linear.flatten())
    ax2.plot(grades, m * grades + b, '-', color='r')
    ax2.set_xlabel('Actual grade')
    ax2.set_ylabel('Predicted')
    text_string = 'MSE: {0:.2f}, [{1:.2f}, {2:.2f}]\nSpearman: {3:.2f}\nWilcoxon: {4:.2f}\n$R^2$: {5:.2f}' \
        .format(mse_boot, l_mse, h_mse, rho[0], wilc[1], r2)
    ax2.text(0.05, 0.95, text_string, transform=ax2.transAxes, fontsize=14, verticalalignment='top')
    for k in range(len(grades)):
        txt = hdr_grades[k] + str(grades[k])
        ax2.annotate(txt, xy=(grades[k], pred_linear[k]), color='r')
    plt.savefig(args.save_path + '\\linear_' + grade_name + '_' + str(args.n_components) + '_' + args.regression,
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Arguments
    choice = '2mm'
    pathbase = r'X:\3DHistoData\Grading\\'
    parser = ArgumentParser()
    parser.add_argument('--test_f_path', type=str, default=pathbase + r'\LBP\Insaf\Features_')
    parser.add_argument('--train_f_path', type=str, default=pathbase + r'\LBP\Isokerays\Features_')
    parser.add_argument('--grades_used', type=str,
                        default=['surf_sub',
                                 'deep_mat',
                                 'deep_cell',
                                 'deep_sub',
                                 'calc_mat',
                                 'calc_vasc',
                                 'calc_sub'
                                 ])
    parser.add_argument('--regression', type=str, choices=['loo', 'logo', 'train_test', 'max_pool'], default='logo')
    parser.add_argument('--save_path', type=str, default=pathbase + '\Tests\\')
    parser.add_argument('--n_components', type=int, default=5)
    parser.add_argument('--n_jobs', type=int, default=12)

    # Insaf (Test set)
    n_samples = 28
    groups_test = duplicate_vector(np.linspace(1, n_samples, num=n_samples), 2)
    parser.add_argument('--n_subvolumes_test', type=int, default=2)
    parser.add_argument('--test_path', type=str,
                        default=pathbase + 'trimmed_grades_Insaf.xlsx')

    # Isokeräys (Training set)
    n_samples = 14
    groups_train = duplicate_vector(np.linspace(1, n_samples, num=n_samples), 9)
    parser.add_argument('--n_subvolumes_train', type=int, default=9)
    parser.add_argument('--train_path', type=str,
                        default=pathbase + 'trimmed_grades_Isokerays.xlsx')
    arguments = parser.parse_args()

    # Start time
    start_time = time()

    # PCA and regression pipeline
    for title in arguments.grades_used:
        pipeline(arguments, title, groups_train, groups_test)

    # Display spent time
    t = time() - start_time
    print('Elapsed time: {0}s'.format(t))
