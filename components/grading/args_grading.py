"""Contains parameters used in grading pipeline.

Includes also a "library" of different LBP parameters.
"""

import numpy as np
from argparse import ArgumentParser

# LBP parameters

# Abstract parameters
sparam_abs = {'ks1': 17, 'sigma1': 7, 'ks2': 17, 'sigma2': 1, 'N': 8, 'R': 23, 'r': 2, 'wc': 5, 'wl': 15, 'ws': 3}
dparam_abs = {'ks1': 15, 'sigma1': 3, 'ks2': 23, 'sigma2': 13, 'N': 8, 'R': 16, 'r': 12, 'wc': 13, 'wl': 15, 'ws': 9}
cparam_abs = {'ks1': 13, 'sigma1': 1, 'ks2': 23, 'sigma2': 7, 'N': 8, 'R': 19, 'r': 18, 'wc': 3, 'wl': 3, 'ws': 11}

# Components based on explained variance
# Trained on Insaf series surface grade (34 samples)
surf_90p = {'ks1': 15, 'sigma1': 1, 'ks2': 25, 'sigma2': 2, 'N': 8, 'R': 22, 'r': 20, 'wc': 7, 'wl': 13, 'ws': 7}
surf_95p = {'ks1': 5, 'sigma1': 1, 'ks2': 15, 'sigma2': 13, 'N': 8, 'R': 20, 'r': 14, 'wc': 13, 'wl': 5, 'ws': 11}
# Correlation (Equal to MSE training 90%)
surf_90p_corr = {'ks1': 15, 'sigma1': 1, 'ks2': 25, 'sigma2': 2, 'N': 8, 'R': 22, 'r': 20, 'wc': 7, 'wl': 13, 'ws': 7}
# 2mm 90%
surf_90p_2m = {'ks1': 15, 'sigma1': 9, 'ks2': 7, 'sigma2': 2, 'N': 8, 'R': 25, 'r': 18, 'wc': 15, 'wl': 13, 'ws': 9}
deep_mat_90p_2m = {'ks1': 21, 'sigma1': 4, 'ks2': 25, 'sigma2': 1, 'N': 8, 'R': 26, 'r': 7, 'wc': 5, 'wl': 9, 'ws': 11}
deep_cell_90p_2m_nocrop = {'ks1': 9, 'sigma1': 6, 'ks2': 23, 'sigma2': 2, 'N': 8, 'R': 14, 'r': 12, 'wc': 13, 'wl': 9, 'ws': 5}
deep_cell_90p_2m = {'ks1': 11, 'sigma1': 3, 'ks2': 25, 'sigma2': 1, 'N': 8, 'R': 22, 'r': 21, 'wc': 15, 'wl': 3, 'ws': 7}
calc_mat_90p_2m_nocrop = {'ks1': 21, 'sigma1': 7, 'ks2': 11, 'sigma2': 3, 'N': 8, 'R': 27, 'r': 1, 'wc': 3, 'wl': 3, 'ws': 7}
calc_mat_90p_2m = {'ks1': 15, 'sigma1': 7, 'ks2': 3, 'sigma2': 1, 'N': 8, 'R': 17, 'r': 2, 'wc': 11, 'wl': 11, 'ws': 13}
calc_vasc_90p_2m_nocrop = {'ks1': 19, 'sigma1': 15, 'ks2': 17, 'sigma2': 5, 'N': 8, 'R': 13, 'r': 11, 'wc': 5, 'wl': 9, 'ws': 15}
calc_vasc_90p_2m = {'ks1': 13, 'sigma1': 12, 'ks2': 19, 'sigma2': 9, 'N': 8, 'R': 17, 'r': 10, 'wc': 5, 'wl': 3, 'ws': 11}

surf_sum = {'ks1': 15, 'sigma1': 15, 'ks2': 13, 'sigma2': 3, 'N': 8, 'R': 27, 'r': 26, 'wc': 15, 'wl': 13, 'ws': 11}
deep_mat_sum = {'ks1': 23, 'sigma1': 19, 'ks2': 5, 'sigma2': 4, 'N': 8, 'R': 27, 'r': 6, 'wc': 3, 'wl': 15, 'ws': 9}
calc_mat_sum = {'ks1': 25, 'sigma1': 19, 'ks2': 3, 'sigma2': 3, 'N': 8, 'R': 16, 'r': 3, 'wc': 13, 'wl': 11, 'ws': 9}
deep_cell_sum = {'ks1': 11, 'sigma1': 3, 'ks2': 25, 'sigma2': 1, 'N': 8, 'R': 22, 'r': 21, 'wc': 15, 'wl': 3, 'ws': 7}

surf_loo = {'N': 8, 'R': 18, 'ks1': 25, 'ks2': 21, 'r': 4, 'sigma1': 4, 'sigma2': 7, 'wc': 15, 'wl': 15, 'ws': 13}
deep_loo = {'N': 8, 'R': 18, 'ks1': 25, 'ks2': 21, 'r': 4, 'sigma1': 4, 'sigma2': 7, 'wc': 15, 'wl': 15, 'ws': 13}
calc_loo = {'N': 8, 'R': 12, 'ks1': 23, 'ks2': 21, 'r': 11, 'sigma1': 4, 'sigma2': 6, 'wc': 9, 'wl': 9, 'ws': 15}

surf_4mm_loo = {'N': 8, 'R': 18, 'ks1': 25, 'ks2': 21, 'r': 4, 'seed': 42, 'sigma1': 4, 'sigma2': 7, 'wc': 15, 'wl': 15, 'ws': 13}
surf_4mm_loo2 = {'N': 8, 'R': 12, 'ks1': 23, 'ks2': 21, 'r': 11, 'seed': 42, 'sigma1': 4, 'sigma2': 6, 'wc': 9, 'wl': 9, 'ws': 15}
surf_4mm_loo = {'N': 8, 'R': 12, 'ks1': 23, 'ks2': 21, 'r': 11, 'seed': 42, 'sigma1': 4, 'sigma2': 6, 'wc': 9, 'wl': 9, 'ws': 15}
calc_4mm_loo = {'N': 8, 'R': 10, 'ks1': 7, 'ks2': 23, 'r': 8, 'seed': 42, 'sigma1': 2, 'sigma2': 13, 'wc': 7, 'wl': 5, 'ws': 9}

surf_loo_exp = {'N': 8, 'R': 18, 'ks1': 25, 'ks2': 21, 'r': 4, 'seed': 42, 'sigma1': 4, 'sigma2': 7, 'wc': 15, 'wl': 15, 'ws': 13}

surf_best_pars = {'N': 8, 'R': 24, 'ks1': 21, 'ks2': 17, 'r': 19, 'seed': 42, 'sigma1': 9, 'sigma2': 8, 'wc': 3, 'wl': 15, 'ws': 15}

# Grades pipeline is tested against
grades = ['surf_sub', 'deep_mat', 'deep_cell', 'deep_sub', 'calc_mat', 'calc_vasc', 'calc_sub']
grades_cut = ['surf_sub', 'deep_mat', 'calc_mat']

# Parameter sets
set_90p = [surf_90p, surf_90p, surf_90p, surf_90p, surf_90p, surf_90p, surf_90p]
set_95p = [surf_95p, surf_95p, surf_95p, surf_95p, surf_95p, surf_95p, surf_95p]
set_90p_2m = [surf_90p_2m, deep_mat_90p_2m, deep_cell_90p_2m, deep_mat_90p_2m, calc_mat_90p_2m, calc_vasc_90p_2m, calc_mat_90p_2m]
set_90p_2m_cut_nocrop = [surf_90p_2m, deep_mat_90p_2m, calc_mat_90p_2m_nocrop]
set_90p_2m_loo = [surf_loo, deep_loo, deep_loo, deep_loo, calc_loo, calc_loo, calc_loo]
set_2m_loo_cut = [surf_loo, deep_loo, calc_loo]
set_2m_rnsearch_cut = [surf_loo, deep_loo, calc_loo]
set_4mm_loo = [surf_4mm_loo, surf_4mm_loo, calc_4mm_loo]

surf_test = [surf_best_pars, surf_best_pars, surf_best_pars]  # Optimized for exp

# Patient groups
groups_2mm = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
                      15, 16, 16, 17, 18, 19, 19])  # 2mm, 34 patients


def return_args(root, choice, pars=set_2m_loo_cut, grade_list=grades_cut):
    """Returns arguments needed in grading pipeline. See grading_pipelines for specifications."""

    parser = ArgumentParser()
    parser.add_argument('--image_path', type=str, default=root + r'/MeanStd_' + choice)
    parser.add_argument('--feature_path', type=str, default=root + r'/Grading/Results/' + choice + r'/Features/')
    parser.add_argument('--save_path', type=str, default=root + r'/Grading/Results/' + choice)
    parser.add_argument('--grade_path', type=str, default=root + r'/Grading/trimmed_grades_' + choice + '.xlsx')
    parser.add_argument('--n_subvolumes', type=int, default=1)
    parser.add_argument('--logistic_limit', type=int, default=1)
    parser.add_argument('--log_pred_threshold', type=int, default=0.5)
    parser.add_argument('--n_jobs', type=int, default=10)
    parser.add_argument('--n_components', type=int, default=0.9)
    parser.add_argument('--str_components', type=str, default='90')
    parser.add_argument('--split', type=str, choices=['loo', 'logo', 'train_test', 'max_pool'], default='loo')
    parser.add_argument('--regression', type=str, choices=['lasso', 'ridge'], default='ridge')
    parser.add_argument('--standardization', type=str, choices=['standardize', 'centering'], default='centering')
    parser.add_argument('--convolution', type=bool, default=False)
    parser.add_argument('--normalize_hist', type=bool, default=True)
    parser.add_argument('--save_images', type=bool, default=True)
    parser.add_argument('--train_regression', type=bool, default=True)
    parser.add_argument('--auto_crop', type=bool, default=True)
    parser.add_argument('--GUI', type=bool, default=False)
    parser.add_argument('--median_filter', type=bool, default=False)
    parser.add_argument('--convert_grades', type=str, choices=['exp', 'log', 'none'], default='none')
    parser.add_argument('--pars', type=dict, default=pars)
    parser.add_argument('--grades_used', type=str, default=grade_list)
    parser.add_argument('--seed', type=int, default=42)  # Random seed
    parser.add_argument('--n_pars', type=int, default=100)  # Parameter optimization
    parser.add_argument('--n_bootstrap', type=int, default=2000)  # Bootstrapping AUC
    return parser.parse_args()
