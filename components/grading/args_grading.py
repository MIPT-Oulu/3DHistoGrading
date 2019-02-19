import numpy as np
from argparse import ArgumentParser

# LBP parameters

# Abstract parameters
sparam_abs = {'ks1': 17, 'sigma1': 7, 'ks2': 17, 'sigma2': 1, 'N': 8, 'R': 23, 'r': 2, 'wc': 5, 'wl': 15, 'ws': 3}
dparam_abs = {'ks1': 15, 'sigma1': 3, 'ks2': 23, 'sigma2': 13, 'N': 8, 'R': 16, 'r': 12, 'wc': 13, 'wl': 15, 'ws': 9}
cparam_abs = {'ks1': 13, 'sigma1': 1, 'ks2': 23, 'sigma2': 7, 'N': 8, 'R': 19, 'r': 18, 'wc': 3, 'wl': 3, 'ws': 11}
# 5 PCA components
surf_5n = {'ks1': 5, 'sigma1': 2, 'ks2': 25, 'sigma2': 19, 'N': 8, 'R': 25, 'r': 18, 'wc': 13, 'wl': 3, 'ws': 13}
deep_mat_5n = {'ks1': 19, 'sigma1': 3, 'ks2': 5, 'sigma2': 4, 'N': 8, 'R': 27, 'r': 8, 'wc': 11, 'wl': 11, 'ws': 3}
deep_cell_5n = {'ks1': 9, 'sigma1': 6, 'ks2': 23, 'sigma2': 2, 'N': 8, 'R': 14, 'r': 12, 'wc': 13, 'wl': 9, 'ws': 5}
deep_sub_5n = {'ks1': 3, 'sigma1': 3, 'ks2': 19, 'sigma2': 2, 'N': 8, 'R': 5, 'r': 1, 'wc': 11, 'wl': 15, 'ws': 3}
calc_mat_5n = {'ks1': 23, 'sigma1': 11, 'ks2': 17, 'sigma2': 4, 'N': 8, 'R': 10, 'r': 8, 'wc': 7, 'wl': 13, 'ws': 11}
calc_vasc_5n = {'ks1': 21, 'sigma1': 3, 'ks2': 13, 'sigma2': 10, 'N': 8, 'R': 12, 'r': 9, 'wc': 3, 'wl': 11, 'ws': 9}
calc_sub_5n = {'ks1': 15, 'sigma1': 7, 'ks2': 23, 'sigma2': 22, 'N': 8, 'R': 18, 'r': 8, 'wc': 9, 'wl': 9, 'ws': 3}
# 10 PCA components (NCM)
surf_10n = {'ks1': 21, 'sigma1': 17, 'ks2': 25, 'sigma2': 20, 'N': 8, 'R': 26, 'r': 5, 'wc': 5, 'wl': 13, 'ws': 11}
deep_mat_10n = {'ks1': 9, 'sigma1': 6, 'ks2': 23, 'sigma2': 2, 'N': 8, 'R': 14, 'r': 12, 'wc': 13, 'wl': 9, 'ws': 5}
deep_cell_10n = {'ks1': 3, 'sigma1': 3, 'ks2': 21, 'sigma2': 3, 'N': 8, 'R': 26, 'r': 4, 'wc': 7, 'wl': 3, 'ws': 7}
calc_mat_10n = {'ks1': 23, 'sigma1': 16, 'ks2': 15, 'sigma2': 6, 'N': 8, 'R': 16, 'r': 2, 'wc': 9, 'wl': 7, 'ws': 7}
calc_vasc_10n = {'ks1': 23, 'sigma1': 20, 'ks2': 7, 'sigma2': 7, 'N': 8, 'R': 26, 'r': 11, 'wc': 13, 'wl': 5, 'ws': 15}
# 15 PCA components
surf_15n = {'ks1': 15, 'sigma1': 8, 'ks2': 13, 'sigma2': 6, 'N': 8, 'R': 3, 'r': 2, 'wc': 13, 'wl': 3, 'ws': 9}
deep_mat_15n = {'ks1': 17, 'sigma1': 8, 'ks2': 11, 'sigma2': 1, 'N': 8, 'R': 25, 'r': 5, 'wc': 13, 'wl': 13, 'ws': 3}
deep_cell_15n = {'ks1': 7, 'sigma1': 4, 'ks2': 9, 'sigma2': 3, 'N': 8, 'R': 18, 'r': 12, 'wc': 11, 'wl': 11, 'ws': 3}
deep_sub_15n = {'ks1': 9, 'sigma1': 6, 'ks2': 23, 'sigma2': 2, 'N': 8, 'R': 14, 'r': 12, 'wc': 13, 'wl': 9, 'ws': 5}
calc_mat_15n = {'ks1': 23, 'sigma1': 5, 'ks2': 7, 'sigma2': 5, 'N': 8, 'R': 12, 'r': 11, 'wc': 5, 'wl': 5, 'ws': 15}
calc_vasc_15n = {'ks1': 15, 'sigma1': 1, 'ks2': 25, 'sigma2': 20, 'N': 8, 'R': 6, 'r': 1, 'wc': 13, 'wl': 3, 'ws': 9}
calc_sub_15n = {'ks1': 19, 'sigma1': 2, 'ks2': 21, 'sigma2': 18, 'N': 8, 'R': 15, 'r': 5, 'wc': 15, 'wl': 3, 'ws': 13}
# 20 PCA components
surf_20n = {'ks1': 3, 'sigma1': 3, 'ks2': 19, 'sigma2': 2, 'N': 8, 'R': 5, 'r': 1, 'wc': 11, 'wl': 15, 'ws': 3}
deep_mat_20n = {'ks1': 17, 'sigma1': 12, 'ks2': 21, 'sigma2': 4, 'N': 8, 'R': 7, 'r': 5, 'wc': 11, 'wl': 15, 'ws': 15}
deep_cell_20n = {'ks1': 23, 'sigma1': 2, 'ks2': 3, 'sigma2': 1, 'N': 8, 'R': 4, 'r': 1, 'wc': 15, 'wl': 3, 'ws': 9}
deep_sub_20n = {'ks1': 9, 'sigma1': 7, 'ks2': 21, 'sigma2': 18, 'N': 8, 'R': 21, 'r': 4, 'wc': 5, 'wl': 3, 'ws': 15}
calc_mat_20n = {'ks1': 13, 'sigma1': 9, 'ks2': 3, 'sigma2': 1, 'N': 8, 'R': 10, 'r': 3, 'wc': 11, 'wl': 3, 'ws': 11}
calc_vasc_20n = {'ks1': 23, 'sigma1': 13, 'ks2': 23, 'sigma2': 7, 'N': 8, 'R': 12, 'r': 5, 'wc': 7, 'wl': 13, 'ws': 11}
calc_sub_20n = {'ks1': 11, 'sigma1': 5, 'ks2': 21, 'sigma2': 14, 'N': 8, 'R': 15, 'r': 5, 'wc': 13, 'wl': 5, 'ws': 13}

# Components based on explained variance
# Trained on Insaf series surface grade (34 samples)
surf_90p = {'ks1': 15, 'sigma1': 1, 'ks2': 25, 'sigma2': 2, 'N': 8, 'R': 22, 'r': 20, 'wc': 7, 'wl': 13, 'ws': 7}
surf_95p = {'ks1': 5, 'sigma1': 1, 'ks2': 15, 'sigma2': 13, 'N': 8, 'R': 20, 'r': 14, 'wc': 13, 'wl': 5, 'ws': 11}
# Correlation (Equal to MSE training 90%)
surf_90p_corr = {'ks1': 15, 'sigma1': 1, 'ks2': 25, 'sigma2': 2, 'N': 8, 'R': 22, 'r': 20, 'wc': 7, 'wl': 13, 'ws': 7}
# 2mm 90%
surf_90p_2m = {'ks1': 15, 'sigma1': 9, 'ks2': 7, 'sigma2': 2, 'N': 8, 'R': 25, 'r': 18, 'wc': 15, 'wl': 13, 'ws': 9}
deep_mat_90p_2m = {'ks1': 21, 'sigma1': 4, 'ks2': 25, 'sigma2': 1, 'N': 8, 'R': 26, 'r': 7, 'wc': 5, 'wl': 9, 'ws': 11}
deep_cell_90p_2m = {'ks1': 9, 'sigma1': 6, 'ks2': 23, 'sigma2': 2, 'N': 8, 'R': 14, 'r': 12, 'wc': 13, 'wl': 9, 'ws': 5}
calc_mat_90p_2m = {'ks1': 21, 'sigma1': 7, 'ks2': 11, 'sigma2': 3, 'N': 8, 'R': 27, 'r': 1, 'wc': 3, 'wl': 3, 'ws': 7}

# Grades pipeline is tested against
grades = ['surf_sub', 'deep_mat', 'deep_cell', 'deep_sub', 'calc_mat', 'calc_vasc', 'calc_sub']
grades_cut = ['surf_sub', 'deep_mat', 'calc_mat']

# Parameter sets
set_5 = [surf_5n, deep_mat_5n, deep_cell_5n, deep_sub_5n, calc_mat_5n, calc_vasc_5n, calc_sub_5n]
set_10 = [surf_10n, deep_mat_10n, deep_cell_10n, deep_sub_15n, calc_mat_10n, calc_vasc_10n, calc_sub_15n]
set_15 = [surf_15n, deep_mat_15n, deep_cell_15n, deep_sub_15n, calc_mat_15n, calc_vasc_15n, calc_sub_15n]
set_20 = [surf_20n, deep_mat_20n, deep_cell_20n, deep_sub_20n, calc_mat_20n, calc_vasc_20n, calc_sub_20n]
set_90p = [surf_90p, surf_90p, surf_90p, surf_90p, surf_90p, surf_90p, surf_90p]
set_95p = [surf_95p, surf_95p, surf_95p, surf_95p, surf_95p, surf_95p, surf_95p]
set_90p_2m = [surf_90p_2m, deep_mat_90p_2m, deep_cell_90p_2m, deep_mat_90p_2m, calc_mat_90p_2m, calc_mat_90p_2m, calc_mat_90p_2m]

# Patient groups
groups_2mm = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
                      15, 16, 16, 17, 18, 19, 19])  # 2mm, 34 patients


def return_args(root, choice, pars=set_90p, grade_list=grades_cut):
    """Returns arguments needed in grading pipeline."""

    parser = ArgumentParser()
    parser.add_argument('--image_path', type=str, default=root + r'\MeanStd_' + choice)
    parser.add_argument('--feature_path', type=str, default=root + r'\Grading\LBP\\' + choice + r'\Features_')
    parser.add_argument('--save_path', type=str, default=root + r'\Grading\LBP\\' + choice)
    parser.add_argument('--grade_path', type=str, default=root + r'\Grading\trimmed_grades_' + choice + '.xlsx')
    parser.add_argument('--n_subvolumes', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=12)
    parser.add_argument('--n_components', type=int, default=0.9)
    parser.add_argument('--str_components', type=str, default='90')
    parser.add_argument('--split', type=str, choices=['loo', 'logo', 'train_test', 'max_pool'], default='loo')
    parser.add_argument('--convolution', type=bool, default=False)
    parser.add_argument('--normalize_hist', type=bool, default=True)
    parser.add_argument('--pars', type=dict, default=pars)
    parser.add_argument('--grades_used', type=str, default=grade_list)
    return parser.parse_args()
