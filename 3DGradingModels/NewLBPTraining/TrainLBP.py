import numpy as np
import os
import h5py
import cv2
import gc
from time import time
import pandas as pd
import gc


from argparse import ArgumentParser

from joblib import Parallel,delayed

from scipy.signal import medfilt
from scipy.ndimage import correlate, zoom

from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge

from Components import find_pars_bforce, make_pred



if __name__ == '__main__':
    #Arguments
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='../cartvoi_surf_new/')
    parser.add_argument('--path_grades', type=str, default='../ERCGrades.xlsx')
    parser.add_argument('--grade_keys', type=str, nargs='+', default='surf_sub')
    parser.add_argument('--grade_mode',type=str,choices = ['sum','mean'], default = 'mean')
    parser.add_argument('--n_pars',type=int, default=10)
    parser.add_argument('--classifier',type=str,choices=['ridge','random_forest'],default='ridge')
    parser.add_argument('--crop', type=int, default=0)
    parser.add_argument('--n_jobs',type=int, default=1)
    args = parser.parse_args()
    
    #Load images
    files = os.listdir(args.path)
    files.sort()

    images = []
    for file in files:
        h5 = h5py.File(os.path.join(args.path,file),'r')
        if args.crop == 1:
            images.append(h5['sum'][24:-24,24:-24])
        elif args.crop == 0:
            images.append(h5['sum'][:])
        h5.close()
    
    #Load grades
    df = pd.read_excel(args.path_grades)
    if type(args.grade_keys) == type('abc'):
        grades = np.array(df[args.grade_keys])
    else:
        for key in args.grade_keys:
            try:
                grades += np.array(df[key])

            except NameError:
                grades = np.array(df[key])

    #Optimize parameters
    pars,error = find_pars_bforce(images,grades,args.n_pars,args.grade_mode,args.n_jobs)
    
    print("Minimum error is : {0}".format(error))
    print("Parameters are:")
    print(pars)
