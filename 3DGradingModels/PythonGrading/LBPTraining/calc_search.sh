#!/bin/bash

touch calcpars.txt
rm calcpars.txt

echo "Searching pars.."

python TrainLBP.py --path ../cartvoi_calc_new/ --crop 1 --grade_keys calc_mat --n_pars 1000 >> calcpars.txt

echo "Done!!"
