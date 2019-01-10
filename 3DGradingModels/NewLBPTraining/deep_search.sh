#!/bin/bash

touch deeppars.txt
rm deeppars.txt

echo "Searching pars.."

python TrainLBP.py --path ../cartvoi_deep_new/ --crop 1 --grade_keys deep_mat --n_pars 1000 --n_jobs 12 >> deeppars.txt

echo "Done!!"
