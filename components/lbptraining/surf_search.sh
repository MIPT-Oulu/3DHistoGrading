#!/bin/bash

touch surfpars.txt
rm surfpars.txt

echo "Searching pars.."

python TrainLBP.py --path ../cartvoi_surf_new/ --crop 0 --grade_keys surf_sub --n_pars 1000 >> surfpars.txt

echo "Done!!"
