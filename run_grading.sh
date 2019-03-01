#!/bin/bash

WRKDIR=/run/user/1003/gvfs/smb-share:server=nili,share=dios2\$/3DHistoData

#docker build -t 3dhistograding_img .

# Install dependencies
pip install -r requirements.txt

echo "Running grading pipeline for 2D images..."

# If you run it first time - remove the option "--from_cache".
#nvidia-docker run -it --name 3dhistograding_inference --rm \
#	      -v $WRKDIR:/workdir/:rw \
#	      3dhistograding_img
python run_grading.py --choice "2mm"\
                    --root /workdir

echo "Grades estimated"