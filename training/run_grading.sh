#!/bin/bash

# Bash variables
ENVPATH=/media/santeri/3CE2190FE218CED0/Users/Santeri/pyenv/bin/activate
LOGPATH=/run/user/1003/gvfs/smb-share:server=nili,share=dios2\$/3DHistoData/Logs/
DAY=$(date +%Y-%m-%d_%H%M%S)
# Python pipeline variables
DATASET="Isokerays"
SUBVOLUMES_X=4
SUBVOLUMES_Y=4
SUBVOLUMES=$((${SUBVOLUMES_X} * ${SUBVOLUMES_Y}))
ROOT=/run/user/1003/gvfs/smb-share:server=nili,share=dios2\$/3DHistoData
IMAGE_PATH="${ROOT}/MeanStd_${DATASET}"
FEATURE_PATH="${ROOT}/Grading/LBP/${DATASET}/Features/"
SAVE_PATH="${ROOT}/Grading/LBP/${DATASET}"
GRADE_PATH="${ROOT}/Grading/trimmed_grades_${DATASET}.xlsx"
LARGE_IMAGES="${ROOT}/Meanstd_${DATASET}_large"

# Get virtual environment
source ${ENVPATH}

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Print
echo "Experiment date: $DAY, dataset name: $DATASET"
echo "Calculating total of $SUBVOLUMES subvolumes..."

# Update subvolumes
python scripts/run_subvolume_images.py \
--data_path ${LARGE_IMAGES} \
--save_image_path ${IMAGE_PATH} \
--subvolumes_x ${SUBVOLUMES_X} \
--subvolumes_y ${SUBVOLUMES_Y} \
--n_subvolumes ${SUBVOLUMES} \
> "${LOGPATH}subvolume_log.txt"

echo "Running grading pipeline for 2D images..."

# Call grading script
python scripts/run_grading.py \
--image_path ${IMAGE_PATH} \
--feature_path ${FEATURE_PATH} \
--save_path ${SAVE_PATH} \
--grade_path ${GRADE_PATH} \
--save_images False \
--train_regression False \
--n_subvolumes ${SUBVOLUMES} \
> "${LOGPATH}grading_log.txt" # Save output and errors as log file. Show also live (tee)

echo "Done. Saved outputs to $LOGPATH."