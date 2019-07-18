#!/bin/bash

# Bash variables
ENVPATH=/media/santeri/3CE2190FE218CED0/Users/Santeri/pyenv/bin/activate
LOGPATH=/run/user/1003/gvfs/smb-share:server=nili,share=dios2\$/3DHistoData/Logs/
DAY=$(date +%Y-%m-%d_%H%M%S)
# Python pipeline variables
DATASET="Isokerays"
SUBVOLUMES_PROCESS=1 # Use 1 subvolume to calculate one large image/sample (subimages can be calculated in run_grading.sh)
SUBVOLUMES_X=4
SUBVOLUMES_Y=4
SUBVOLUMES=$((${SUBVOLUMES_X} * ${SUBVOLUMES_Y}))
SIZE_WIDE=None # Input x width for edge cropping (symmetric if None, 640 used for Test set 1)
ROOT=/run/user/1003/gvfs/smb-share:server=nili,share=dios2\$/3DHistoData
SAVE_IMAGE_PATH="${ROOT}/MeanStd_${DATASET}"
MODEL_PATH="${ROOT}/components/segmentation/unet/"
SNAPSHOTS="${ROOT}/components/segmentation/2018_12_03_15_25/"
LARGE_IMAGES="${ROOT}/Meanstd_${DATASET}_large"
FEATURE_PATH="${ROOT}/Grading/LBP/${DATASET}/Features/"
SAVE_PATH="${ROOT}/Grading/LBP/${DATASET}"
GRADE_PATH="${ROOT}/Grading/trimmed_grades_${DATASET}.xlsx"

# Get virtual environment
source ${ENVPATH}

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Print
echo "Experiment date: $DAY, dataset name: $DATASET"
echo "Calculating mean and std images from µCT data..."

# Call precprocess script
python scripts/run_images.py \
--data_path ${ROOT} \
--save_image_path ${LARGE_IMAGES} \
--model_path ${MODEL_PATH} \
--snapshots ${SNAPSHOTS} \
--size_wide ${SIZE_WIDE} \
--n_subvolumes ${SUBVOLUMES_PROCESS} \
> "${LOGPATH}images_log.txt" # Save output and errors as log file. Show also live (tee)


# Print
echo "Calculating total of $SUBVOLUMES subvolumes..."

# Update subvolumes
python scripts/run_subvolume_images.py \
--data_path ${LARGE_IMAGES} \
--save_image_path ${SAVE_IMAGE_PATH} \
--subvolumes_x ${SUBVOLUMES_X} \
--subvolumes_y ${SUBVOLUMES_Y} \
--n_subvolumes ${SUBVOLUMES} \
> "${LOGPATH}subvolume_log.txt"

echo "Running grading pipeline for 2D images..."

# Call grading script
python scripts/run_grading.py \
--image_path ${SAVE_IMAGE_PATH} \
--feature_path ${FEATURE_PATH} \
--save_path ${SAVE_PATH} \
--grade_path ${GRADE_PATH} \
--save_images False \
--train_regression False \
--n_subvolumes ${SUBVOLUMES} \
> "${LOGPATH}grading_log.txt" # Save output and errors as log file. Show also live (tee)

echo "Done. Saved outputs to $LOGPATH."