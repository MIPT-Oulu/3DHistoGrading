#!/bin/bash

# Bash variables
ENVPATH=/media/santeri/3CE2190FE218CED0/Users/Santeri/pyenv/bin/activate
LOGPATH=/run/user/1003/gvfs/smb-share:server=nili,share=dios2\$/3DHistoData/Logs/
DAY=$(date +%Y-%m-%d_%H%M%S)
# Python pipeline variables
DATASET="Isokerays"
SUBVOLUMES=1 # Use 1 subvolume to calculate one large image/sample (subimages can be calculated in run_grading.sh)
SIZE_WIDE=None # Input x width for edge cropping (symmetric if None, 640 used for Test set 1)
ROOT=/run/user/1003/gvfs/smb-share:server=nili,share=dios2\$/3DHistoData
SAVE_PATH="${ROOT}/MeanStd_${DATASET}_large"
MODEL_PATH="${ROOT}/components/segmentation/unet/"
SNAPSHOTS="${ROOT}/components/segmentation/2018_12_03_15_25/"

# Get virtual environment
source ${ENVPATH}

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Print
echo "Experiment date: $DAY, dataset name: $DATASET"
echo "Calculating mean and std images from µCT data..."

# Call processing script
python scripts/run_images.py \
--data_path ${ROOT} \
--save_image_path ${SAVE_PATH} \
--model_path ${MODEL_PATH} \
--snapshots ${SNAPSHOTS} \
--size_wide ${SIZE_WIDE} \
--n_subvolumes ${SUBVOLUMES} \
> "${LOGPATH}images_log.txt" # Save output and errors as log file. Show also live (tee)

echo "Done. Saved outputs to $LOGPATH."