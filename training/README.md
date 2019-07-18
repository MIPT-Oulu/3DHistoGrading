# 3DHistoGrading-Training
*Contains Python codes for grading models that are used in 3D-histo-Grading prototype software.*

(c) Santeri Rytky, University of Oulu, 2018-2019

<center>
<img src="https://github.com/sarytky/3DHistoGrading/blob/master/documentation/flowchart.PNG" width="900"/> 
</center>

## Background

This repository is used to create linear regression models to evaluate degeneration of osteochondral samples.
Samples should be imaged with micro-computed tomography using Phosphotungstic acid stain. 
Detailed describtion for imaging and grading procedure can be found from our previous paper:
 
*Nieminen HJ, Gahunia HK, Pritzker KPH, et al. 
3D histopathological grading of osteochondral tissue using contrast-enhanced micro-computed tomography. 
Osteoarthritis Cartilage. 2017;25(10):1680-1689.*

More about the analysis procedure implemented in this repository can be found in our upcoming paper.

## Prerequisites

Dependencies are loaded automatically in the bash scripts.

## Usage

### Preprocessing
Phosphotungstic acid -stained osteochondral samples can be preprocessed into mean and standard deviation images with script `run_images.sh`.

### Grading
Mean and standard deviation images can be used to calculate LBP images and estimate degeneration grades using script `run_grading.sh`.

### Full pipeline
These two steps are combined in script `run_full.sh`.

## License

This software and the pretrained models can be used only for research purposes.
