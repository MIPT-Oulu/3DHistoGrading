# 3DHistoGrading

*Software for 3D grading osteochondral Phosphotungstic acid -stained tissue samples. We currently support only Windows.*

[![Build status](https://ci.appveyor.com/api/projects/status/6lbb2719xekk5rrx?svg=true "Build status")](https://ci.appveyor.com/project/sarytky/3dhistograding)
[![DOI](https://zenodo.org/badge/144540603.svg)](https://zenodo.org/badge/latestdoi/144540603)

![Analysis pipeline](../master/training/documentation/flowchart.PNG)

 
## Background
This repository contains a software prototype and training codes used to assess degenerative features of osteochondral samples.
Samples should be imaged with micro-computed tomography using Phosphotungstic acid stain. 
Detailed describtion for imaging and grading procedure can be found from [our previous paper](https://doi.org/10.1016/j.joca.2017.05.021):
 
*Nieminen HJ, Gahunia HK, Pritzker KPH, et al. 
3D histopathological grading of osteochondral tissue using contrast-enhanced micro-computed tomography. 
Osteoarthritis Cartilage. 2017;25(10):1680-1689.*

The texture analysis methods used for feature extraction are implemented using our [LocalBinaryPattern](https://github.com/MIPT-Oulu/LocalBinaryPattern) repository.

More about the analysis procedure implemented in this repository can be found in [the article published on Osteoarthritis and Cartilage.](https://doi.org/10.1016/j.joca.2020.05.002)

 
## Prerequisites
* To avoid memory issues, the software runs on 64-bit systems only. 
* Until installer file is released, software has to be executed through MS Visual Studio. However, the compiled application can be provided upon request.


## Installation
* Currently, software can be used by opening 3DHistoGrading.sln on MS Visual Studio, compiling and running the project.
* We are planning to create an installer file that allows installing the software without additional dependencies.

## Application usage
Currently available features:
* Visualize 3D image datasets (.png, .tiff, .bmp) using 3D rendering and three orthogonal planes
* Load Mask on top of visualized dataset (mask should be registered with the dataset)
* Surface artefact cropping tool for coronal and sagittal plane
* Automatic sample alignment
* Automatic segmentation of bone-cartilage -interface using CNTK (Microsoft Cognitive Toolkit)
* Automatic bone (calcified tissues) and articular cartilage segmentation
* Automatic extraction of different volumes-of-interest (surface cartilage, deep cartilage and calcified tissue)
* Automatic grading from different osteochondral zones

## Outputs
* Result of degeneration detection (logistic regression) and corresponding µCT grade (ridge regression) from analysed sample volumes-of-interest
* Extracted analysis volumes can be saved as separate datasets
* Automatically segmented calcified tissue mask can be saved
* Sample data with performed processing steps

## Examples
[Screenshots from our software in action](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/master/pictures/examples.md)

## License
This software is distributed under the MIT License. This software and the pretrained models can be used only for research purposes.

## Citation
If you use the software or the source code in your work, please cite [our paper.](https://doi.org/10.1016/j.joca.2020.05.002)

```
@article {Rytky713800,
		title = {Automating Three-dimensional Osteoarthritis Histopathological Grading of Human Osteochondral Tissue using Machine Learning on Contrast-Enhanced Micro-Computed Tomography},
 author = {Rytky, S.J.O. and Tiulpin, A. and Frondelius, T. and Finnil{\"a}, M.A.J. and Karhula, S.S. and Leino, J. and Pritzker, K.P.H. and Valkealahti, M. and Lehenkari, P. and Joukainen, A. and Kr{\"o}ger, H. and Nieminen, H.J. and Saarakkala, S.},
 journal = {Osteoarthritis and Cartilage}
 year = {2020},
}
```
