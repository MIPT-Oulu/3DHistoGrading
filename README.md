# 3D-Histo-Grading
## About
Software for 3D grading osteochondral tissue samples. We currently support only Windows.

Current build status and code coverage:

[![Build status](https://ci.appveyor.com/api/projects/status/6lbb2719xekk5rrx?svg=true "Build status")](https://ci.appveyor.com/project/sarytky/3dhistograding)
[![codecov](https://codecov.io/gh/MIPT-Oulu/3D-Histo-Grading/branch/master/graph/badge.svg "Code coverage")](https://codecov.io/gh/MIPT-Oulu/3D-Histo-Grading)

[![codecov](https://codecov.io/gh/MIPT-Oulu/3D-Histo-Grading/branch/master/graphs/icicle.svg "Code coverage graph. Top section represents entire project, middle section folders and bottom section individual files.")](https://codecov.io/gh/MIPT-Oulu/3D-Histo-Grading/tree/master/3DHistoGrading)


User interface components are removed from coverage analysis.
Unit testing is focused on software functionalities only.

## Prerequisites
To avoid memory issues, the software runs on 64-bit systems only. 

## Installation
* Download and extract the repository to local directory or clone repository
* Navigate to 3DHistoGrading/bin/64x/Debug and run 3DHistoGrading.exe
* Software can also be used by opening 3DHistoGrading.sln on MS Visual Studio

## Application usage
Currently available features:
* Visualize 3D image datasets (.png, .tiff, .bmp) using 3D rendering and three orthogonal planes
* Load Mask on top of visualized dataset (mask should be registered with the dataset)
* Automatically estimate surface grade from center of the sample
* Automatic segmentation of bone-cartilage -interface using CNTK

Features that are on development:
* 3D volume-of-interest selection
* Automatic bone and cartilage segmentation
  * False positive suppression
* Automatic segmentation of cartilage into different zones
* Automatic grading from different osteochondral zones

## Outputs
OA grade from analysed sample.

## Examples
Example images from 3D rendering and 2D viewing osteochondral sample.

![Rendering image](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/grading/pictures/rendering.PNG "3D rendered image")
![Slice image](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/grading/pictures/slice.PNG "2D coronal slice")

![Rendering mask](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/grading/pictures/rendering_mask.PNG "3D rendered image with mask")
![Slice mask](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/grading/pictures/slice_mask.PNG "2D coronal slice with mask")


Example images from 3D grading process.
![Grading image](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/grading/pictures/grading.PNG "Grafing window shows mean and standard deviation images from automatically selected volume-of-interest. Calculated LBP patterns are shown.")
![Grading param image](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/grading/pictures/grading_parameters.PNG "Used parameters can be checked by hovering mouse over parameters label.")

## License
This software is distributed under the MIT License.

## Citation
```
@misc{3DGrading2018,
  author = {Frondelius, Tuomas and Rytky, Santeri and Tiulpin, Aleksei and Saarakkala Simo},
  title = {Local Binary Pattern},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/MIPT-Oulu/3D-Histo-Grading}},
}
```
