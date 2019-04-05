# 3D-Histo-Grading

## About
Software for 3D grading osteochondral Phosphotungstic acid -stained tissue samples. We currently support only Windows.
<!---
Current build status and code coverage:

[![Build status](https://ci.appveyor.com/api/projects/status/6lbb2719xekk5rrx?svg=true "Build status")](https://ci.appveyor.com/project/sarytky/3dhistograding)
[![codecov](https://codecov.io/gh/MIPT-Oulu/3D-Histo-Grading/branch/master/graph/badge.svg "Code coverage")](https://codecov.io/gh/MIPT-Oulu/3D-Histo-Grading)


User interface components are removed from code coverage analysis.
Unit testing is focused on software functionalities only.
 --->
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
Example images from 3D rendering and 2D viewing osteochondral sample.

![Rendering image](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/master/pictures/rendering.PNG "3D rendered image")
![Slice image](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/master/pictures/slice.PNG "2D coronal slice")

Oriented sample

![Orient image](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/master/pictures/slice_oriented.PNG "Oriented slice")

Artefact cropping

![Artefact image](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/master/pictures/artefact.PNG "Artefact cropping tool")

Results from automatic calcified zone segmentation.

![Rendering mask](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/master/pictures/rendering_mask.PNG "3D rendered image with mask")
![Slice mask](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/master/pictures/slice_mask.PNG "2D coronal slice with mask")

All extracted volumes-of-interest.

![VOI render](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/master/pictures/vois_render.PNG "3D rendered image with volumes-of-interest")
![VOI slice](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/master/pictures/vois.PNG "2D coronal slice with VOIs")

Example images from 3D grading process.
![Surface grading](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/master/pictures/surf_grading.PNG "Grafing window shows mean and standard deviation images from automatically selected volume-of-interest. Calculated LBP patterns are shown.")
![Deep grading](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/master/pictures/deep_grading_parameters_zoom.PNG "Used parameters can be checked by hovering mouse over parameters label. Windows support resizing to fullscreen.")
![Calcified grading](https://github.com/MIPT-Oulu/3D-Histo-Grading/blob/master/pictures/calc_grading.PNG "Separate windows are created for each zone.")

## License
This software is distributed under the MIT License. This software and the pretrained models can be used only for research purposes.

## Citation
```
@misc{3DGrading2018,
  author = {Frondelius, Tuomas and Rytky, Santeri and Tiulpin, Aleksei and Saarakkala Simo},
  title = {3D Histological grading},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/MIPT-Oulu/3D-Histo-Grading}},
}
```
