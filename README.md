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
To avoid memory issues, the software runs on 64-bit systems only. 

## Installation
* Download and extract the repository to local directory or clone repository
* Navigate to 3DHistoGrading/bin/64x/Debug and run 3DHistoGrading.exe
* Software can also be used by opening 3DHistoGrading.sln on MS Visual Studio

## Application usage
Currently available features:
* Visualize 3D image datasets (.png, .tiff, .bmp)
    * 3D rendering and three orthogonal 2D planes
    * Load Mask on top of visualized dataset (mask should be registered with the dataset)
    * Segmented masks are visualised on top of the sample using different colors, including grayscale dynamics
    * Mean, standard deviation and LBP images are visualized on grading window
* Surface artefact cropping tool for coronal and sagittal plane
    * User can fit a line on coronal and sagittal plane to remove artefacts/parts of the sample
    * Cropped sample can be saved as png
* Other pre-processing methods
    * Automatic sample alignment (based on gradient descent)
    * Automatic edge cropping around sample's center of mass
* Automatic segmentation of bone-cartilage -interface using U-Net deep neural network
* Automatic extraction of different volumes-of-interest (surface cartilage, deep cartilage and calcified tissue)
* Automatic grading from different osteochondral zones (based on MRELBP, PCA and linear regression)
* All sample processing steps, volumes-of interest and grading results can be saved

## Outputs
* Processed sample
* Automatically segmented calcified tissue mask
* Extracted analysis volumes
* OA grade from analysed sample volumes-of-interest (prediction of sample degeneration)

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
This software is distributed under the MIT License.

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
