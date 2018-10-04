﻿using Accord.Math;
using Kitware.VTK;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace HistoGrading.Components
{
    /// <summary>
    /// Includes methods that are used in sample processing.
    /// Features: surface extraction.
    /// </summary>
    public class Processing
    {
        /// <summary>
        /// Thresholds given data to array with containing only 0 and 255 values.
        /// </summary>
        /// <param name="array">Input array to be thresholded.</param>
        /// <param name="upperBound">Upper limit for thresholding.</param>
        /// <param name="lowerBound">Lower limit for thresholding.</param>
        /// <returns>Thresholded array.</returns>
        public static vtkImageData Threshold(vtkImageData array, int upperBound, int lowerBound)
        {
            // Assign new variables
            var source = new vtkImageMandelbrotSource();
            source.SetInput(array);
            source.Update();
            var threshold = new vtkImageThreshold();

            // Thresholding
            threshold.SetInputConnection(source.GetOutputPort());
            threshold.ThresholdBetween(lowerBound, upperBound);
            threshold.ReplaceInOn();
            threshold.SetInValue(255);
            threshold.Update();

            return threshold.GetOutput();
        }

        /// <summary>
        /// Calculates size[0] x size[0] x size[1] cubic volume of surface from the center of the sample.
        /// Currently performs the calculation from upper third of the sample to save memory.
        /// Creates a copy from the full data array, then crops the copy to 1/3 size.
        /// </summary>
        /// <param name="volume">Input sample data that contains vtkImageData.</param>
        /// <param name="threshold">Grayscale threshold for sample surface and centering.</param>
        /// <param name="size">Size of the calculated surface volume.</param>
        /// <param name="mode">Cartilage area selected fro VOI extraction. Possible ones: "surface", "deep", "calcified".</param>
        /// <param name="surfacecoordinates">Array of z-coordinates of sample surface. These can be used with size parameter to visualize the surface volume.</param>
        /// <param name="surfacevoi">Calculated surface volume.</param>
        public static void VOIExtraction(ref Rendering.renderPipeLine volume, int threshold, int[] size, string mode,
            out int[,] surfacecoordinates, out byte[,,] surfacevoi)
        {
            // Get cropping dimensions
            int[] crop = volume.idata.GetExtent();
            crop[4] = (int)Math.Round(crop[5] / 2.0);

            // Crop and flip the volume
            var cropped = volume.getVOI(crop);
            var flipper = vtkImageFlip.New();
            flipper.SetInput(cropped);
            flipper.SetFilteredAxes(2);
            flipper.Update();

            // Render cropped volume
            //Rendering.RenderToNewWindow(flipper.GetOutput());

            // Convert vtkImageData to byte[,,]
            int[] dims = new int[] { crop[1] + 1, crop[3] + 1, (crop[5] - crop[4]) + 1 };
            byte[,,] byteVolume =
                DataTypes.VectorToVolume(
                DataTypes.vtkToByte(flipper.GetOutput()), dims);

            // Get sample center coordinates
            int[] center = GetCenter(byteVolume, threshold);

            // Get surface
            GetSurface(byteVolume, center, size, threshold, out surfacecoordinates, out surfacevoi);

            // Free memory
            byteVolume = null;
            cropped = null;
            flipper = null;
        }

        /// <summary>
        /// Computes mean along each column of the array
        /// and subtracts it along the columns.
        /// If mean vector is given, it is subtracted from array.
        /// </summary>
        /// <param name="array">Array to be calculated.</param>
        /// <param name="mean">Mean vector to be subtracted.</param>
        /// <returns>Subtracted array.</returns>
        public static double[,] SubtractMean(double[,] array, double[] mean = null)
        {
            int w = array.GetLength(0), l = array.GetLength(1);
            double[,] dataAdjust = new double[0, 0];
            double[] means = new double[w];

            for (int i = 0; i < w; i++)
            {
                // Select column
                double[] vector =
                    LBPLibrary.Functions.ArrayToVector(
                    LBPLibrary.Functions.GetSubMatrix(array, i, i, 0, l - 1));

                // Subtract mean
                if (mean == null) // Calculate from features, if mean vector not given
                {
                    means[i] = vector.Average();
                    vector = Elementwise.Subtract(vector, means[i]);
                }
                else // Use mean vector if given
                {
                    vector = Elementwise.Subtract(vector, mean[i]);
                }

                // Concatenate
                dataAdjust = Matrix.Concatenate(dataAdjust, vector);
            }

            return dataAdjust;
        }

        /// <summary>
        /// Get center pixels of sample in XY.
        /// Calculates center of samples projection along z-axis.
        /// </summary>
        /// <param name="volume">Input volume.</param>
        /// <param name="threshold">Gray value threshold.</param>
        /// <returns>Center pixel coordinates.</returns>
        public static int[] GetCenter(byte[,,] volume, int threshold)
        {
            int[] center = new int[2];
            int[] dims = new int[] { volume.GetLength(0), volume.GetLength(1), volume.GetLength(2) };
            int[,] sumarray = new int[dims[0], dims[1]];
            int N = 0;

            // Calculate sum array
            Parallel.For(0, dims[0], y =>
            {
                Parallel.For(0, dims[1], x =>
                {
                    // Sum all values through z dimension
                    int sum = 0;
                    for (int z = 0; z < dims[2]; z++)
                    {
                        if (volume[y, x, z] > threshold)
                            sum ++;
                    }
                    sumarray[y, x] = sum;
                });
            });

            // Calculate center of samples projection on z axis
            for (int i = 0; i < dims[0]; i++)
            {
                for (int j = 0; j < dims[1]; j++)
                {
                    if (sumarray[i, j] > 0)
                    {
                        //center[0] += i * sumarray[i, j]; // center of mass
                        //center[1] += j * sumarray[i, j];

                        center[0] += i; // center of projection
                        center[1] += j;
                        N++;
                    }
                }
            }
            //center[0] = (int)Math.Round((double)center[0] / sumarray.Sum()); // CoM
            //center[1] = (int)Math.Round((double)center[1] / sumarray.Sum());
            center[0] = (int)Math.Round((double)center[0] / N);
            center[1] = (int)Math.Round((double)center[1] / N);

            return center;
        }

        /// <summary>
        /// Get surface volume and coordinates.
        /// </summary>
        /// <param name="volume">Input volume.</param>
        /// <param name="center">Center pixel of VOI in XY.</param>
        /// <param name="size">VOI size: {XY, Z}.</param>
        /// <param name="threshold">Surface threshold gray value.</param>
        /// <param name="surfaceCoordinates">Surface z-coordinate array.</param>
        /// <param name="surfaceVOI">Surface volume array.</param>
        public static void GetSurface(byte[,,] volume, int[] center, int[] size, int threshold, 
            out int[,] surfaceCoordinates, out byte[,,] surfaceVOI)
        {
            int[,] coordinates = new int[size[0], size[0]];
            byte[,,] VOI = new byte[size[0], size[0], size[1]];
            int[] dims = new int[] { volume.GetLength(0), volume.GetLength(1), volume.GetLength(2) };

            // Calculate starting coordinates
            int[] start = new int[] { center[0] - (int)Math.Floor((double)size[0] / 2), center[1] - (int)Math.Floor((double)size[0] / 2) };

            // Find surface coordinate for each pixel
            Parallel.For(start[0], start[0] + size[0], y =>
            {
                Parallel.For(start[1], start[1] + size[0], x =>
                {
                    for (int z = 0; z < dims[2]; z++)
                    //for (int z = dims[2]; z <= 0; z--)
                    {
                        if (volume[y, x, z] > threshold)
                        {
                            // Update surface coordinate
                            coordinates[y - start[0], x - start[1]] = z;

                            // Update surface VOI
                            int zlim = z + size[1];
                            //int zlim = z - size[1];
                            if (zlim > dims[2]) // avoid exceeding array
                            {
                                zlim = dims[2];
                            }
                            for (int zz = z; zz < zlim; zz++)
                            //for (int zz = z; zz < zlim; zz--)
                            {
                                VOI[y - start[0], x - start[1], zz - z] = volume[y, x, zz];
                            }
                            break; // Move to next pixel, once surface is found
                        }
                    }
                });
            });
            surfaceCoordinates = coordinates;
            surfaceVOI = VOI;
        }

        /// <summary>
        /// Calculates mean and standard deviation images from volume-of-interest
        /// along third axis (z).
        /// </summary>
        /// <param name="surfaceVOI">Input volume.</param>
        /// <param name="meanImage">Mean 2D image.</param>
        /// <param name="stdImage">Standard deviation 2D image.</param>
        public static void MeanAndStd(byte[,,] surfaceVOI, out double[,] meanImage, out double[,] stdImage)
        {
            int[] dims = new int[] { surfaceVOI.GetLength(0), surfaceVOI.GetLength(1), surfaceVOI.GetLength(2) };
            double[,] mean = new double[dims[0], dims[1]];
            double[,] std = new double[dims[0], dims[1]];

            Parallel.For(0, dims[0], i =>
            {
                Parallel.For(0, dims[1], j =>
                {
                    double[] temp = new double[dims[2]]; // has to be initialized in the loop
                    for (int k = 0; k < dims[2]; k++)
                    {
                        temp[k] = surfaceVOI[i, j, k];
                    }
                    mean[i, j] = temp.Average();
                    std[i, j] =
                        Math.Sqrt(temp
                        .Subtract(temp.Average())
                        .Pow(2)
                        .Sum()
                        / (temp.Length - 1));
                });
            });
            meanImage = mean;
            stdImage = std;
        }

        /// <summary>
        /// Rotates volume along given axis.
        /// </summary>
        /// <param name="volume"></param>
        /// <param name="angles"></param>
        public static void RotateData(ref vtkImageData volume, double[] angles)
        {
            // Rotation along image center
            double[] center = volume.GetExtent().Divide(2);
            // Dimensions of rotated image
            int[] outExtent = volume.GetExtent().Multiply(1.1).Round().ToInt32();

            // Rotation parameters
            var rotate = new vtkTransform();
            rotate.Translate(center[1], center[3], center[5]);
            rotate.RotateX(angles[0]);
            rotate.RotateY(angles[1]);
            rotate.RotateZ(angles[2]); // z angle should be 0
            rotate.Translate(-center[1], -center[3], -center[5]);

            // Perform rotation
            var slice = new vtkImageReslice();
            slice.SetInput(volume);
            slice.SetResliceTransform(rotate);
            slice.SetInterpolationModeToCubic();
            slice.SetOutputSpacing(volume.GetSpacing()[0], volume.GetSpacing()[1], volume.GetSpacing()[2]);
            slice.SetOutputOrigin(volume.GetOrigin()[0], volume.GetOrigin()[1], volume.GetOrigin()[2]);
            slice.SetOutputExtent(outExtent[0], outExtent[1], outExtent[2], outExtent[3], outExtent[4], outExtent[5]);
        }

        public static int[] find_center(vtkImageData stack, double threshold = 80.0)
        {
            //Get byte data
            byte[] bytedata = DataTypes.vtkToByte(stack);

            //Get data dimensions
            int[] dims = stack.GetExtent();
            int h = dims[1] - dims[0] + 1;
            int w = dims[3] - dims[2] + 1;
            int d = dims[5] - dims[4] + 1;
            //Get strides
            int stride_d = h * w;
            int stride_h = 1;
            int stride_w = h;

            //Empty array for binary mask
            byte[,] BW = new byte[h, w];

            Parallel.For(0, h, (int y) =>
            {
                Parallel.For(0, w, (int x) =>
                {
                    for (int z = 0; z < d; z++)
                    {
                        int pos = (z * stride_d) + (x * stride_w) + (y * stride_h);
                        byte val = bytedata[pos];
                        if (val > threshold) { BW[y, x] = 255; }
                    }

                });
            });

            Mat sumim = new Mat(h, w, MatType.CV_8UC1, BW);

            /*
            using (var window = new Window("window", image: sumim, flags: WindowMode.AutoSize))
            {
                Cv2.WaitKey();
            }
            */

            int x1; int x2; int y1; int y2;
            Functions.get_bbox(out x1, out x2, out y1, out y2, sumim);

            //Compute center
            int[] center = new int[2];
            center[0] = (y2 + y1) / 2;
            center[1] = (x2 + x1) / 2;

            return center;
        }

        public static vtkImageData center_crop(vtkImageData stack, int side = 400)
        {
            //Get input dimensions
            int[] dims = stack.GetExtent();

            //Find the center of the sample

            int[] center = find_center(stack);//GetCenter(bytedata,80);
            //Compute new volume sides
            int y2 = Math.Min(center[0] + (side / 2), dims[1]);
            int y1 = Math.Max(y2 - side + 1, dims[0]);
            int x2 = Math.Min(center[1] + (side / 2), dims[3]);
            int x1 = Math.Max(x2 - side + 1, dims[2]);

            //Create VOI extractor
            vtkExtractVOI cropper = vtkExtractVOI.New();
            cropper.SetVOI(y1, y2, x1, x2, dims[4], dims[5]);
            cropper.SetInput(stack);
            cropper.Update();

            vtkImageData output = cropper.GetOutput();
            output.SetExtent(0, (y2 - y1), 0, (x2 - x1), dims[4], dims[5]);
            output.Update();

            return output;
        }

        /*
        public static double[,] average_tiles(vtkImageData input, int n_tiles = 16)
        {
            //Get dimensions
            int[] dims = input.GetExtent();
            int h = dims[1] - dims[0] + 1;
            int w = dims[3] - dims[2] + 1;
            int d = dims[5] - dims[4] + 1;

            //Input to byte array
            byte[] bytedata = DataTypes.vtkToByte(input);

            //Generate tile coordinates
            int N = (int)Math.Sqrt(n_tiles);
            int wh = h / N;
            int ww = w / N;

            for(int kh = 0; kh <N; kh++)
            {
                for (int kh = 0; kh < N; kh++)
                {

                }
            }

        }
        */
    }
}
