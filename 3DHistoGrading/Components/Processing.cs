using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using Accord.Math;

using Kitware.VTK;
using OpenCvSharp;
using OpenCvSharp.Extensions;

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
        /// <param name="surfacecoordinates">Array of z-coordinates of sample surface. These can be used with size parameter to visualize the surface volume.</param>
        /// <param name="surfacevoi">Calculated surface volume.</param>
        public static void SurfaceExtraction(ref Rendering.renderPipeLine volume, int threshold, int[] size, 
            out int[,] surfacecoordinates, out byte[,,] surfacevoi)
        {
            // Get cropping dimensions
            int[] crop = volume.idata.GetExtent();
            crop[4] = (int)Math.Round(crop[5] / 2.0);
            //crop[5] = (int)Math.Round((double)crop[5] / 3);
            
            // Crop and flip the volume
            var cropped = volume.getVOI(crop);
            var flipper = vtkImageFlip.New();
            flipper.SetInput(cropped);
            flipper.SetFilteredAxes(2);
            flipper.Update();
            
            // Render cropped volume
            Rendering.RenderToNewWindow(flipper.GetOutput());

            // Convert vtkImageData to byte[,,]
            int[] dims = new int[] { crop[1] + 1, crop[3] + 1, (crop[5]-crop[4]) + 1 };
            byte[,,] byteVolume =
                DataTypes.VectorToVolume(
                DataTypes.vtkToByte(flipper.GetOutput()), dims);

            //// Crop to upper third of the sample
            //int[] crop = { 0, byteVolume.GetLength(0) - 1, 0, byteVolume.GetLength(1) - 1, 0, (int)Math.Floor((double)byteVolume.GetLength(2) / 3) };
            //byteVolume = Functions.Crop3D(byteVolume, crop);

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

            //double[] mean = new double[32] // Here, actually columns are written out
                //{ 72981.69444444, 69902.30555556, 0, 190.77777778, 2024.63888889, 6115.13888889, 9087.77777778, 7083.69444444, 2719.61111111,
                //    351.75, 0, 115310.61111111, 0, 241.11111111, 9692.44444444, 26931.41666667,  31327.11111111, 28872.97222222, 11496.33333333,
                //    282.83333333, 0, 34039.77777778, 6671.13888889, 13672.25, 8390.47222222, 7282.30555556, 7168.47222222, 7271.5,
                //    8368.16666667, 13541.88888889, 6752, 63765.80555556};

            for (int i = 0; i < w; i++)
            {
                // Select column
                double[] vector =
                    LBPLibrary.Functions.ArrayToVector(
                    LBPLibrary.Functions.GetSubMatrix(array, i, i, 0, l - 1));

                // Subtract mean
                if (mean == null)
                {
                    means[i] = vector.Average();
                    vector = Elementwise.Subtract(vector, means[i]);
                }
                else
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
    }
}
