using Accord.Math;
using Kitware.VTK;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

using Accord;
using Accord.Statistics.Analysis;

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
            int[] dims = new int[] { (crop[1] - crop[0]) + 1, (crop[3] - crop[2]) + 1, (crop[5] - crop[4]) + 1 };

            byte[,,] byteVolume =
                DataTypes.VectorToVolume(
                DataTypes.vtkToByte(flipper.GetOutput()), dims);

            // Get sample center coordinates
            //int[] center = GetCenter(byteVolume, threshold);
            int[] center = new int[] { dims[0] / 2, dims[1] / 2 };

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

        //Remove artefacts
        public static vtkImageData remove_artefacts(vtkImageData volume, double[] points, int axis)
        {
            //Get input volume dimensions            
            int[] dims = volume.GetExtent();
            int h = dims[1] - dims[0] + 1;
            int w = dims[3] - dims[2] + 1;
            int d = dims[5] - dims[4] + 1;
            Console.WriteLine("Got dims: {0},{1},{2}",h,w,d);
            //Convert volume to byte array
            byte[] bytedata = DataTypes.vtkToByte(volume);
            Console.WriteLine("Got bytedata");
            //Compute slope (k) and zero crossing (b) from given point for a line y=kx+b
            //Compute the slope of the line from points
            double slope = (points[3] - points[1]) / (points[2] - points[0]);
            //Compute zero crossing
            double b = points[3] - slope * points[2] - dims[4];
            Console.WriteLine("Got line equation");
            //Iterate over the data
            Parallel.For(0, h, (int y) =>
            {
                Parallel.For(0, w, (int x) =>
                {
                    //Compute extent for zeroing
                    int zstart = 0;
                    int zstop = d;
                    if(axis == 0)
                    {
                        zstart = (int)((double)(x+dims[2]) * slope + b);
                        zstart = Math.Max(zstart, 0);
                    }
                    if (axis == 1)
                    {
                        zstart = (int)((double)(y+dims[0]) * slope + b);
                        zstart = Math.Max(zstart, 0);
                    }
                    //Iterate over z-axis
                    for(int z = zstart; z<zstop; z++)
                    {
                        int pos = z * (h * w) + x * h + y;
                        bytedata[pos] = 0;
                    }
                });
            });
            Console.WriteLine("Zeroed");
            //Convert byte data back to vtkdata
            vtkImageData output = DataTypes.byteToVTK1D(bytedata,dims);

            return output;
        }

        /// <summary>
        /// Rotates 3D vtk volume around x and y axes.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="angles"></param>
        /// <param name="mode"></param>
        /// <param name="invert"></param>
        /// <returns></returns>
        public static vtkImageData rotate_sample(vtkImageData input, double angle, int axis, int out_extent = 0)
        {            
            //get input data dimensions
            int[] dims = input.GetExtent();
            //Compute centers
            int[] centers = new int[] { (dims[1] + dims[0]) / 2, (dims[3] + dims[2]) / 2, (dims[5] + dims[4]) / 2 };

            //Set rotation axis
            int[] axes = new int[3];
            axes[axis] = 1;

            int[] new_dims = new int[] { dims[0], dims[1], dims[2], dims[3], dims[4], dims[5] };            
            int[] new_centers = new int[] { centers[0], centers[1], centers[2] };

            //Compute new sample dimensions
            if (axis == 0)
            {
                new_dims[3] = (int)(Math.Cos(Math.Abs(angle / 180) * Math.PI) * new_dims[3] + Math.Sin(Math.Abs(angle / 180) * Math.PI) * new_dims[5]);
                new_dims[5] = (int)(Math.Sin(Math.Abs(angle / 180) * Math.PI) * new_dims[3] + Math.Cos(Math.Abs(angle / 180) * Math.PI) * new_dims[5]);

                new_centers[1] = (Math.Abs(new_dims[3]) + Math.Abs(new_dims[2])) / 2;
                new_centers[2] = (Math.Abs(new_dims[5]) + Math.Abs(new_dims[4])) / 2;                
            }
            if(axis==1)
            {
                new_dims[1] = (int)(Math.Cos(Math.Abs(angle / 180) * Math.PI) * new_dims[1] + Math.Sin(Math.Abs(angle / 180) * Math.PI) * new_dims[5]);
                new_dims[5] = (int)(Math.Sin(Math.Abs(angle / 180) * Math.PI) * new_dims[1] + Math.Cos(Math.Abs(angle / 180) * Math.PI) * new_dims[5]);                

                new_centers[0] = (Math.Abs(new_dims[0]) + Math.Abs(new_dims[1])) / 2;
                new_centers[2] = (Math.Abs(new_dims[5]) + Math.Abs(new_dims[4])) / 2;                
            }

            
            //Image transformation
            vtkTransform transform = vtkTransform.New();
            transform.Translate(centers[0], centers[1], centers[2]);
            transform.RotateWXYZ(angle, axes[0], axes[1], axes[2]);
            if (out_extent == 0)
            {
                transform.Translate(-centers[0], -centers[1], -centers[2]);
            }
            else
            {
                transform.Translate(-new_centers[0], -new_centers[1], -new_centers[2]);
            }

            //Console.ReadKey();

            transform.Update();

            //Compute new data extent
            int[] diff = new int[] { new_dims[1] - dims[1], new_dims[3] - dims[3], new_dims[5] - dims[5] };
            new_dims[0] += diff[0] / 2; new_dims[1] -= diff[0] / 2;
            new_dims[2] += diff[1] / 2; new_dims[3] -= diff[1] / 2;
            new_dims[4] += diff[2] / 2; new_dims[5] -= diff[2] / 2;

            

            //Image reslicing
            vtkImageReslice rotater = vtkImageReslice.New();
            rotater.SetInput(input);
            rotater.SetInformationInput(input);
            rotater.SetResliceTransform(transform);
            rotater.SetInterpolationModeToCubic();
            //rotater.SetInterpolationModeToLinear();
            if (out_extent == 1)
            {
                rotater.SetOutputSpacing(input.GetSpacing()[0], input.GetSpacing()[1], input.GetSpacing()[2]);
                rotater.SetOutputOrigin(input.GetOrigin()[0], input.GetOrigin()[1], input.GetOrigin()[2]);
                rotater.SetOutputExtent(new_dims[0], new_dims[1], new_dims[2], new_dims[3], new_dims[4], new_dims[5]);
            }            
            rotater.Update();

            vtkImageData output = vtkImageData.New();
            output.DeepCopy(rotater.GetOutput());

            rotater.Dispose();
            transform.Dispose();

            return output;
        }

        public static vtkImageData rescale_sample(vtkImageData input, double scale)
        {
            //Get sample dimensions
            int[] dims = input.GetExtent();

            vtkImageResample samplery = vtkImageResample.New();
            samplery.SetInput(input);
            samplery.SetOutputSpacing(input.GetSpacing()[0], input.GetSpacing()[1], input.GetSpacing()[2]);
            samplery.SetOutputOrigin(input.GetOrigin()[0], input.GetOrigin()[1], input.GetOrigin()[2]);
            samplery.SetOutputExtent((int)(scale * dims[0]), (int)(scale * dims[1]), dims[2], dims[3], dims[4], dims[5]);
            samplery.SetInterpolationModeToCubic();
            samplery.SetAxisMagnificationFactor(0, scale);
            samplery.Update();

            vtkImageResample samplerx = vtkImageResample.New();
            samplerx.SetInputConnection(samplery.GetOutputPort());
            samplerx.SetOutputSpacing(samplery.GetOutputSpacing()[0], samplery.GetOutputSpacing()[1], samplery.GetOutputSpacing()[2]);
            samplerx.SetOutputOrigin(samplery.GetOutputOrigin()[0], samplery.GetOutputOrigin()[1], samplery.GetOutputOrigin()[2]);
            samplerx.SetOutputExtent((int)(scale * dims[0]), (int)(scale * dims[1]), (int)(scale * dims[2]), (int)(scale * dims[3]), dims[4], dims[5]);
            samplerx.SetInterpolationModeToCubic();
            samplerx.SetAxisMagnificationFactor(1, scale);
            samplerx.Update();

            vtkImageResample samplerz = vtkImageResample.New();
            samplerz.SetInputConnection(samplerx.GetOutputPort());
            samplerz.SetOutputSpacing(samplerx.GetOutputSpacing()[0], samplerx.GetOutputSpacing()[1], samplerx.GetOutputSpacing()[2]);
            samplerz.SetOutputOrigin(samplerx.GetOutputOrigin()[0], samplerx.GetOutputOrigin()[1], samplerx.GetOutputOrigin()[2]);
            samplerz.SetOutputExtent((int)(scale * dims[0]), (int)(scale * dims[1]), (int)(scale * dims[2]),
                (int)(scale * dims[3]), (int)(scale * dims[4]), (int)(scale *dims[5]));
            samplerz.SetInterpolationModeToCubic();
            samplerz.SetAxisMagnificationFactor(2, scale);
            samplerz.Update();

            vtkImageData output = vtkImageData.New();
            output.DeepCopy(samplerz.GetOutput());

            samplerz.Dispose();
            samplerx.Dispose();
            samplery.Dispose();

            return output;
        }

        public static int[] find_center(vtkImageData stack, double threshold = 70.0, int[] zrange = null)
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

            //z range
            int zstart = 0; int zstop = 0;
            if(zrange == null)
            {
                zstart = 0;
                zstop = d;                
            }
            else
            {
                zstart = zrange[0];
                zstop = zrange[1];
            }

            Parallel.For(0, h, (int y) =>
            {
                Parallel.For(0, w, (int x) =>
                {
                    for (int z = zstart; z < zstop; z++)
                    {
                        int pos = (z * stride_d) + (x * stride_w) + (y * stride_h);
                        byte val = bytedata[pos];
                        if (val > threshold) { BW[y, x] = 255; }
                    }

                });
            });

            Mat sumim = new Mat(h, w, MatType.CV_8UC1, BW);

            //Get largest binary object
            sumim = Functions.largest_connected_component(sumim);

            int x1; int x2; int y1; int y2;
            Functions.get_bbox(out x1, out x2, out y1, out y2, sumim);

            //Compute center
            int[] center = new int[2];
            center[0] = (y2 + y1) / 2 + dims[0];
            center[1] = (x2 + x1) / 2 + dims[2];

            return center;
        }

        public static vtkImageData center_crop(vtkImageData stack, int side = 400)
        {
            //Get input dimensions
            int[] dims = stack.GetExtent();

            //Find the center of the sample

            int[] center = find_center(stack,70, new int[] { 0, (dims[5] - dims[4] + 1)/3 });//GetCenter(bytedata,80);
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
                        
            return cropper.GetOutput();
        }

        public static void average_tiles(out double[,,] averages, out int[] steps, vtkImageData input, int n_tiles = 16)
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

            steps = new int[] { wh, ww};

            List<int[]> tiles = new List<int[]>();            

            for (int kh = 0; kh <N; kh++)
            {
                for (int kw = 0; kw < N; kw++)
                {
                    int[] tmp = new int[] { kh * wh, (kh + 1) * wh, kw * ww, (kw + 1) * ww };
                    tiles.Add(tmp);
                }
            }

            //Iterate over tiles, and average the grayscale values

            //Empty array for averages
            double[,,] _averages = new double[N,N,d];
            //Number of elements
            N = wh * ww;
            foreach(int[] tile in tiles)
            {
                for (int z = 0; z < d; z++)
                {
                    Parallel.For(tile[0], tile[1], (int y) =>
                      {
                          Parallel.For(tile[2], tile[3], (int x) =>
                          {                              
                              int pos = z * (h * w) + x * (h) + y;
                              byte val = bytedata[pos];
                              _averages[y / wh, x / ww, z] += (double)val;
                          });
                      });

                    _averages[(tile[1]-1) / wh, (tile[3]-1) / ww, z] /= (double)N;
                }
            }

            averages = _averages;
        }

        public static vtkImageData otsu3D(vtkImageData input,int axis = 0, double threshold = 60.0)
        {
            //Get dimensions
            int[] dims = input.GetExtent();
            int h = dims[1] - dims[0] + 1;
            int w = dims[3] - dims[2] + 1;
            int d = dims[5] - dims[4] + 1;

            //Convert to byte array
            byte[] bytedata = DataTypes.vtkToByte(input);
            byte[] output = new byte[bytedata.Length];

            //Iterate over axis
            if (axis == 0)
            {
                for (int x = 0; x < w; x++)
                {
                    byte[,] plane = new byte[d, h];
                    Parallel.For(0, h, (int y) =>
                    {
                        Parallel.For(0, d, (int z) =>
                        {
                            int pos = z * (h * w) + y * w + x;
                            plane[z, y] = bytedata[pos];
                        });
                    });

                    //Convert 2D byte array to opencv Mat
                    Mat image = new Mat(d, h, MatType.CV_8UC1, plane);
                    Mat BW = image.Threshold(threshold, 255.0, ThresholdTypes.Otsu);

                    IntPtr pointer = BW.Data;
                    byte[] tmp = new byte[h * d];
                    Marshal.Copy(pointer, tmp, 0, h * d);
                    Parallel.For(0, h, (int y) =>
                    {
                        Parallel.For(0, d, (int z) =>
                        {
                            int tmppos = z * h + y;
                            int pos = z * (h * w) + y * w + x;
                            output[pos] = tmp[tmppos];
                        });
                    });
                }
            }
            else
            {
                for (int y = 0; y < h; y++)
                {
                    byte[,] plane = new byte[d, w];
                    Parallel.For(0, w, (int x) =>
                    {
                        Parallel.For(0, d, (int z) =>
                        {
                            int pos = z * (h * w) + y * w + x;
                            plane[z, x] = bytedata[pos];
                        });
                    });

                    //Convert 2D byte array to opencv Mat
                    Mat image = new Mat(d, w, MatType.CV_8UC1, plane);
                    Mat BW = image.Threshold(threshold, 255.0, ThresholdTypes.Otsu);

                    IntPtr pointer = BW.Data;
                    byte[] tmp = new byte[w * d];
                    Marshal.Copy(pointer, tmp, 0, w * d);
                    Parallel.For(0, w, (int x) =>
                    {
                        Parallel.For(0, d, (int z) =>
                        {
                            int tmppos = z * w + x;
                            int pos = z * (h * w) + y * w + x;
                            output[pos] = tmp[tmppos];
                        });
                    });
                }
            }

            return DataTypes.byteToVTK1D(output,dims);
        }

        /// <summary>
        /// Calculates mean and standard deviation images from volume-of-interest
        /// along third axis (z).
        /// </summary>
        /// <param name="input">Input volume as vtkImageData.</param>
        /// <param name="meanImage">Mean 2D image.</param>
        /// <param name="stdImage">Standard deviation 2D image.</param>
        public static void MeanAndStd(vtkImageData input, out double[,] meanImage, out double[,] stdImage)
        {
            //Get data extent
            int[] ext = input.GetExtent();
            int[] dims = new int[] { ext[3] - ext[2] + 1, ext[1] - ext[0] + 1, ext[5] - ext[4] + 1 };
            Console.WriteLine("Input shape: {0}, {1}, {2}".Format(dims[0], dims[1], dims[2]));
            // Convert to byte volume
            byte[] bytedata = DataTypes.vtkToByte(input);
            byte[,,] bytevolume = DataTypes.VectorToVolume(bytedata, dims);

            //int[] dims = new int[] { bytedata.GetLength(0), bytedata.GetLength(1), bytedata.GetLength(2) };
            double[,] mean = new double[dims[0], dims[1]];
            double[,] std = new double[dims[0], dims[1]];

            Parallel.For(0, dims[0], i =>
            {
                Parallel.For(0, dims[1], j =>
                {
                    //double[] temp = new double[dims[2]]; // has to be initialized in the loop
                    double[] temp = new double[0]; // has to be initialized in the loop
                    for (int k = 0; k < dims[2]; k++)
                    {
                        //temp[k] = bytevolume[i, j, k];
                        if (bytevolume[i, j, k] > 0)
                            temp.Concatenate(bytevolume[i, j, k]);
                        
                    }
                    if (temp.Length > 0)
                    {
                        mean[i, j] = temp.Average();
                        std[i, j] =
                            Math.Sqrt(temp
                            .Subtract(temp.Average())
                            .Pow(2)
                            .Sum()
                            / (temp.Length - 1));
                    }
                    else
                    {
                        mean[i, j] = 0;
                        std[i, j] = 0;
                    }
                    
                });
            });
            meanImage = mean;
            stdImage = std;
        }

        public static void get_voi_mu_std(out vtkImageData output, out double[,] mu, out double[,] std, vtkImageData input, int depth, double threshold = 70.0)
        {
            //Get data extent
            int[] dims = input.GetExtent();
            int h = dims[1] - dims[0] + 1;
            int w = dims[3] - dims[2] + 1;
            int d = dims[5] - dims[4] + 1;

            //Copute strides
            int stridew = 1;
            int strideh = w;
            int strided = h * w;

            byte[] bytedata = DataTypes.vtkToByte(input);

            /*
            //Create top and bottom images
            byte[] bottom = new byte[h*w];
            byte[] top = new byte[h*w];

            for(int k = 0; k < h*w; k++)
            {
                int pos1 = k;
                int pos2 = (d-1) * (h * w) + k;

                bottom[k] = bytedata[pos1];
                top[k] = bytedata[pos2];
            }

            Mat im1 = new Mat(h, w, MatType.CV_8UC1, bottom);
            Mat im2 = new Mat(h, w, MatType.CV_8UC1, top);

            using (var window = new Window("bottom", image: im1, flags: WindowMode.AutoSize))
            {
                Cv2.WaitKey();
            }

            using (var window = new Window("top", image: im2, flags: WindowMode.AutoSize))
            {
                Cv2.WaitKey();
            }
            */

            byte[] voidata = new byte[bytedata.Length];

            double[,] _mu = new double[h,w];
            double[,] _std = new double[h, w];

            //Get voi indices
            Parallel.For(24, h-24, (int y) =>
            {
                Parallel.For(24, w-24, (int x) =>
                {
                    int start = d-1;
                    int stop = 0;                    
                    //Compute mean
                    for (int z = d-1; z > 0; z-=1)
                    {
                        int pos = z * strided + y * strideh + x * stridew;
                        double val = (double)bytedata[pos];
                        if(val>=threshold)
                        {
                            start = z;
                            stop = Math.Max(z - depth,0);
                            //Compute mean and std
                            for (int zz = start; zz > stop; zz -= 1)
                            {
                                int newpos = zz * strided + y * strideh + x * stridew;
                                double newval = (double)bytedata[newpos];
                                voidata[newpos] = (byte)newval;
                                _mu[y, x] = newval / (double)depth;
                            }

                            for (int zz = start; zz > stop; zz -= 1)
                            {
                                int newpos = zz * strided + y * strideh + x * stridew;
                                double newval = (double)bytedata[newpos];
                                _std[y, x] += ((double)newval - _mu[y, x]) * ((double)newval - _mu[y, x]);
                            }

                            _std[y, x] = Math.Pow(_std[y, x] / (depth - 1), 0.5);
                            break;
                        }
                    }


                });
            });

            mu = _mu;
            std = _std;
            output = vtkImageData.New();
            //Copy voi data to input array
            vtkUnsignedCharArray charArray = vtkUnsignedCharArray.New();
            //Pin byte array
            GCHandle pinnedArray = GCHandle.Alloc(voidata, GCHandleType.Pinned);
            //Set character array input
            charArray.SetArray(pinnedArray.AddrOfPinnedObject(), h * w * d, 1);
            //Set vtkdata properties and connect array
            //Data from char array
            output.GetPointData().SetScalars(charArray);
            //Number of scalars/pixel
            output.SetNumberOfScalarComponents(1);
            //Data extent, 1st and last axis are swapped from the char array
            //Data is converted back to original orientation
            output.SetExtent(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
            //Scalar type
            output.SetScalarTypeToUnsignedChar();
            output.Update();            
        }

        static double find_ori_pca(double[][] data, int axis = 1, int use_radians = 1)
        {
            //Get first principal component            
            var pca = new PrincipalComponentAnalysis();// data, AnalysisMethod.Standardize);
            pca.Learn(data);
            var components = pca.Components;
            var x = components.First().Eigenvector;

            //Normalize and find orientation between y-axis and 1st pc
            var _x = x.Normalize();
            var _ypos = new double[2];
            var _yneg = new double[2];
            _ypos[axis] = 1; _yneg[axis] = -1;

            double theta1 = Math.Acos(_ypos.Dot(_x));
            double theta2 = Math.Acos(_yneg.Dot(_x));
            double theta = 0.0;


            if (theta1<theta2){ theta = theta1; }
            else { theta = -theta2; }

            if (use_radians == 1)
            {
                theta *= 180 / Math.PI;
            }           

            return theta;
        }

        public static double find_slice_ori(vtkImageData slice, double threshold = 70.0)
        {
            //Get dimensions
            int[] dims = slice.GetExtent();
            int h = dims[1] - dims[0] + 1;
            int w = dims[3] - dims[2] + 1;

            //Convert slice to mat
            byte[] bytedata = DataTypes.vtkToByte(slice);

            
            List<int[]> _idx = new List<int[]>();

            byte[] cont = new byte[h * w];

            //for (int k = 0; k < cont.Length; k++)
            for (int k = 0; k < bytedata.Length; k++)
            {
                int _x = k / h;
                int _y = k % h;                
                if ((double)bytedata[k] > threshold)
                {
                    _idx.Add(new int[] { _y, _x });
                    cont[k] = 255;
                }                
            }

            Mat im = new Mat(w, h, MatType.CV_8UC1, cont);
            using (var window = new Window("Indices", image: im, flags: WindowMode.AutoSize))
            {
                Cv2.WaitKey();
            }

            //Convert list to double array
            double[][] idx = new double[_idx.Count()][];            

            //Iterate over the list and collect indices to the array
            for (int k = 0; k < _idx.Count(); k++)
            {
                int[] _tmp = _idx.ElementAt(k);
                idx[k] = new double[] { _tmp[0], _tmp[1] };                
            }

            double theta = find_ori_pca(idx,0);

            return theta;
        }
        
        public static double circle_loss(vtkImageData input)
        {
            //Get dimensions
            int[] dims = input.GetExtent();
            int h = dims[1] - dims[0] + 1;
            int w = dims[3] - dims[2] + 1;
            int d = dims[5] - dims[4] + 1;

            //Get non zero indices form projection image
            byte[] bytedata = DataTypes.vtkToByte(input);
            byte[] surf = new byte[h*w];
            Parallel.For(0, h, (int ky) =>
            {
                Parallel.For(0, w, (int kx) =>
                {
                    for(int kz = d-1; kz > 0; kz -=1 )
                    {
                        int pos = kz*(h * w) + kx * h + ky;
                        int val = bytedata[pos];
                        if (val > 70.0)
                        {
                            surf[ky*w + kx] = 255;
                            break;
                        }
                    }
                });
            });

            //Get nonzero indices
            Mat surfcv = new Mat(h, w, MatType.CV_8UC1, surf);
            //Find largest blob
            surfcv = Functions.largest_connected_component(surfcv);

            //Indices
            Mat nonzero = surfcv.FindNonZero();

            //Fit circle
            Point2f center; float R;
            nonzero.MinEnclosingCircle(out center, out R);
            //Generate a circle
            byte[] circle = new byte[h*w];
            Parallel.For(0, h, (int ky) =>
            {
                Parallel.For(0, w, (int kx) =>
                {
                    float val = (ky - center.Y) * (ky - center.Y) + (kx - center.X) * (kx - center.X);
                    if(val < R*R){ circle[ky*w + kx] = 255; }
                });
            });

            //Get dice score
            double dice = Functions.dice_score_2d(surf, circle);

            surfcv.Dispose();
            nonzero.Dispose();
            bytedata = null;

            return 1 - dice;

        }
        
        public static double[] grad_descent(vtkImageData data, double alpha = 0.5, double h = 5.0, int n_iter = 60)
        {
            //Starting orientation
            double[] ori = new double[2];
            int[] dims = data.GetExtent();
            int y = dims[1] - dims[0] + 1;
            int x = dims[3] - dims[2] + 1;
            int z = dims[5] - dims[4] + 1;

            for(int k = 1; k < n_iter+1; k++)
            {
                //Iintialize gradient
                double[] grads = new double[2];

                //Rotate the sample
                vtkImageData rotated1 = rotate_sample(data, ori[0] + h, 0, 1);
                if (ori[1] != 0) { rotated1 = rotate_sample(rotated1, ori[1], 1, 1); }

                vtkImageData rotated2 = rotate_sample(data, ori[0] - h, 0, 1);
                if (ori[1] != 0) { rotated2 = rotate_sample(rotated2, ori[1], 1, 1); }

                //Get losses
                double d1 = circle_loss(rotated1);
                double d2 = circle_loss(rotated2);

                //Compute gradient
                grads[0] = (d1 - d2) / (2 * h);

                //Rotate the sample
                vtkImageData rotated3 = rotate_sample(data, ori[1] + h, 1, 1);
                if (ori[0] != 0) { rotated3 = rotate_sample(rotated3, ori[0], 0, 1); }

                vtkImageData rotated4 = rotate_sample(data, ori[1] - h, 1, 1);
                if (ori[0] != 0) { rotated4 = rotate_sample(rotated4, ori[0], 0, 1); }

                //Get losses
                double d3 = circle_loss(rotated3);
                double d4 = circle_loss(rotated4);

                //Compute gradient
                grads[1] = (d3 - d4) / (2 * h);

                //Update the orientation
                ori[0] -= Math.Sign(grads[0]) * alpha;
                ori[1] -= Math.Sign(grads[1]) * alpha;

                if(k % n_iter/2 == 0){alpha /= 2;}

                rotated1.Dispose();
                rotated2.Dispose();
                rotated3.Dispose();
                rotated4.Dispose();
            }

            return ori;
        }

        public static void get_mean_sd(out double[,] mean, out double[,] sd, vtkImageData VOI, int voi_depth = 0, int crop_size = 24)
        {
            //Get input extent
            int[] dims = VOI.GetExtent();

            //Compute dimensions
            int h = dims[1] - dims[0] + 1;
            int w = dims[3] - dims[2] + 1;
            int d = dims[5] - dims[4] + 1;

            //Get byte data from vtkImageData
            byte[] bytedata = DataTypes.vtkToByte(VOI);

            double[,] mu = new double[h - crop_size * 2, w - crop_size * 2];
            byte[,] muim = new byte[h - crop_size * 2, w - crop_size * 2];
            int[,] Ns = new int[h - crop_size * 2, w - crop_size * 2];

            //Set void depth
            if(voi_depth == 0) { voi_depth = d; }

            //Iterate over data and compute mean image
            for (int y = crop_size; y < h - crop_size; y++)
            {
                for (int x = crop_size; x < w - crop_size; x++)
                {
                    double sum = 0.0;
                    int N = 0;
                    for(int z = d-1; z >= 0; z-=1)
                    {
                        //Compute position
                        int pos = z * (h * w) + x * h + y;
                        //Get byte value
                        byte val = bytedata[pos];
                        //Add to sum
                        sum += (double)val;
                        //If value is nonzero, increase count
                        if(val > 0) { N += 1; }
                        //If count is equal to VOI depth, break
                        if (N == voi_depth) { break; }                        
                    }
                    mu[y- crop_size, x - crop_size] = sum/(double)N;
                    muim[y- crop_size, x - crop_size] = (byte)mu[y - crop_size, x - crop_size];
                    Ns[y- crop_size, x - crop_size] = N;
                }
            }

            double[,] sigma = new double[h - crop_size * 2, w - crop_size * 2];
            byte[,] sdim = new byte[h - crop_size * 2, w - crop_size * 2];

            //Iterate over data and compute sd image
            for(int y = crop_size; y< h - crop_size; y++)
            {
                for (int x = crop_size; x < w - crop_size; x++)
                {
                    double sum = 0.0;
                    int N = 0;
                    for (int z = d - 1; z >= 0; z -= 1)
                    {
                        //Compute position
                        int pos = z * (h * w) + x * h + y;
                        //Get byte value
                        byte val = bytedata[pos];
                        //If value is non-zero, subtract from value and square
                        if(val > 0)
                        {
                            double tmp = (double)val - mu[y - crop_size, x - crop_size];
                            sum += tmp * tmp;
                            N += 1;
                        }
                        //If count is equal to VOI depth, break
                        if (N == voi_depth) { break; }
                    }
                    sigma[y - crop_size, x - crop_size] = Math.Sqrt(sum / ((double)Ns[y - crop_size, x - crop_size] - 1.0));
                    if (N == 0) { sigma[y - crop_size, x - crop_size] = 0.0; }
                    sdim[y - crop_size, x - crop_size] = (byte)sigma[y - crop_size, x - crop_size];
                }
            }

            //Return mu and sd
            mean = mu;
            sd = sigma;

            //Mat meanmat = new Mat(h - crop_size * 2, w - crop_size * 2, MatType.CV_8UC1, muim);
            //Mat sdmat = new Mat(h - crop_size * 2, w - crop_size * 2, MatType.CV_8UC1, sdim);

            //using (Window win = new Window("Mean", WindowMode.AutoSize, meanmat))
            //{
            //    Cv2.WaitKey();
            //}

            //using (Window win = new Window("SD", WindowMode.AutoSize, sdmat))
            //{
            //    Cv2.WaitKey();
            //}
        }
    }
}
