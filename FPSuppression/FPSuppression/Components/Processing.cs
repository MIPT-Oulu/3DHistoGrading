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
            // Convert vtkImageData to byte[,,]
            int[] dims = volume.getDims();
            dims = new int[] { dims[1] + 1, dims[3] + 1, dims[5] + 1 };
            byte[,,] byteVolume =
                DataTypes.VectorToVolume(
                DataTypes.vtkToByte(volume.idata), dims);

            // Crop to upper third of the sample
            int[] crop = { 0, byteVolume.GetLength(0) - 1, 0, byteVolume.GetLength(1) - 1, 0, (int)Math.Floor((double)byteVolume.GetLength(2) / 3) };
            byteVolume = Functions.Crop3D(byteVolume, crop);

            // Get sample center coordinates
            int[] center = GetCenter(byteVolume, threshold);

            // Get surface
            GetSurface(byteVolume, center, size, threshold, out surfacecoordinates, out surfacevoi);

            // Free memory
            byteVolume = null;
        }

        /// <summary>
        /// Computes mean along each column of the array
        /// and subtracts it along the columns.
        /// </summary>
        /// <param name="array">Array to be calculated.</param>
        /// <returns>Subtracted array.</returns>
        public static double[,] SubtractMean(double[,] array)
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
                means[i] = vector.Average();
                vector = Elementwise.Subtract(vector, means[i]);

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
        /// Finds largest connected binary component from a 2D 8-bit grayscale image
        /// </summary>
        /// <param name="image">Input image.</param>
        /// <param name="threshold">Threshold value.</param>
        /// <returns></returns>
        public static Mat LargestBWObject(Mat image, double threshold)
        {
            //Binarize the inputimage
            Mat bw = image.Threshold(threshold, 255.0, ThresholdTypes.Binary);
            //Empty label image
            OutputArray labelArray = OutputArray.Create(new Mat(bw.Height, bw.Width, MatType.CV_16SC1));
            //Get connected components
            bw.ConnectedComponents(labelArray, PixelConnectivity.Connectivity8);
            //Get label min and max
            double min, max;
            labelArray.GetMat().MinMaxIdx(out min, out max);
            //Iterate over the labels and select largest connected component
            Scalar curmax = 0;
            Mat output = new Mat();
            for (int k = 1; k < max + 1; k++)
            {
                Mat _tmp = selectGS(labelArray.GetMat(), (double)k);
                Scalar _sum = _tmp.Sum();
                if (_sum.Val0 > curmax.Val0)
                {
                    curmax = _sum;
                    output = _tmp;
                }
            }
            return output;
        }

        /// <summary>
        /// Returns 
        /// </summary>
        /// <param name="image"></param>
        /// <param name="val"></param>
        /// <returns></returns>
        public static Mat selectGS(Mat image, double val)
        {
            Mat bw1 = image.LessThanOrEqual(val);
            Mat bw2 = image.GreaterThanOrEqual(val);
            return bw1.Mul(bw2);
        }

        /// <summary>
        /// Scans the input mask slice by slice and selects the largest binary component of each slice.
        /// Return cleaned mask as vtkImageData
        /// </summary>
        /// <param name="input"></param>
        /// <param name="extent"></param>
        /// <param name="threshold"></param>
        /// <param name="axes"></param>
        /// <returns></returns>
        public static vtkImageData FalsePositiveSuppresion(vtkImageData input, int[] extent, double threshold, int axis, double scale = 1.0)
        {
            //Slice extractor
            vtkExtractVOI slicer = vtkExtractVOI.New();
            //Permuter
            vtkImagePermute permuter = vtkImagePermute.New();
            //List of outputs
            List<byte[,,]> outputs = new List<byte[,,]>();
            //List of output orientations
            List<int[]> orientations = new List<int[]>();
            //vtkImageData size
            int[] full_extent = input.GetExtent();

            //Set range for slices
            int start = 0, stop = 0;
            int[] size = new int[2];
            int[] outextent = new int[4];
            if (axis == 0)
            {
                start = extent[0];
                stop = extent[1];
                size = new int[] { extent[3] - extent[2] + 1, extent[5] - extent[4] + 1 };
                outextent = new int[] { extent[2], extent[3] + 1, extent[4], extent[5] + 1 };
            }
            if (axis == 1)
            {
                start = extent[2];
                stop = extent[3];
                size = new int[] { extent[1] - extent[0] + 1, extent[5] - extent[4] + 1 };
                outextent = new int[] { extent[0], extent[1] + 1, extent[4], extent[5] + 1 };
            }
            if (axis == 2)
            {
                start = extent[4];
                stop = extent[5];
                size = new int[] { extent[1] - extent[0] + 1, extent[3] - extent[2] + 1 };
                outextent = new int[] { extent[0], extent[1] + 1, extent[2], extent[3] + 1 };
            }

            //Temporary array for output
            byte[,,] output = new byte[full_extent[1] + 1, full_extent[3] + 1, full_extent[5] + 1];
            int[] outsize = new int[] { size[0], size[1], stop - start + 1 };
            //Loop over current axis
            for (int k = start; k < stop; k++)
            {
                byte[] bytedata = new byte[size[0] * size[1]];
                //Select slice
                if (axis == 0)
                {
                    slicer.Dispose();
                    slicer = vtkExtractVOI.New();
                    slicer.SetInput(input);
                    slicer.SetVOI(k, k, extent[2], extent[3], extent[4], extent[5]);
                    slicer.Update();
                    permuter.Dispose();
                    permuter = vtkImagePermute.New();
                    permuter.SetInput(slicer.GetOutput());
                    permuter.SetFilteredAxes(1, 2, 0);
                    permuter.Update();
                }
                if (axis == 1)
                {
                    slicer.Dispose();
                    slicer = vtkExtractVOI.New();
                    slicer.SetInput(input);
                    slicer.SetVOI(extent[0], extent[1], k, k, extent[4], extent[5]);
                    slicer.Update();
                    permuter.Dispose();
                    permuter = vtkImagePermute.New();
                    permuter.SetInput(slicer.GetOutput());
                    permuter.SetFilteredAxes(0, 2, 1);
                    permuter.Update();
                }
                if (axis == 2)
                {
                    slicer.Dispose();
                    slicer = vtkExtractVOI.New();
                    slicer.SetInput(input);
                    slicer.SetVOI(extent[0], extent[1], extent[2], extent[3], k, k);
                    slicer.Update();
                    permuter.Dispose();
                    permuter = vtkImagePermute.New();
                    permuter.SetInput(slicer.GetOutput());
                    permuter.SetFilteredAxes(0, 1, 2);
                    permuter.Update();
                }
                //Convert data to byte
                bytedata = DataTypes.vtkToByte(permuter.GetOutput());
                slicer.Dispose();
                permuter.Dispose();
                //convert data to Mat
                Mat image = new Mat(size[1], size[0], MatType.CV_8UC1, bytedata);
                //Get largest binary object
                Mat bw = Processing.LargestBWObject(image, 0.7 * 255.0);
                //Set slice to byte array
                if (bw.Sum().Val0 > 0)
                {
                    output = DataTypes.setByteSlice(output, bw, outextent, axis, k);
                }
            }

            return DataTypes.byteToVTK(output);
        }

        public static vtkImageData vtkFPSuppression(vtkImageData mask, vtkImageData samplemask, int kernel_size = 25)
        {
            //Subtract BCI mask from cartilage mask
            vtkImageMathematics math = vtkImageMathematics.New();
            math.SetInput1(samplemask);
            math.SetInput2(mask);
            math.SetOperationToSubtract();
            math.Update();

            //Closing
            vtkImageDilateErode3D dilate = vtkImageDilateErode3D.New();
            dilate.SetErodeValue(0.0);
            dilate.SetDilateValue(1.0);
            dilate.SetKernelSize(kernel_size,kernel_size,kernel_size);
            dilate.SetInputConnection(math.GetOutputPort());
            dilate.Update();

            vtkImageDilateErode3D erode = vtkImageDilateErode3D.New();
            erode.SetErodeValue(1.0);
            erode.SetDilateValue(0.0);
            erode.SetKernelSize(kernel_size, kernel_size, kernel_size);
            erode.SetInputConnection(dilate.GetOutputPort());
            erode.Update();

            //Subtract closed mask from BCI mask
            math.Dispose();
            math = vtkImageMathematics.New();
            math.SetInput1(mask);
            math.SetInput2(erode.GetOutput());
            math.SetOperationToSubtract();
            math.Update();

            //Dilate the BCI mask
            dilate.Dispose();
            dilate = vtkImageDilateErode3D.New();
            dilate.SetErodeValue(0.0);
            dilate.SetDilateValue(1.0);
            dilate.SetKernelSize(kernel_size, kernel_size, kernel_size);
            dilate.SetInputConnection(math.GetOutputPort());
            dilate.Update();

            //Multiply the original BCI mask with the dilated
            math.Dispose();
            math = vtkImageMathematics.New();
            math.SetInput1(mask);
            math.SetInput2(dilate.GetOutput());
            math.SetOperationToMultiply();
            math.Update();

            return math.GetOutput();
        }

        public static vtkImageData OCVFPSuppression(vtkImageData BCI, vtkImageData sample, int axis = 0, double BC_threshold = 0.7 * 255.0, double sample_threshold = 80)
        {
            //Get dimensions
            int[] dims = BCI.GetExtent();

            //Get byte data from input samples
            byte[] BCIByte = DataTypes.vtkToByte(BCI);
            byte[] SampleByte = DataTypes.vtkToByte(sample);

            //Output byte array
            byte[,,] OutByte = new byte[(dims[1] - dims[0] + 1), (dims[3] - dims[2] + 1), (dims[5] - dims[4] + 1)];

            //Loop parameters            
            int start = 0; int stop = 0; int h = 0; int w = 0; int strideh = 0; int stridew = 0; int pos = 0;

            if (axis == 0)
            {
                start = dims[0];
                stop = dims[1] + 1;
                h = dims[5] + 1;
                w = dims[3] + 1;
                strideh = (dims[1] + 1) * (dims[3] + 1);
                stridew = 1;
            }
            if (axis == 1)
            {
                start = dims[2];
                stop = dims[3] + 1;
                h = dims[5] + 1;
                w = dims[1] + 1;
                strideh = (dims[1] + 1) * (dims[3] + 1);
                stridew = (dims[3] + 1);
            }
            if (axis == 2)
            {
                start = dims[4];
                stop = dims[5] + 1;
                h = dims[1] + 1;
                w = dims[3] + 1;
                strideh = dims[3] + 1;
                stridew = 1;
            }

            //Iterate over samples in parallel
            Parallel.For(start, stop, (int k) =>
            {
                //Get slices to OpenCV matrices
                Mat BCMat = new Mat(h,w,MatType.CV_8UC1);
                var BCIndexer = BCMat.GetGenericIndexer<Vec2b>();
                Mat SampleMat = new Mat(h, w, MatType.CV_8UC1);
                var SampleIndexer = SampleMat.GetGenericIndexer<Vec2b>();

                //Iterate over indices
                Parallel.For(0, h, (int kh) =>
                {
                    Parallel.For(0, w, (int kw) =>
                    {
                        //Current slice index                        
                        if (axis == 0){ pos = k * (dims[1] + 1); };
                        if (axis == 1) { pos = k; };
                        if (axis == 2) { pos = k * (dims[1] + 1) * (dims[3] + 1); };

                        //Current index
                        int cur = pos + kh * strideh + kw * stridew;

                        //Get values
                        Vec2b bcval = BCIndexer[kh,kw];
                        bcval.Item0 = BCIByte[cur];
                        bcval.Item1 = BCIByte[cur];

                        Vec2b sampleval = SampleIndexer[kh, kw];
                        sampleval.Item0 = SampleByte[cur];
                        sampleval.Item1 = SampleByte[cur];

                        //Update matrices
                        BCIndexer[kh, kw] = bcval;
                        SampleIndexer[kh, kw] = sampleval;
                    });
                });

                if (k == 500)
                {
                    using (var window = new Window("BCMat", image: BCMat, flags: WindowMode.AutoSize))
                    {
                        Cv2.WaitKey();
                    }

                    using (var window = new Window("Sample", image: SampleMat, flags: WindowMode.AutoSize))
                    {
                        Cv2.WaitKey();
                    }
                }
                

                

                //Get binary masks
                Mat BCBW = BCMat.Threshold(BC_threshold,255.0,ThresholdTypes.Binary);
                Mat SampleBW = SampleMat.Threshold(sample_threshold,255.0, ThresholdTypes.Binary);

                if (k == 500)
                {
                    using (var window = new Window("BCMat", image: BCBW, flags: WindowMode.AutoSize))
                    {
                        Cv2.WaitKey();
                    }

                    using (var window = new Window("Sample", image: SampleBW, flags: WindowMode.AutoSize))
                    {
                        Cv2.WaitKey();
                    }
                }

                //Subtract BCI mask from sample mask
                Mat D = SampleBW - BCBW;
                if (k == 500)
                {
                    using (var window = new Window("D", image: D, flags: WindowMode.AutoSize))
                    {
                        Cv2.WaitKey();
                    }
                }
                //Closing

                //Generate structuring element
                InputArray element = InputArray.Create(new Mat(1,50,MatType.CV_8UC1));

                D = D.Dilate(element);
                D = D.Erode(element);

                if (k == 500)
                {
                    using (var window = new Window("D", image: D, flags: WindowMode.AutoSize))
                    {
                        Cv2.WaitKey();
                    }
                }

                //Subtract closed mask from BCI mask
                BCBW = BCBW - D;

                if (k == 500)
                {
                    using (var window = new Window("Cleaned BC", image: BCBW, flags: WindowMode.AutoSize))
                    {
                        Cv2.WaitKey();
                    }
                }

                //Dilate
                BCBW = BCBW.Dilate(element);

                if (k == 500)
                {
                    using (var window = new Window("Dilated BC", image: BCBW, flags: WindowMode.AutoSize))
                    {
                        Cv2.WaitKey();
                    }
                }

                //Multiply BCI matrix with the mask
                BCMat = BCMat.Mul(BCBW);
                BCIndexer = BCMat.GetGenericIndexer<Vec2b>();

                if (k == 500)
                {
                    using (var window = new Window("Dilated BC", image: BCBW, flags: WindowMode.AutoSize))
                    {
                        Cv2.WaitKey();
                    }
                }

                //Return the elements to output byte array                
                Parallel.For(0, h, (int kh) =>
                {
                    Parallel.For(0, w, (int kw) =>
                    {
                        //Get value
                        Vec2b bcval = BCIndexer[kh, kw];

                        //Update outputarray
                        if (axis == 0 && bcval.Item0 > BC_threshold) { OutByte[kh, k, kw] = 255; };
                        if (axis == 1 && bcval.Item0 > BC_threshold) { OutByte[kh, kw, k] = 255; };
                        if (axis == 2 && bcval.Item0 > BC_threshold) { OutByte[k, kh, kw] = 255; };
                    });
                });
            });

            //Convert outputarray to VTK data
            return DataTypes.byteToVTK(OutByte);
        }

        public static vtkImageData SWFPSuppression(vtkImageData BCI, int[] extent, double threshold = 0.7 * 255.0, int min = 0, int max = 650, int step = 4, int ws = 50, double wt = 20.0, double minones = 192.0)
        {
            //Get dimensions
            int[] dims = BCI.GetExtent();
            int h = dims[1] - dims[0] + 1;
            int w = dims[3] - dims[2] + 1;
            int d = dims[5] - dims[4] + 1;

            //Make array of extents
            List<int[]> regions = new List<int[]>();
            for (int k1 = 0; k1 < 4; k1++)
            {
                for (int k2 = 0; k2 < 4; k2++)
                {
                    int step1 = (extent[1] - extent[0] + 1) / 8;
                    int step2 = (extent[3] - extent[2] + 1) / 8;
                    int[] _tmp = new int[] { extent[0] + k1*step1, extent[0] + (k1+1) * step1, extent[2] + k2 * step2, extent[2] + (k2 + 1) * step2 };
                    regions.Add(_tmp);
                    Console.WriteLine("Region:");
                    Console.WriteLine("{0},{1},{2},{3}",_tmp[0],_tmp[1],_tmp[2],_tmp[3]);
                }
            }

            //Get byte data from input samples
            byte[] BCIByte = DataTypes.vtkToByte(BCI);
            List<byte[]> sums = new List<byte[]>();

            //Itereate over pixels
            foreach(int[] region in regions)
            {
                byte[] tmp = new byte[max-min];
                for (int k = min; k < max; k += step)
                {
                    int w1 = k; int w2 = w1 + ws;

                    int roisum = 0;

                    Parallel.For(w1, w2, (int kz) =>
                    {
                        Parallel.For(region[0], region[1], (int ky) =>
                        {
                            Parallel.For(region[2], region[3], (int kx) =>
                            {
                                int pos = kz * (h * w) + ky * w + kx;
                                if ((double)BCIByte[pos] > threshold)
                                {
                                    roisum += (int)BCIByte[pos];
                                }
                            });
                        });
                    });

                    //Set sliding window index at k to 1 if enough non-zero pixels were found
                    if (((double)roisum / (minones * minones)) > wt)
                    {
                        tmp[k] = 1;                        
                    }


                    //Check if there's a drop between consecutive results
                    if (k > min + step && tmp[k] == 0 && tmp[k - step] == 1)
                    {
                        sums.Add(tmp);
                        break;
                    }
                    //Otherswise append the data at the end
                    else
                    {
                        if (k + step >= max)
                        {
                            sums.Add(tmp);
                        }
                    }
                }
            }

            int[] idx = new int[sums.Count];

            //Scan sums vector find last non-zero index
            int c = 0;
            foreach (byte[] sum in sums)
            {
                for (int k = sum.Length - 1; k >= 0; k -= 1)
                {
                    if (sum[k] == 1)
                    {
                        idx[c] = k;
                        c++;
                        break;
                    }
                }
            }

            byte[,,] output = new byte[d,h,w];

            //Set array indices below idx to outputvalues
            c = 0;
            foreach (int[] region in regions)
            {
                int _idx = idx[c];
                Console.WriteLine("Max idx: {0} | Height: {1}", _idx, d);
                int z = 0;
                if (_idx == 0){z = d;}
                else{ z = Math.Min(_idx + 20, d); }
                Parallel.For(0, z, (int kz) =>
                {
                    Parallel.For(region[0], region[1], (int ky) =>
                    {
                        Parallel.For(region[2], region[3], (int kx) =>
                        {
                            int pos = kz * (h * w) + ky * (w) + kx;
                            output[kz, ky, kx] = BCIByte[pos];
                        });
                    });
                });
                c++;
            }

            return DataTypes.byteToVTK(output);
        }
    }
}
