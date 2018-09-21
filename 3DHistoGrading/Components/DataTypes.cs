using System;
using System.IO;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Kitware.VTK;
using OpenCvSharp;
using OpenCvSharp.Extensions;

using Accord.Math;

namespace HistoGrading.Components
{
    /// <summary>
    /// Contains functions for data type conversions.
    /// </summary>
    public class DataTypes
    {
        /// <summary>
        /// Converts 3D byte array to vtkImageData.
        /// </summary>
        /// <param name="data">Input array.</param>
        /// <returns>Converted array.</returns>
        public static vtkImageData byteToVTK(byte[,,] data, int[] orientation = null)
        {
            //Get input dimensions
            int[] dims = new int[] { data.GetLength(0), data.GetLength(1), data.GetLength(2) };
            //Create VTK data for putput
            vtkImageData vtkdata = vtkImageData.New();
            //Character array for conversion
            vtkUnsignedCharArray charArray = vtkUnsignedCharArray.New();
            //Pin byte array
            GCHandle pinnedArray = GCHandle.Alloc(data, GCHandleType.Pinned);
            //Set character array input
            charArray.SetArray(pinnedArray.AddrOfPinnedObject(), dims[0] * dims[1] * dims[2], 1);
            //Set vtkdata properties and connect array
            //Data from char array
            vtkdata.GetPointData().SetScalars(charArray);
            //Number of scalars/pixel
            vtkdata.SetNumberOfScalarComponents(1);
            //Data extent, 1st and last axis are swapped from the char array
            //Data is converted back to original orientation
            vtkdata.SetExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1);
            //Scalar type
            vtkdata.SetScalarTypeToUnsignedChar();
            vtkdata.Update();

            //Return vtk data
            if (orientation == null)
            {
                return vtkdata;
            }
            else
            {
                vtkImagePermute permuter = vtkImagePermute.New();
                permuter.SetInput(vtkdata);
                permuter.SetFilteredAxes(orientation[0], orientation[1], orientation[2]);
                permuter.Update();
                return permuter.GetOutput();
            }
        }

        /// <summary>
        /// Converts 3D vtkImageData to 1D byte array.
        /// </summary>
        /// <param name="vtkdata">Input data.</param>
        /// <param name="dims">Dimensions of the converted array. Give these as input to <seealso cref="VectorToVolume{T}(T[], int[])"/> function to convert from 1D to 3D.</param>
        /// <returns>Converted 1D array of vtkImageData.</returns>
        public static byte[] vtkToByte(vtkImageData vtkdata)
        {
            //Get vtk data dimensions
            int[] extent = vtkdata.GetExtent();
            int[] dims = new int[] { extent[1] - extent[0] + 1, extent[3] - extent[2] + 1, extent[5] - extent[4] + 1 };
            //New byte array for conversions
            byte[] bytedata = new byte[dims[0] * dims[1] * dims[2]];
            //pin bytedata to memory
            GCHandle pinnedArray = GCHandle.Alloc(bytedata, GCHandleType.Pinned);
            //Get pointer to pinned array
            IntPtr ptr = pinnedArray.AddrOfPinnedObject();
            //VTK exporter
            vtkImageExport exporter = vtkImageExport.New();
            //Connect input data to exporter
            exporter.SetInput(vtkdata);
            exporter.Update();
            //Export data to byte array
            exporter.Export(ptr);
            //Free pinned array
            pinnedArray.Free();
            //Return byte array
            return bytedata;
        }

        /// <summary>
        /// Converts 1D vector to 3D array.
        /// </summary>
        /// <typeparam name="T">Data type for array can be chosen by user.</typeparam>
        /// <param name="vector">1D vector.</param>
        /// <param name="dims">Dimensions of the 3D array. Order: y, x, z.</param>
        /// <returns>3D volume.</returns>
        public static T[,,] VectorToVolume<T>(T[] vector, int[] dims)
        {
            T[,,] volume = new T[dims[0], dims[1], dims[2]];

            Parallel.For(0, dims[0], y =>
            {
                Parallel.For(0, dims[1], x =>
                {
                        Parallel.For(0, dims[2], z =>
                        {
                              volume[y, x, z] = vector[z * dims[0] * dims[1] + y * dims[1] + x];
                        });
                });
            });
            return volume;
        }

        /// <summary>
        /// Extracts 2D array from 3D volume.
        /// </summary>
        /// <typeparam name="T">Data type for array can be chosen by user.</typeparam>
        /// <param name="volume">3D volume.</param>
        /// <param name="n">Number of slice on given axis.</param>
        /// <param name="axis">Axis to obtain slice from.</param>
        /// <returns>2D array.</returns>
        public static T[,] VolumeToSlice<T>(T[,,] volume, int n, int axis)
        {
            T[,] slice;
            int[] dims = new int[] { volume.GetLength(0), volume.GetLength(1), volume.GetLength(2)};

            switch (axis) // Select axis to be sliced from.
            {
                case 0: // yz
                    slice = new T[dims[1], dims[2]];
                    dims[0] = dims[1]; dims[1] = dims[2];
                    // Extract slice
                    Parallel.For(0, dims[0], i =>
                    {
                        Parallel.For(0, dims[1], j =>
                        {
                            slice[i, j] = volume[n, i, j];
                        });
                    });
                    break;

                case 1: // xz
                    slice = new T[dims[0], dims[2]];
                    dims[1] = dims[2];
                    // Extract slice
                    Parallel.For(0, dims[0], i =>
                    {
                        Parallel.For(0, dims[1], j =>
                        {
                            slice[i, j] = volume[i, n, j];
                        });
                    });
                    break;

                case 2: // xy
                    slice = new T[dims[0], dims[1]];
                    // Extract slice
                    Parallel.For(0, dims[0], i =>
                    {
                        Parallel.For(0, dims[1], j =>
                        {
                            slice[i, j] = volume[i, j, n];
                        });
                    });
                    break;

                default:
                    throw new Exception("Invalid axis given. Give axis as an integer between 0 and 2.");
            }
            return slice;
        }

        /// <summary>
        /// Converts byte array to float array. Normalizes the data, if normalization
        /// parameters (mean and standard deviation) are given
        /// </summary>
        /// <returns>Float array.</returns>
        public static float[] byteToFloat(byte[] bytedata, float mu = 0, float sd = 0)
        {
            //Empty array for putput
            float[] floatdata = new float[bytedata.Length];

            //Iterate over bytedata
            Parallel.For(0, bytedata.Length, (int k) =>
            {
                //If normalization parameters are not give, return bytedata cast as float data
                if (mu == 0)
                {
                    floatdata[k] = bytedata[k];
                }
                //Otherwise normalize data
                else
                {
                    floatdata[k] = bytedata[k] - mu;
                    if (sd != 0)
                    {
                        floatdata[k] /= sd;
                    }
                }
            });

            //return float data
            return floatdata;
        }

        /// <summary>
        /// Converts minibatch data to 3D byte array.
        /// </summary>
        /// <returns>3D byte array.</returns>
        public static byte[,,] batchToByte(IList<IList<float>> batch, int[] output_size = null, int[] extent = null)
        {
            //Get number of slices
            int n_slices = batch.Count();
            //Get slice dimensions
            IList<float> tmpL = batch.First();
            float[] tmpA = tmpL.ToArray();
            int dim = (int)Math.Sqrt(tmpA.Length);

            //outarray
            byte[,,] outarray;
            if (output_size == null)
            {
                outarray = new byte[n_slices, dim, dim];
            }
            else
            {
                outarray = new byte[output_size[0], output_size[1], output_size[2]];
            }

            if (extent == null)
            {
                extent = new int[] { 0, n_slices - 1, 0, dim - 1, 0, dim - 1, 0 };
            }
            //Iterate over the list and collect the data to an array
            int d = extent[0];
            int stride = dim * dim;
            foreach (IList<float> item in batch)
            {
                //List to array
                float[] tmp = item.ToArray();
                //Iterate over the array in parallel
                Parallel.For(extent[2], extent[3], (int h) =>
                {
                    Parallel.For(extent[4], extent[5], (int w) =>
                    {
                        int pos = (h - extent[2]) * dim + w - extent[4];
                        byte val = (byte)(tmp[pos] * 255);
                        outarray[d, h, w] = val;
                    });
                });
                d++;
            }

            return outarray;
        }

        /// <summary>
        /// Convert 2D double array to Bitmap.
        /// Scales array from 0 to 255. Should be used mainly for visualizations.
        /// </summary>
        /// <param name="array">2D array to be converted.</param>
        /// <returns>Bitmap</returns>
        public static Bitmap DoubleToBitmap(double[,] array)
        {
            // Scale
            array = LBPLibrary.Functions.Normalize(array).Multiply(255);
            // To byte
            byte[,] bytearray = array.Round().ToByte(); // Round and convert to byte
            // To Bitmap
            return new Bitmap(LBPLibrary.Functions.ByteMatrixToBitmap(bytearray));
        }

    }
}
