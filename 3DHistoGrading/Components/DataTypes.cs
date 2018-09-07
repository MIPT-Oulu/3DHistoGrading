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
        public static vtkImageData byteToVTK(byte[,,] data)
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
            return vtkdata;
        }

        /// <summary>
        /// Converts 3D vtkImageData to 1D byte array.
        /// </summary>
        /// <param name="vtkdata">Input data.</param>
        /// <param name="dims">Dimensions of the converted array. Give these as input to <seealso cref="VectorToVolume{T}(T[], int[])"/> function to convert from 1D to 3D.</param>
        /// <returns>Converted 1D array of vtkImageData.</returns>
        public static byte[] vtkToByte(vtkImageData vtkdata, out int[] dims)
        {
            //Get vtk data dimensions
            int[] extent = vtkdata.GetExtent();
            dims = new int[] { extent[1] - extent[0] + 1, extent[3] - extent[2] + 1, extent[5] - extent[4] + 1 };
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
    }
}
