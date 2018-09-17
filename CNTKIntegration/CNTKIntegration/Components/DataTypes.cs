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

namespace CNTKIntegration.Components
{
    class DataTypes
    {
        //3D byte array to vtkimagedata
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
            if(orientation == null)
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

        //vtkimagedata to 1d byte array
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

        //byte array to float
        public static float[] byteToFloat(byte[] bytedata, float mu = (float)0, float sd = (float)0)
        {
            //Empty array for putput
            float[] floatdata = new float[bytedata.Length];

            //Iterate over bytedata
            Parallel.For(0,bytedata.Length, (int k) =>
            {
                //If normalization parameters are not give, return bytedata cast as float data
                if (mu == 0)
                {
                    floatdata[k] = (float)bytedata[k];
                }
                //Otherwise normalize data
                else
                {
                    floatdata[k] = (float)bytedata[k]-mu;
                    if (sd != 0)
                    {
                        floatdata[k] /= sd;
                    }
                }
            });

            //return float data
            return floatdata;
        }

        //Minibatch to byte
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
            if(output_size == null)
            {
                outarray = new byte[n_slices, dim, dim];
            }
            else
            {
                outarray = new byte[output_size[0], output_size[1], output_size[2]];
            }

            if(extent == null)
            {
                extent = new int[] {0,n_slices-1,00,dim-1,0,dim-1,0};
            }
            //Iterate over the list and collect the data to an array
            int d = extent[0];
            int stride = dim * dim;
            foreach(IList<float> item in batch)
            {
                //List to array
                float[] tmp = item.ToArray();
                //Iterate over the array in parallel
                Parallel.For(extent[2], extent[3], (int h) =>
                {
                    Parallel.For(extent[4], extent[5], (int w) =>
                    {
                        int pos = (h - extent[2]) * dim + w - extent[4];
                        byte val = (byte)(tmp[pos] * (float)255);
                        outarray[d, h, w] = val;
                    });
                });
                d++;
            }

            return outarray;
        }
    }
}
