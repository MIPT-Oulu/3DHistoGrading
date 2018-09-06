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

namespace HistoGrading.Components
{
    class DataTypes
    {
        //3D byte array to vtkimagedata
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

        //vtkimagedata to 1d byte array
        public static byte[] vtkToByte(vtkImageData vtkdata)
        {
            //Get vtk data dimensions
            int[] extent = vtkdata.GetExtent();
            int[] dims = new int[] { extent[1] - extent[0], extent[3] - extent[2], extent[5] - extent[4] };
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
            Parallel.For(0, bytedata.Length, k =>
            {
                //If normalization parameters are not give, return bytedata cast as float data
                if (mu == 0)
                {
                    floatdata[k] = (float)bytedata[k];
                }
                //Otherwise normalize data
                else
                {
                    floatdata[k] = (float)bytedata[k] - mu;
                    if (sd != 0)
                    {
                        floatdata[k] /= sd;
                    }
                }
            });

            //return float data
            return floatdata;
        }
    }
}