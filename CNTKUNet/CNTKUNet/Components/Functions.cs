using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace CNTKUNet.Components
{
    class Functions
    {   //Read image
        public static float[,,] readImage(string file, int[] dims)
        {
            //Read image from file idx. The image is read using OpenCV, and converted to Bitmap.
            //Bitmap is then read to the bytearray.
            Mat _tmp = new Mat(file, ImreadModes.GrayScale);
            Bitmap _image = BitmapConverter.ToBitmap(_tmp);
            //Lock bits
            Rectangle _rect = new Rectangle(0, 0, dims[1], dims[0]);
            BitmapData _bmpData =
                _image.LockBits(_rect, ImageLockMode.ReadOnly, _image.PixelFormat);

            //Get the address of first line
            IntPtr _ptr = _bmpData.Scan0;

            //Declare new array for gray scale values
            int _bytes = Math.Abs(_bmpData.Stride) * _bmpData.Height;
            byte[] _grayValues = new byte[_bytes];

            //Copy the rgb values to the new array
            Marshal.Copy(_ptr, _grayValues, 0, _bytes);

            //Method for correct pixel mapping
            Func<int, int, int, int> mapPixel = GetPixelMapper(_image.PixelFormat, _bmpData.Stride);

            //Read bits to byte array in parallel

            float[,,] data = new float[dims[0], dims[1], 1];

            Parallel.For(0, dims[0], (int h) =>
            {
                Parallel.For(0, dims[1], (int w) =>
                {
                    data[h, w, 0] = (float)_grayValues[mapPixel(h, w, 0)];
                });
            });

            return data;
        }

        //Save byte array to image
        public static void writeImage(byte[] imagedata, int[] dims)
        {
            var src = new Mat(dims[1], dims[0], MatType.CV_8UC1);
            var indexer = src.GetGenericIndexer<Vec2b>();
            for(int h = 0; h<dims[0]; h++)
            {
                for(int w = 0; w<dims[1]; w++)
                {
                    int pos = h * dims[1] + w;
                    Vec2b value = indexer[w,h];
                    value.Item0 = imagedata[pos];
                    indexer[h, w] = value;
                }
            }
            src.SaveImage("c:\\users\\jfrondel\\desktop\\GITS\\output.bmp");
        }

        //Function for correctly mapping the pixel values, copied from CNTK examples
        private static Func<int, int, int, int> GetPixelMapper(PixelFormat pixelFormat, int stride)
        {
            switch (pixelFormat)
            {
                /*
                case PixelFormat.Format32bppArgb:
                    return (h, w, c) => h * stride + w * 4 + c;  // bytes are B-G-R-A
                case PixelFormat.Format24bppRgb:
                    return (h, w, c) => h * stride + w * 3 + c;  // bytes are B-G-R
                */
                case PixelFormat.Format8bppIndexed:
                default:
                    return (h, w, c) => h * stride + w;
            }
        }
    }
}
