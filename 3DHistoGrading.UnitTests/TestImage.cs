using System;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Text;
using System.Threading.Tasks;
using LBPLibrary;

using Accord.Math;

namespace _3DHistoGrading.UnitTests
{
    /// <summary>
    /// Class for creating test images. 
    /// Contains also static methods for displaying arrays and vectors during tests.
    /// </summary>
    public class TestImage
    {
        // Class properties
        public float[,] Image { get; set; }
        public string Method { get; set; }
        public int[] Size { get; set; }
        public Bitmap Bmp { get; set; }

        // Create test image
        public void New()
        {   /// Default image type: "Running numbers".
            /// Other types are "Quarters" and "Ones"

            Method = "Running numbers";
            Size = new int[] { 10, 15 };
            Image = CreateImage(Method, Size);
            Bmp = Functions.ByteMatrixToBitmap(Image.ToByte());
        }

        public void New(string method) // Image type input
        {
            Method = method;
            Size = new int[] { 10, 15 };
            Image = CreateImage(Method, Size);
            if (Image.Max() < 256)
                Bmp = Functions.ByteMatrixToBitmap(Image.ToByte());
        }

        public void New(int[] size) // Size input
        {
            Method = "Running numbers";
            Size = size;
            Image = CreateImage(Method, Size);
            if (Image.Max() < 256)
                Bmp = Functions.ByteMatrixToBitmap(Image.ToByte());
        }

        public void New(string method, int[] size) // Type and size input
        {
            Method = method; Size = size;
            Image = CreateImage(Method, Size);
            if (Image.Max() < 256)
                Bmp = Functions.ByteMatrixToBitmap(Image.ToByte());
        }

        public float[,] CreateImage(string method, int[] size) // Create float image using given properties
        {
            float[,] image = new float[size[0], size[1]];

            if (method == "Quarters")
            {
                for (int i = 0; i < Math.Floor((double)size[0] / 2); i++) // 1st quarter
                {
                    for (int j = 0; j < Math.Floor((double)size[1] / 2); j++)
                    { image[i, j] = 1; }
                }
                for (int i = (int)Math.Floor((double)size[0] / 2); i < size[0]; i++) // 2nd quarter
                {
                    for (int j = 0; j < Math.Floor((double)size[1] / 2); j++)
                    { image[i, j] = 2; }
                }
                for (int i = 0; i < Math.Floor((double)size[0] / 2); i++) // 3rd quarter
                {
                    for (int j = (int)Math.Floor((double)size[1] / 2); j < size[1]; j++)
                    { image[i, j] = 3; }
                }
                for (int i = (int)Math.Floor((double)size[0] / 2); i < size[0]; i++) // 4th quarter
                {
                    for (int j = (int)Math.Floor((double)size[1] / 2); j < size[1]; j++)
                    { image[i, j] = 4; }
                }
            }

            if (method == "Running numbers" || method == "Add residual")
            {
                for (int i = 0; i < size[0]; i++)
                {
                    for (int j = 0; j < size[1]; j++)
                    {
                        image[i, j] = (i + j);
                    }
                }
            }

            if (method == "Add residual")
                image = image.Add(0.00001).ToSingle();

            if (method == "Ones")
            {
                for (int i = 0; i < size[0]; i++)
                {
                    for (int j = 0; j < size[1]; j++)
                    {
                        image[i, j] = 1;
                    }
                }
            }
            return image;
        }

        /// <summary>
        /// Displays 2D array on command line separated by ":".
        /// Decimal values are shown with 2 digit accuracy.
        /// </summary>
        /// <typeparam name="T">Data type can be chosen by user.</typeparam>
        /// <param name="array">Array to be displayed.</param>
        public static void DisplayArray<T>(T[,] array)
        {  
            for (int k = 0; k < array.GetLength(1); k++)
            {
                for (int kk = 0; kk < array.GetLength(0); kk++)
                {
                    Console.Write("{0:####.##}:", array[kk, k].ToString());
                }
                Console.WriteLine("");
            }
            Console.WriteLine("");
        }

        /// <summary>
        /// Displays 1D vector on command line separated by ":".
        /// Decimal values are shown with 2 digit accuracy.
        /// </summary>
        /// <typeparam name="T">Data type can be chosen by user.</typeparam>
        /// <param name="vector">Vector to be displayed.</param>
        public static void DisplayVector<T>(T[] vector)
        {   
            for (int k = 0; k < vector.Length; k++)
            {
                Console.Write("{0:####.##}:", vector[k].ToString());
            }
            Console.WriteLine("");
        }
    }
}
