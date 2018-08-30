using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNTK;

using CNTKUNet.Components;

using CNTKUNet.Models;

namespace CNTKUNet
{
    class Program
    {
        //UNet test using simulated data
        static void Main()
        {
            //Path to weights
            string wpath = "c:\\users\\jfrondel\\Desktop\\GITS\\UNetE3bn.h5";

            //Path to test image
            string impath = "c:\\users\\jfrondel\\desktop\\GITS\\sample.png";
            //Image dimensions
            int[] dims = new int[] { 384, 384, 1 };
            //Load test image
            float[,,] imdata = Functions.readImage(impath, dims);
            Console.WriteLine(imdata.GetLength(0));
            Console.WriteLine(imdata.GetLength(1));
            Console.WriteLine(imdata.GetLength(2));
            //Flatten the data adn normalize
            float mu = (float)113.05652141; float sd = (float)39.87462853;
            float[] dataflat = new float[dims[0]*dims[1]];
            for (int k = 0; k < dims[0]; k++)
            {
                for (int kk = 0; kk < dims[1]; kk++)
                {
                    dataflat[k* dims[1] + kk] = ((float)imdata[k, kk, 0] - mu) / sd;
                }
            }
            //Declare new model
            UNet new_unet = new UNet();
            //Initialize the model
            new_unet.Initialize(24, dims, wpath);

            //Inference
            float[] output = new_unet.Inference(dataflat);

            //Convert to byte
            byte[] outbyte = new byte[dims[0]*dims[1]];

            for (int k = 0; k < output.Length; k++)
            {
                outbyte[k] = (byte)(Math.Floor(output[k] * (float)255));
            }
            

            Console.WriteLine("Inference done!");

            Functions.writeImage(outbyte, dims);

            Console.ReadKey();

        }
    }

}
