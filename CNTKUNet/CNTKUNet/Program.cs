using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNTK;

using CNTKUNet.Models;

namespace CNTKUNet
{
    class Program
    {
        //UNet test using simulated data
        static void Main()
        {
            //Path to weights
            string wpath = "c:\\users\\jfrondel\\Desktop\\GITS\\UNetE3.h5";

            //Parameters for simulated data
            int[] dims = new int[] {384,384,1};
            //Declare new model
            UNet new_unet = new UNet();
            //Initialize the model
            new_unet.Initialize(24,dims,wpath);

            //Generate data
            float[] data = new float[dims[0] * dims[1]];
            for(int k =0; k<data.Length; k++)
            {
                data[k] = (float)k/(384*384);
                //Console.WriteLine(data[k]);
            }

            //Inference
            float[] output = new_unet.Inference(data);

            Console.WriteLine("Inference done!");

            Console.ReadKey();
            
        }
    }
}
