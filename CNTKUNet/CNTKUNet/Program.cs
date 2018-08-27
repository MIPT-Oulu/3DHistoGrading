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
        //UNet test using simulated data *NOT TESTED WITH ANY DATA, SINCE MODEL INITIALIZATION HAS BUGS*
        static void Main()
        {
            //Parameters for simulated data
            int[] dims = new int[] {384,384,1};
            //Declare new model
            UNet new_unet = new UNet();
            //Initialize
            //new_unet.Initialize(24,dims);

            int[] dims2 = new int[] { 16, 16, 1};
            float[] data = new float[16 * 16];
            for(int k =0; k<16*16; k++)
            {
                data[k] = (float)k;
            }
            Console.WriteLine("Works");

            //Test bilinear upsampling
            float[] upsampled = new_unet.bilinear_upsampling(data,dims2);
            foreach(float item in upsampled)
            {
                Console.WriteLine(item);
            }
            Console.ReadKey();
        }
    }
}
