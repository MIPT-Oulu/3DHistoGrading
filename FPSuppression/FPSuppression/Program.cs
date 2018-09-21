using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Kitware.VTK;
using OpenCvSharp;
using OpenCvSharp.Extensions;

using HistoGrading.Components;
using HistoGrading.Models;

namespace FPSuppression
{
    class Program
    {
        static void Main(string[] args)
        {
            //Path to data
            string path = "Z:\\3DHistoData\\rekisteroidyt\\14_R3L_2_PTA_48h_Rec\\Registration\\14_R3L_2_PTA_48h__rec_Tar00000046.png";
            string modelpath = "Z:\\Tuomas\\UNetE3bn.h5";

            //New volume
            Rendering.renderPipeLine volume = new Rendering.renderPipeLine();

            int[] voi = new int[] { 300, 684, 300, 684, 100, 501 };
            volume.connectData(path, voi);

            int[] size = volume.getDims();

            //Connect new rendering window
            vtkRenderWindow renWin = vtkRenderWindow.New();
            volume.connectWindow(renWin);
            volume.updateCurrent(new int[] { 300, 300, 150 }, 1, new int[] { 0, 200 });

            //Render
            //volume.renderImage();

            //Segment BCInterface

            //Segmentation reange
            int[] extent = new int[] { size[0], size[1], size[2], size[3], 0, 383 };

            //Segmentation
            vtkImageData mask = IO.segmentation_pipeline(volume, new int[] { 384, 384, 1 }, extent, new int[] { 0 }, 16);
            Console.WriteLine("Inference done!!");
            Console.WriteLine("Suppressing false positives");
            vtkImageData newmask = Processing.FalsePositiveSuppresion(mask, extent, 0.7 * 255.0, new int[] { 0 });
            Console.WriteLine("Done!");
            Console.ReadKey();
            volume.connectMaskFromData(newmask);

            volume.renderImage();
            volume.renderImageMask();
            /*
            for (int iterator = 0; iterator < 100; iterator += 10)
            {
                //Get slice and convert to Mat
                byte[] byteData = DataTypes.vtkToByte(volume.getMaskVOI(new int[] { 0,767, 300+iterator, 300 + iterator, 0, 799 }, new int[] { 0, 2, 1 }));
                byte[] byteData2 = DataTypes.vtkToByte(volume.getVOI(new int[] { 0, 767, 300 + iterator, 300 + iterator, 0, 799 }, new int[] { 0, 2, 1 }));
                Console.WriteLine("Byte conversion done!!");
                //Save output slice
                Mat newmat = new Mat(800, 768, MatType.CV_8UC1, byteData);
                Mat newmat2 = new Mat(800, 768, MatType.CV_8UC1, byteData2);
                newmat.ImWrite(String.Format("d:\\segres{0}.png", iterator));
                newmat2.ImWrite(String.Format("d:\\segim{0}.png", iterator));
                using (var window = new Window("window", image: newmat, flags: WindowMode.AutoSize))
                {
                    Cv2.WaitKey();
                }
                using (var window = new Window("window", image: newmat2, flags: WindowMode.AutoSize))
                {
                    Cv2.WaitKey();
                }
                Mat bw = Processing.LargestBWObject(newmat, 0.7 * 255.0);
                bw.ImWrite(String.Format("d:\\clean{0}.png", iterator));
                using (var window = new Window("window", image: bw, flags: WindowMode.AutoSize))
                {
                    Cv2.WaitKey();
                }

                Console.WriteLine("Mat conversion done");
            }
            //Console.WriteLine("Processing mask");
            /*
            //Create structuring element
            byte[] elarray = new byte[21*21];
            Parallel.For(0, 21*21, (int k) =>
            {
                elarray[k] = 1;
            });
            //Closing
            Mat processed = newmat;
            for(int k = 0; k < 3; k++)
            {
                processed = processed.Erode(new Mat(21, 21, MatType.CV_8UC1, elarray));
                processed = processed.Dilate(new Mat(21, 21, MatType.CV_8UC1, elarray));
                processed.ImWrite(string.Format("d:\\processed_{0}_.png",k));
            }
            newmat = newmat.Mul(processed);
            newmat.ImWrite("d:\\outimg2.png");
            Console.WriteLine("Saving done!!");
            */
            Console.ReadKey();
        }
    }
}
