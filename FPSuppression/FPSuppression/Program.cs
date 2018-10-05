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
            string path = "Z:\\3DHistoData\\rekisteroidyt\\32_L6MT_2_PTA_48h_Rec\\32_L6MT_2_PTA_48h_Rec\\Registration\\32_L6MT_2_PTA_48h__rec_Tar00000038.png";
            string modelpath = "Z:\\Tuomas\\UNetE3bn.h5";

            Mat im = new Mat(path, ImreadModes.GrayScale);
            Mat BW = im.Threshold(80.0,255.0,ThresholdTypes.Binary);
            Mat E = BW.Sobel(MatType.CV_8UC1, 1, 1);
            Mat[] conts;            
            HierarchyIndex[] H;

            E.FindContours(out conts, H, RetrievalModes.List,ContourApproximationModes.ApproxTC89KCOS);
            
            InputArray.Create(conts);
            Rect bbox = Cv2.BoundingRect(conts);

            int xmin = bbox.Left;
            int xmax = bbox.Right;

            int ymin = bbox.Bottom;
            int ymax = bbox.Top;

            Console.WriteLine("{0},{1}", xmin, ymin);
            Console.WriteLine("{0},{1}", xmin, ymax);
            Console.WriteLine("{0},{1}", xmax, ymin);
            Console.WriteLine("{0},{1}", xmax, ymax);
            Console.ReadKey();
            //New volume
            Rendering.renderPipeLine volume = new Rendering.renderPipeLine();

            int[] voi = new int[] { 101, 909, 101, 909, 0, 800 };
            volume.connectData(path, voi);

            int[] size = volume.getDims();

            //Connect new rendering window
            vtkRenderWindow renWin = vtkRenderWindow.New();
            volume.connectWindow(renWin);
            volume.updateCurrent(new int[] { 500, 500, 150 }, 1, new int[] { 0, 200 });

            //Render
            //volume.renderImage();

            //Segment BCInterface

            //Segmentation reange
            int[] extent = new int[] { 0, 767, 0, 767, 0, 767 };

            //Segmentation
            List<vtkImageData> outputs = new List<vtkImageData>();
            IO.segmentation_pipeline(out outputs, volume, new int[] { 768, 768, 1 }, extent, new int[] { 0 }, 16);            
            //Suppress false positives            
            Console.WriteLine("Suppressing false positives");
            vtkImageData newmask1 = Processing.SWFPSuppression(outputs.ElementAt(0), extent);

            volume.connectMaskFromData(newmask1);

            vtkImageData I1 = volume.getVOI(new int[] { 0, 767, 500, 500, 0, 799 });
            vtkImageData I2 = volume.getMaskVOI(new int[] { 0, 767, 500, 500, 0, 799 });

            byte[] B1 = DataTypes.vtkToByte(I1);
            byte[] B2 = DataTypes.vtkToByte(I2);

            Mat image = new Mat(800,768,MatType.CV_8UC1,B1);
            Mat image_mask = new Mat(800, 768, MatType.CV_8UC1, B2);

            using (var window = new Window("window", image: image, flags: WindowMode.AutoSize))
            {
                Cv2.WaitKey();
            }

            using (var window = new Window("window", image: image_mask, flags: WindowMode.AutoSize))
            {
                Cv2.WaitKey();
            }


            Console.ReadKey();
        }
    }
}
