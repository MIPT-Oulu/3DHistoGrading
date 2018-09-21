using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Kitware.VTK;
using OpenCvSharp;
using OpenCvSharp.Extensions;

using CNTKIntegration.Components;
using CNTKIntegration.Models;

namespace CNTKIntegration
{
    class Program
    {
        static void visualize_data(vtkImageData data)
        {
            //Visualize
            vtkImageData slice = Functions.volumeSlicer(data, new int[] { 0, 150, 0 }, 1);
            vtkImageActor actor = vtkImageActor.New();
            actor.SetInput(slice);
            vtkRenderWindow renWin = vtkRenderWindow.New();
            vtkRenderer renderer = vtkRenderer.New();
            renderer.AddActor(actor);
            renWin.AddRenderer(renderer);
            renWin.Render();
        }
        static void Main(string[] args)
        {
            //Load VOI from CTStack
            string path = "D:\\3D-Histo\\3D_histo_REC_data\\PTAjaCA4+\\13_R3L_2_PTA_48h_Rec\\13_R3L_2_PTA_48h__rec00000044.bmp";
            int[] extent = new int[] { 140, 160, 141, 908, 0, 767 };
            
            //Rendering pipeline
            Rendering.renderPipeLine volume = new Rendering.renderPipeLine();
            volume.connectData(path);
            
            //Connect rendering window
            vtkRenderWindow renWin = vtkRenderWindow.New();
            volume.connectWindow(renWin);
            volume.updateCurrent(new int[] { 150, 150, 150 }, 1, new int[] { 0, 225 });
            
            //Load UNet
            string modelpath = "c:\\users\\jfrondel\\Desktop\\GITS\\UNetE3BN.h5";
            UNet model = new UNet();
            model.Initialize(24, new int[] { 768, 768, 1 }, modelpath, false);
            
            //Segment vtk data
            IList<IList<float>> result = Models.IO.segment_sample(volume, model, extent, 0, 4,(float)113.05652141, (float)39.87462853);
            Console.WriteLine("Inference done!!");

            //Convert back to vtk data
            int[] full_size = volume.getDims();
            int[] array_size = new int[] { full_size[1]+1, full_size[3] + 1, full_size[5] + 1 };
            vtkImageData mask = Models.IO.inference_to_vtk(result, array_size, extent, 1);
            volume.connectMaskFromMemory(mask);
            volume.updateCurrent(new int[] { 150, 150, 150 }, 1, new int[] { 0, 225 });

            Console.WriteLine("VTK conversion done!!");
            //Render input data
            volume.renderVolume();
            volume.setVolumeColor();

            //Connect mask to  render window
            volume.connectMaskFromMemory(mask);
            volume.renderVolumeMask();

            /*
            //Loop over output list and save images
            int d = 0;
            foreach(IList<float> image in output)
            {
                int[] dims = new int[] { extent[1] + 1, extent[5] + 1 };
                var src = new Mat(dims[0], dims[1],  MatType.CV_8UC1);
                var indexer = src.GetGenericIndexer<Vec2b>();

                //Iterator
                int c = 0;
                foreach(float k in image)
                {
                    int h = c / dims[1];
                    int w = c - h * dims[0];
                    
                    int pos = h * dims[1] + w;
                    Vec2b value = indexer[w, h];
                    value.Item0 = (byte)(k*(float)255);
                    indexer[h, w] = value;
                    c += 1;
                }
                string savestring = string.Format("inference{0}.png", d);
                src.SaveImage("c:\\users\\jfrondel\\desktop\\GITS\\"+savestring);
                d += 1;
            }

            Console.WriteLine("Saving Done");
            */
            Console.ReadKey();
        }
    }
}
