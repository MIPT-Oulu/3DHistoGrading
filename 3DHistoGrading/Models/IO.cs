﻿using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNTK;
using Kitware.VTK;

using HistoGrading.Components;

namespace HistoGrading.Models
{
    class IO
    {
        public static IList<IList<float>> segment_sample(Rendering.renderPipeLine vtkObject, UNet model, int[] extent, int axis,
            int step = 1, float mu = 0, float sd = 0)
        {
            //Segmentation range
            int[] bounds = new int[] { extent[axis * 2], extent[axis * 2 + 1] + 1 };
            //Output list
            IList<IList<float>> output = null;
            //Iterate over vtk data
            for (int k = 0; k < (bounds[1] - bounds[0]) / step; k++)
            {
                //Set current VOI and orientation
                int[] _curext = new int[6];
                int[] _ori = new int[3];
                if (axis == 0)
                {
                    int start = extent[0] + k * step;
                    int stop = Math.Min(start + step - 1, extent[1]);
                    _curext = new int[] { start, stop, extent[2], extent[3], extent[4], extent[5] };
                    _ori = new int[] { 1, 2, 0 };
                }
                if (axis == 1)
                {
                    int start = extent[2] + k * step;
                    int stop = Math.Min(start + step - 1, extent[3]);
                    _curext = new int[] { extent[0], extent[1], start, stop, extent[4], extent[5] };
                    _ori = new int[] { 0, 2, 1 };
                }
                if (axis == 2)
                {
                    int start = extent[4] + k * step;
                    int stop = Math.Min(start + step - 1, extent[5]);
                    _curext = new int[] { extent[0], extent[1], extent[2], extent[3], start, stop };
                    _ori = new int[] { 0, 1, 2 };
                }

                //Extract VOI to float array
                float[] input_array = DataTypes.byteToFloat(DataTypes.vtkToByte(vtkObject.getVOI(_curext, _ori)), mu, sd);
                
                //Segment current slice
                IList<IList<float>> _cur = model.Inference(input_array);
                //Append results to output list
                if (output == null)
                {
                    output = _cur;
                }
                else
                {
                    //Loop over slices
                    foreach (IList<float> item in _cur)
                    {
                        output.Add(item);
                    }
                }
                input_array = null;
                GC.Collect();
            }

            //Return output list
            return output;
        }

        public static vtkImageData inference_to_vtk(IList<IList<float>> input, int[] output_size, int[] extent, int axis)
        {
            int[] orientation = new int[3];
            if (axis == 0)
            {
                orientation = new int[] { 2, 0, 1 };
                extent = new int[] { extent[0], extent[1], extent[4], extent[5], extent[2], extent[3] };
                output_size = new int[] { output_size[0], output_size[2], output_size[1] };
            }
            if (axis == 1)
            {
                orientation = new int[] { 0, 2, 1 };
                extent = new int[] { extent[2], extent[3], extent[4], extent[5], extent[0], extent[1] };
                output_size = new int[] { output_size[1], output_size[2], output_size[0] };
            }
            if (axis == 2)
            {
                orientation = new int[] { 0, 1, 2 };
                extent = new int[] { extent[4], extent[5], extent[1], extent[2], extent[3], extent[4] };
                output_size = new int[] { output_size[2], output_size[0], output_size[1] };
            }

            //Data to byte array
            byte[,,] bytedata = DataTypes.batchToByte(input, output_size, extent);
            vtkImageData output = DataTypes.byteToVTK(bytedata, orientation);
            bytedata = null;
            input = null;
            GC.Collect();
            return output;
        }

        public static void segmentation_pipeline(out List<vtkImageData> outputs, Rendering.renderPipeLine volume, int[] batch_d, int[] extent, int[] axes, string wpath, int bs = 2)
        {
            //Outputs
            outputs = new List<vtkImageData>();
            //Get input dimensions
            int[] dims = volume.getDims();

            // Initialize UNet
            UNet model = new UNet();
            try
            {
                model.Initialize(24, batch_d, wpath, false);
            }
            catch (FileNotFoundException)
            {
                Console.WriteLine("Model not found");

                return;
            }
                        
            for(int k = 0; k< axes.Length; k++)
            {
                outputs.Add(vtkImageData.New());
            }

            //Segment BCI from axis
            for(int k = 0; k<axes.Length; k++)
            {
                //Inference
                IList<IList<float>> result = segment_sample(volume, model, extent, axes[k], bs, (float)108.58790588378906, (float)47.622276306152344);
                //Copy result to vtkImagedata
                vtkImageData _tmp = IO.inference_to_vtk(result, new int[] { dims[1] + 1, dims[3] + 1, dims[5] + 1 }, extent, axes[k]);
                
                outputs.ElementAt(k).DeepCopy(_tmp);
                _tmp.Dispose();                
                result = null;
            }
            model.Dispose();
        }
    }
}