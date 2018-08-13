using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Kitware.VTK;

namespace HistoGrading.Components
{
    class Rendering
    {
        //PipeLines
        private class volumePipeLine
        {
            /*Volume rendering pipeline. Contains methods for connecting components
             *and memory management.*/

            //Volume components

            //VTKVolume
            private vtkVolume vol = vtkVolume.New();
            //Mapper
            private vtkFixedPointVolumeRayCastMapper mapper = vtkFixedPointVolumeRayCastMapper.New();
            //Colortransfer function for gray values
            private vtkColorTransferFunction ctf = vtkColorTransferFunction.New();
            //Picewise function for opacity
            private vtkPiecewiseFunction spwf = vtkPiecewiseFunction.New();

            //Mask components, same as above
            private vtkVolume maskvol = vtkVolume.New();
            private vtkFixedPointVolumeRayCastMapper maskmapper = vtkFixedPointVolumeRayCastMapper.New();
            private vtkColorTransferFunction maskctf = vtkColorTransferFunction.New();
            private vtkPiecewiseFunction maskspwf = vtkPiecewiseFunction.New();

            //Renderer
            public vtkRenderer renderer = vtkRenderer.New();

            //Method for initializing components
            public void Initialize()
            {
                //Initialize new volume components
                this.vol = vtkVolume.New();
                this.mapper = vtkFixedPointVolumeRayCastMapper.New();
                this.ctf = vtkColorTransferFunction.New();
                this.spwf = vtkPiecewiseFunction.New();
                this.renderer = vtkRenderer.New();

                //Initialize new mask components
                this.maskvol = vtkVolume.New();
                this.maskmapper = vtkFixedPointVolumeRayCastMapper.New();
                this.maskctf = vtkColorTransferFunction.New();
                this.maskspwf = vtkPiecewiseFunction.New();
            }

            //Method for disposing components, useful for memory management
            public void Dispose()
            {
                //Dispose volume components
                this.vol.Dispose();
                this.mapper.Dispose();
                this.ctf.Dispose();
                this.spwf.Dispose();
                this.renderer.Dispose();
            }

            //Method for initializing mask components
            public void InitializeMask()
            {
                //Initialize mask
                this.maskvol = vtkVolume.New();
                this.maskmapper = vtkFixedPointVolumeRayCastMapper.New();
                this.maskctf = vtkColorTransferFunction.New();
                this.maskspwf = vtkPiecewiseFunction.New();
            }

            //Method for disposing mask components, useful for memory management
            public void DisposeMask()
            {
                //Dispose mask components
                this.maskvol.Dispose();
                this.maskmapper.Dispose();
                this.maskctf.Dispose();
                this.maskspwf.Dispose();
            }

            //Method for updating volume color
            public void setColor(int cmin, int cmax)
            {
                /*Takes gray value range as input arguments*/
                //Clear ctf
                this.ctf.Dispose();
                this.ctf = vtkColorTransferFunction.New();
                //New range for gray values
                this.ctf.AddRGBPoint(cmin, 0, 0, 0);
                this.ctf.AddRGBPoint(cmax, 0.8, 0.8, 0.8);
                //Update volume color
                this.vol.GetProperty().SetColor(ctf);
                this.vol.Update();
            }

            //Method for connecting volume rendering components
            public void connectComponents(vtkImageData input, vtkRenderer inputRenderer)
            {
                /*Arguments: volumetric data and renderer*/

                //Set renderer
                this.renderer = inputRenderer;
                //Mapper
                this.mapper.SetInput(input);
                this.mapper.Update();
                //Color
                this.ctf.AddRGBPoint(0, 0.0, 0.0, 0.0);
                this.ctf.AddRGBPoint(255, 0.8, 0.8, 0.8);
                //Opacity, background in microCT data is < 70
                this.spwf.AddPoint(0, 0);
                this.spwf.AddPoint(70, 0.0);
                this.spwf.AddPoint(80, 0.7);
                this.spwf.AddPoint(150, 0.9);
                this.spwf.AddPoint(255, 0.95);
                //Volume parameters
                this.vol.GetProperty().SetColor(ctf);
                this.vol.GetProperty().SetScalarOpacity(spwf);
                this.vol.SetMapper(this.mapper);
                this.vol.Update();
                //Renderer back ground
                renderer.SetBackground(0, 0, 0);
                renderer.AddVolume(vol);
                //Set Camera
                renderer.GetActiveCamera().SetPosition(0.5, 1, 0);
                renderer.GetActiveCamera().SetFocalPoint(0, 0, 0);
                renderer.GetActiveCamera().SetViewUp(0, 0, 1);
                renderer.ResetCamera();
            }

            //Method for connecting mask components
            public void connectMask(vtkImageData mask)
            {
                /*Takes bone mask data as input*/
                //Mapper
                this.maskmapper.SetInput(mask);
                this.maskmapper.Update();
                //Color
                this.maskctf.AddRGBPoint(0, 0, 0, 0);
                this.maskctf.AddRGBPoint(255, 0.6, 0, 0);
                //Opacity
                this.maskspwf.AddPoint(0, 0);
                this.maskspwf.AddPoint(180, 0);
                this.maskspwf.AddPoint(181, 0.1);
                this.maskspwf.AddPoint(255, 0.15);
                //
                //Volume parameters
                this.maskvol.GetProperty().SetColor(maskctf);
                this.maskvol.GetProperty().SetScalarOpacity(maskspwf);
                this.maskvol.SetMapper(this.maskmapper);
                this.maskvol.Update();
                //Renderer back ground
                renderer.AddVolume(maskvol);
            }
        }

        public class imagePipeLine
        {
            //Image variables
            private vtkImageActor actor = vtkImageActor.New();
            private vtkLookupTable colorTable = vtkLookupTable.New();
            private vtkImageMapToColors colorMapper = vtkImageMapToColors.New();

            //Bone mask variables
            private vtkImageActor maskactor = vtkImageActor.New();
            private vtkLookupTable maskcolorTable = vtkLookupTable.New();
            private vtkImageMapToColors maskcolorMapper = vtkImageMapToColors.New();

            //Renderer
            public vtkRenderer renderer = vtkRenderer.New();

            //Initialize components, similar as in volume pipeline
            public void Initialize()
            {
                this.actor = vtkImageActor.New();
                this.colorTable = vtkLookupTable.New();
                this.colorMapper = vtkImageMapToColors.New();
            }

            //Dispose, memory management
            public void Dispose()
            {
                this.actor.Dispose();
                this.colorTable.Dispose();
                this.colorMapper.Dispose();
            }

            //Initialize mask components
            public void InitializeMask()
            {
                this.maskactor = vtkImageActor.New();
                this.maskcolorTable = vtkLookupTable.New();
                this.maskcolorMapper = vtkImageMapToColors.New();
            }

            //Dispose mask components, memory management
            public void DisposeMask()
            {
                this.maskactor.Dispose();
                this.maskcolorTable.Dispose();
                this.maskcolorMapper.Dispose();
            }

            //Method for setting color table
            public void setGrayLevel(int cmin, int cmax)
            {
                /*Creates lookup table for grayvalues*/
                //Set lookup table range
                this.colorTable.SetTableRange(cmin, cmax);
                //Loop over range and add points to the table
                for (int cvalue = 0; cvalue <= 255; cvalue++)
                {
                    //Current int value / max value
                    double val = (double)cvalue / (double)cmax;
                    //Values below maximum are set to appropriate value
                    if (val < 1.0 && cvalue >= cmin)
                    {
                        this.colorTable.SetTableValue(cvalue, val, val, val, 1);
                    }
                    if (val < 1.0 && cvalue < cmin)
                    {
                        this.colorTable.SetTableValue(cvalue, 0, 0, 0, 1);
                    }
                    //Values over maximum are set to 1
                    if (val >= 1 && cvalue >= cmin)
                    {
                        this.colorTable.SetTableValue(cvalue, 1.0, 1.0, 1.0, 1);
                    }
                }
                //Build the table
                this.colorTable.Build();
                //Attach to color mapper
                this.colorMapper.SetLookupTable(this.colorTable);
            }

            //Method for setting gray mask color table
            public void setMaskGrayLevel(int cmin, int cmax)
            {
                //Create lookup table
                this.maskcolorTable.SetTableRange(cmin, cmax);
                //Min value is set to 0, max value is set 1 => binary table
                this.maskcolorTable.SetTableValue(cmin, 0.0, 0.0, 0.0, 0.0);
                this.maskcolorTable.SetTableValue(cmax, 1.0, 0.0, 0.0, 1.0);
                //Build table
                this.maskcolorTable.Build();
                //Attach to color mapper
                this.maskcolorMapper.SetLookupTable(this.maskcolorTable);
            }

            //Method for connecting image components
            public void connectComponents(vtkImageData I, vtkRenderer inputRenderer, int cmin, int cmax)
            {
                /*Arguments: input image, renderer, color range*/

                //Set renderer
                this.renderer = inputRenderer;
                //Set color
                setGrayLevel(cmin, cmax);
                this.colorMapper.SetInput(I);
                this.colorMapper.Update();
                //Set mapper
                this.actor.SetInput(colorMapper.GetOutput());
                //Connect to renderer
                renderer.AddActor(this.actor);
                renderer.SetBackground(0.0, 0.0, 0.0);
            }

            //Method for connecting mask components
            public void connectMask(vtkImageData mask)
            {
                //Set mask color and mapper
                setMaskGrayLevel(0, 255);
                this.maskcolorMapper.SetInput(mask);
                this.maskcolorMapper.Update();
                //Connect mapper
                this.maskactor.SetInput(this.maskcolorMapper.GetOutput());
                //Connect to renderer
                this.renderer.AddActor(this.maskactor);
            }
        }

        public class renderPipeLine
        {
            /*Class for rendering images and 3D volumes. Volume and image pipelines defined above are called¨,
             depending on the input*/

            //Declarations

            //Renderwindow
            vtkRenderWindow renWin;
            //Renderer
            vtkRenderer renderer = vtkRenderer.New();
            //Volume/image data
            vtkImageData idata = vtkImageData.New();
            vtkImageData imask = vtkImageData.New();
            //Rendering pipelines
            volumePipeLine volPipe = new volumePipeLine();
            imagePipeLine imPipe = new imagePipeLine();

            //Current slice and gray values
            int[] sliceN = new int[3] { 0, 0, 0 };
            int curAx = 0;
            int[] gray = new int[2] { 0, 255 };

            //Methods for communicating with GUI

            //Connect renderwindow
            public void connectWindow(vtkRenderWindow input)
            {
                renWin = input;
            }

            //Connect input volume
            public void connectData(string input)
            {
                idata = Functions.loadVTK(input);
            }

            //Connect bone mask
            public void connectMask(string input)
            {
                imask = Functions.loadVTK(input);
            }

            //Update slice
            public void updateCurrent(int[] inpSlice, int inpAx, int[] inpGray)
            {
                sliceN = inpSlice;
                curAx = inpAx;
                gray = inpGray;
            }

            //Volume rendering
            public void renderVolume()
            {
                //Detach first renderer from render window. Prevents multiple images from being
                //rendered on top of each other, and helps with memory management.
                renWin.RemoveRenderer(renWin.GetRenderers().GetFirstRenderer());

                //Initialize renderer, dispose existing renderer and connect new renderer
                renderer.Dispose();
                renderer = vtkRenderer.New();

                //Initialize new volume rendering pipeline and connect components
                //Disposes existing pipeline and initializes new pipeline
                volPipe.Dispose();
                volPipe.Initialize();
                //Connect input data and renderer to rendering pipeline
                volPipe.connectComponents(idata, renderer);
                //Connect renderer to render window
                renWin.AddRenderer(renderer);

                //Render volume
                renWin.Render();

            }
            //Image rendering
            public void renderImage()
            {
                //Detach first renderer from render window. Prevents multiple images from being
                //rendered on top of each other, and helps with memory management.
                renWin.RemoveRenderer(renWin.GetRenderers().GetFirstRenderer());

                //Initialize new image rendering pipeline and connect components
                //Disposes existing pipeline and initializes new pipeline
                vtkImageData slice = Functions.volumeSlicer(idata, sliceN, curAx);

                //Initialize objects
                imPipe.Initialize();

                //Initialize renderer
                renderer.Dispose();
                renderer = vtkRenderer.New();

                //Connect components to rendering pipeline
                imPipe.connectComponents(slice, renderer, gray[0], gray[1]);
                //Connect renderer to render window
                renWin.AddRenderer(renderer);

                //Render image
                renWin.Render();
            }

            //Volume mask rendering
            public void renderVolumeMask()
            {
                /*Render 3D bone mask, follows similar pipeline as regular volume rendering.
                  Doesn't require new renderers to be initialized/connected*/
                volPipe.DisposeMask();
                volPipe.InitializeMask();

                //Initialize new volume rendering and connect components
                volPipe.connectMask(imask);
                //Render volume
                renWin.Render();
            }
            //Image mask rendering
            public void renderImageMask()
            {
                /*Connect 2D mask to image rendering pipeline*/
                imPipe.DisposeMask();
                imPipe.InitializeMask();

                //Get mask slice
                vtkImageData maskSlice = Functions.volumeSlicer(imask, sliceN, curAx);
                imPipe.connectMask(maskSlice);

                //Render image
                renWin.Render();
            }

            //Set color
            public void setVolumeColor()
            {
                volPipe.setColor(gray[0], gray[1]);
                renWin.Render();
            }

            //Reset camera
            public void resetCamera()
            {
                renderer.GetActiveCamera().SetPosition(0.5, 1, 0);
                renderer.GetActiveCamera().SetFocalPoint(0, 0, 0);
                renderer.GetActiveCamera().SetViewUp(0, 0, 1);
                renderer.ResetCamera();
                renWin.Render();
            }

            //Remove bone mask
            public void removeMask()
            {
                imask.Dispose();
                imask = vtkImageData.New();
            }

            //Get data dimensions
            public int[] getDims()
            {
                int[] dims = idata.GetExtent();
                return dims;
            }

        }
    }
}
