using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Kitware.VTK;

namespace HistoGrading.Components
{
    /// <summary>
    /// Class for rendering vtkImageData
    /// </summary>
    public class Rendering
    {
        //PipeLines

        /// <summary>
        /// Volume rendering pipeline. Contains methods for connecting components
        /// and memory management.
        /// </summary>
        public class volumePipeLine
        {            
            /**/

            //Volume components

            //VTKVolume
            private vtkVolume vol = vtkVolume.New();
            //Mapper
            private vtkFixedPointVolumeRayCastMapper mapper = vtkFixedPointVolumeRayCastMapper.New();            
            //private vtkSmartVolumeMapper mapper = vtkSmartVolumeMapper.New();

            //Colortransfer function for gray values
            private vtkColorTransferFunction ctf = vtkColorTransferFunction.New();
            //Picewise function for opacity
            private vtkPiecewiseFunction spwf = vtkPiecewiseFunction.New();

            //Mask components, same as above
            private vtkVolume maskvol = vtkVolume.New();
            private vtkFixedPointVolumeRayCastMapper maskmapper = vtkFixedPointVolumeRayCastMapper.New();
            //private vtkSmartVolumeMapper maskmapper = vtkSmartVolumeMapper.New();
            private vtkColorTransferFunction maskctf = vtkColorTransferFunction.New();
            private vtkPiecewiseFunction maskspwf = vtkPiecewiseFunction.New();            

            //Renderer
            public vtkRenderer renderer = vtkRenderer.New();

            //Method for initializing components
            public void Initialize()
            {                
                //Initialize new volume components
                vol = vtkVolume.New();
                //mapper = vtkSmartVolumeMapper.New();
                mapper = vtkFixedPointVolumeRayCastMapper.New();
                ctf = vtkColorTransferFunction.New();
                spwf = vtkPiecewiseFunction.New();
                renderer = vtkRenderer.New();

            }

            //Method for disposing components, useful for memory management
            public void Dispose()
            {
                //Dispose volume components
                vol.Dispose();
                mapper.Dispose();
                ctf.Dispose();
                spwf.Dispose();
                renderer.Dispose();
            }

            //Method for initializing mask components
            public void InitializeMask()
            {
                //Initialize mask
                maskvol = vtkVolume.New();
                //maskmapper = vtkSmartVolumeMapper.New();
                maskmapper = vtkFixedPointVolumeRayCastMapper.New();
                maskctf = vtkColorTransferFunction.New();
                maskspwf = vtkPiecewiseFunction.New();
            }

            //Method for disposing mask components, useful for memory management
            public void DisposeMask()
            {
                //Dispose mask components
                maskvol.Dispose();
                maskmapper.Dispose();
                maskctf.Dispose();
                maskspwf.Dispose();
            }

            //Method for updating volume color
            public void setColor(int cmin, int cmax)
            {
                /*Takes gray value range as input arguments*/
                //Clear ctf
                ctf.Dispose();
                ctf = vtkColorTransferFunction.New();
                //New range for gray values
                ctf.AddRGBPoint(cmin, 0, 0, 0);
                ctf.AddRGBPoint(cmax, 0.8, 0.8, 0.8);
                //Update volume color
                vol.GetProperty().SetColor(ctf);
                vol.Update();
            }

            /// <summary>
            /// Method for connecting volume rendering components.
            /// </summary>
            /// <param name="input">Volume data input.</param>
            /// <param name="inputRenderer">Renderer object.</param>
            public void connectComponents(vtkImageData input, vtkRenderer inputRenderer, int cmin, int cmax)
            {
                /*Arguments: volumetric data and renderer*/

                //Set renderer
                renderer = inputRenderer;
                //Mapper
                mapper.SetInput(input);
                mapper.Update();
                //Color
                ctf.AddRGBPoint(cmin, 0.0, 0.0, 0.0);
                ctf.AddRGBPoint(cmax, 0.8, 0.8, 0.8);
                //Opacity, background in microCT data is < 70
                spwf.AddPoint(0, 0);
                spwf.AddPoint(70, 0.0);
                spwf.AddPoint(80, 0.6);
                spwf.AddPoint(150, 0.8);
                spwf.AddPoint(255, 0.85);
                //Volume parameters
                vol.GetProperty().SetColor(ctf);
                vol.GetProperty().SetScalarOpacity(spwf);
                vol.SetMapper(mapper);
                vol.Update();
                //Renderer back ground
                renderer.SetBackground(0, 0, 0);
                renderer.AddVolume(vol);                
                //Set Camera
                renderer.GetActiveCamera().SetPosition(0.5, 1, 0);
                renderer.GetActiveCamera().SetFocalPoint(0, 0, 0);
                renderer.GetActiveCamera().SetViewUp(0, 0, 1);
                renderer.ResetCamera();
            }

            /// <summary>
            /// Method for connecting mask components.
            /// </summary>
            /// <param name="mask"></param>
            public void connectMask(vtkImageData mask)
            {
                /*Takes bone mask data as input*/
                //Mapper
                maskmapper.SetInput(mask);
                maskmapper.Update();
                //Color
                maskctf.AddRGBPoint(0, 0, 0, 0);
                maskctf.AddRGBPoint(255, 0.9, 0, 0);
                //Opacity, background in microCT data is < 70
                maskspwf.AddPoint(0, 0);
                maskspwf.AddPoint(70, 0.0);
                maskspwf.AddPoint(80, 0.6);
                maskspwf.AddPoint(150, 0.8);
                maskspwf.AddPoint(255, 0.85);                
                //
                //Volume parameters
                maskvol.GetProperty().SetColor(maskctf);
                maskvol.GetProperty().SetScalarOpacity(maskspwf);
                maskvol.SetMapper(maskmapper);
                maskvol.Update();
                //Renderer back ground
                renderer.AddVolume(maskvol);
            }
        }

        /// <summary>
        /// Pipeline for image rendering. Contains methods for connecting components
        /// and memory management.
        /// </summary>
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

            /// <summary>
            /// Renderer object.
            /// </summary>
            public vtkRenderer renderer = vtkRenderer.New();

            /// <summary>
            /// Initialize components, similar as in volume pipeline
            /// </summary>
            public void Initialize()
            {
                actor = vtkImageActor.New();
                colorTable = vtkLookupTable.New();
                colorMapper = vtkImageMapToColors.New();
            }

            /// <summary>
            /// Dispose, memory management
            /// </summary>
            public void Dispose()
            {
                actor.Dispose();
                colorTable.Dispose();
                colorMapper.Dispose();
            }

            /// <summary>
            /// Initialize mask components
            /// </summary>
            public void InitializeMask()
            {
                maskactor = vtkImageActor.New();
                maskcolorTable = vtkLookupTable.New();
                maskcolorMapper = vtkImageMapToColors.New();
            }

            /// <summary>
            /// Dispose mask components, memory management
            /// </summary>
            public void DisposeMask()
            {
                maskactor.Dispose();
                maskcolorTable.Dispose();
                maskcolorMapper.Dispose();
            }

            /// <summary>
            /// Method for setting color table.
            /// Creates lookup table for grayvalues.
            /// </summary>
            /// <param name="cmin">Lower limit for color table.</param>
            /// <param name="cmax">Upper limit for color table.</param>
            public void setGrayLevel(int cmin, int cmax)
            {
                //Set lookup table range
                colorTable.SetTableRange(cmin, cmax);
                //Loop over range and add points to the table
                for (int cvalue = 0; cvalue <= 255; cvalue++)
                {
                    //Current int value / max value
                    double val = (double)cvalue / (double)cmax;
                    //Values below maximum are set to appropriate value
                    if (val < 1.0 && cvalue >= cmin)
                    {
                        colorTable.SetTableValue(cvalue, val, val, val, 1);
                    }
                    if (val < 1.0 && cvalue < cmin)
                    {
                        colorTable.SetTableValue(cvalue, 0, 0, 0, 1);
                    }
                    //Values over maximum are set to 1
                    if (val >= 1 && cvalue >= cmin)
                    {
                        colorTable.SetTableValue(cvalue, 1.0, 1.0, 1.0, 1);
                    }
                }
                //Build the table
                colorTable.Build();
                //Attach to color mapper
                colorMapper.SetLookupTable(colorTable);
            }

            /// <summary>
            /// Method for setting mask color table.
            /// Creates lookup table for grayvalues.
            /// </summary>
            /// <param name="cmin">Lower limit for color table.</param>
            /// <param name="cmax">Upper limit for color table.</param>
            public void setMaskGrayLevel(int cmin, int cmax)
            {
                //Set lookup table range
                maskcolorTable.SetTableRange(cmin, cmax);
                //Loop over range and add points to the table
                for (int cvalue = 0; cvalue <= 255; cvalue++)
                {
                    //Current int value / max value
                    double val = (double)cvalue / (double)cmax;
                    //Values below maximum are set to appropriate value
                    if (val < 1.0 && cvalue > cmin)
                    {
                        maskcolorTable.SetTableValue(cvalue, val, 0, 0, 0.9);
                    }
                    if (val < 1.0 && cvalue <= cmin)
                    {
                        maskcolorTable.SetTableValue(cvalue, 0, 0, 0, 0);
                    }
                    //Values over maximum are set to 1
                    if (val >= 1 && cvalue > cmin)
                    {
                        maskcolorTable.SetTableValue(cvalue, 1.0, 0, 0, 0.9);
                    }
                }
                //Build the table
                maskcolorTable.Build();
                //Attach to color mapper
                maskcolorMapper.SetLookupTable(maskcolorTable);

                /*
                //###//
                //Create lookup table
                maskcolorTable.SetTableRange(cmin, cmax);
                //Min value is set to 0, max value is set 1 => binary table
                maskcolorTable.SetTableValue(cmin, 0.0, 0.0, 0.0, 0.0);
                maskcolorTable.SetTableValue(cmax, 1.0, 0.0, 0.0, 1.0);
                //Build table
                maskcolorTable.Build();
                //Attach to color mapper
                maskcolorMapper.SetLookupTable(maskcolorTable);
                */
            }

            //Method for connecting image components
            public void connectComponents(vtkImageData I, vtkRenderer inputRenderer, int cmin, int cmax)
            {
                /*Arguments: input image, renderer, color range*/

                //Set renderer
                renderer = inputRenderer;
                //Set color
                setGrayLevel(cmin, cmax);
                colorMapper.SetInput(I);
                colorMapper.Update();
                //Set mapper
                actor.SetInput(colorMapper.GetOutput());
                //Connect to renderer
                renderer.AddActor(actor);
                renderer.SetBackground(0.0, 0.0, 0.0);
            }

            /// <summary>
            /// Method for connecting mask components
            /// </summary>
            /// <param name="mask">Mask image data.</param>
            public void connectMask(vtkImageData mask, int cmin, int cmax)
            {
                //Set mask color and mapper
                setMaskGrayLevel(cmin, cmax);
                maskcolorMapper.SetInput(mask);
                maskcolorMapper.Update();
                //Connect mapper
                maskactor.SetInput(maskcolorMapper.GetOutput());
                //Connect to renderer
                renderer.AddActor(maskactor);
            }
        }

        /// <summary>
        /// Class for rendering images and 3D volumes. Volume and image pipelines defined above are called,
        /// depending on the input.
        /// </summary>
        public class renderPipeLine
        {
            //Declarations

            //Renderwindow
            vtkRenderWindow renWin;
            //Renderer
            vtkRenderer renderer = vtkRenderer.New();
            //Volume/image data
            /// <summary>
            /// Original loaded image data as vtkImageData object.
            /// </summary>
            public vtkImageData idata = vtkImageData.New();
            vtkImageData imask = vtkImageData.New();
            //Rendering pipelines
            volumePipeLine volPipe = new volumePipeLine();
            imagePipeLine imPipe = new imagePipeLine();

            //Current slice and gray values
            int[] sliceN = new int[3] { 0, 0, 0 };
            int curAx = 0;
            int[] gray = new int[2] { 0, 255 };

            //Methods for communicating with GUI

            /// <summary>
            /// Connect renderwindow.
            /// </summary>
            /// <param name="input"></param>
            public void connectWindow(vtkRenderWindow input)
            {
                renWin = input;
            }

            /// <summary>
            /// Connect input volume.
            /// </summary>
            /// <param name="input">Data input to be connected.</param>
            public void connectData(string input)
            {
                idata = Functions.loadVTK(input,1);
            }

            /// <summary>
            /// Connect input volume.
            /// </summary>
            /// <param name="input">Data input to be connected.</param>
            public void connectDataFromMemory(vtkImageData input)
            {
                idata = input;
            }

            /// <summary>
            /// Connect bone mask.
            /// </summary>
            /// <param name="input">Bone mask input to be connected.</param>
            public void connectMask(string input)

            {
                imask = Functions.loadVTK(input,1);
                //Set graylevel
                vtkImageMathematics math = vtkImageMathematics.New();
                math.SetInput1(idata);
                math.SetInput2(imask);
                math.SetOperationToMultiply();
                math.Update();
                imask = math.GetOutput();

            }

            /// <summary>
            /// Connect bone mask from memory.
            /// </summary>
            /// <param name="input_mask">Bone mask input to be connected.</param>
            public void connectMaskFromData(vtkImageData input_mask)
            {
                /*
                vtkImageMathematics math = vtkImageMathematics.New();
                math.SetInput1(idata);                
                math.SetInput2(input_mask);
                math.SetOperationToMultiply();
                math.SetNumberOfThreads(24);
                math.Update();
                imask = math.GetOutput();
                */
                imask = input_mask;
            }

            /// <summary>
            /// Update slice.
            /// </summary>
            /// <param name="inpSlice">Slice number.</param>
            /// <param name="inpAx">Current axis.</param>
            /// <param name="inpGray">Gray values.</param>
            public void updateCurrent(int[] inpSlice, int inpAx, int[] inpGray)
            {
                sliceN = inpSlice;
                curAx = inpAx;
                gray = inpGray;
            }

            /// <summary>
            /// 3D volume rendering.
            /// </summary>
            public void renderVolume()
            {
                vtkFileOutputWindow fow = vtkFileOutputWindow.New();
                fow.SetFileName("c:\\users\\Tuomas Frondelius\\Desktop\\errors.txt");
                vtkOutputWindow.SetInstance(fow);
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
                volPipe.connectComponents(idata, renderer, gray[0], gray[1]);
                //Connect renderer to render window
                renWin.AddRenderer(renderer);

                //Render volume
                renWin.Render();

            }

            /// <summary>
            /// 2D Image rendering.
            /// </summary>
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

            /// <summary>
            /// Render 3D bone mask, follows similar pipeline as regular volume rendering.
            /// Doesn't require new renderers to be initialized/connected
            /// </summary>
            public void renderVolumeMask()
            {
                volPipe.DisposeMask();
                volPipe.InitializeMask();

                //Initialize new volume rendering and connect components
                volPipe.connectMask(imask);
                //Render volume
                renWin.Render();
            }

            /// <summary>
            /// Render image mask, see <seealso cref="renderVolumeMask()"/>.
            /// </summary>
            public void renderImageMask()
            {
                /*Connect 2D mask to image rendering pipeline*/
                imPipe.DisposeMask();
                imPipe.InitializeMask();

                //Get mask slice
                vtkImageData maskSlice = Functions.volumeSlicer(imask, sliceN, curAx);
                imPipe.connectMask(maskSlice,gray[0],gray[1]);

                //Render image
                renWin.Render();
            }

            /// <summary>
            /// Set color for volume.
            /// </summary>
            public void setVolumeColor()
            {
                volPipe.setColor(gray[0], gray[1]);
                renWin.Render();
            }

            /// <summary>
            /// Reset camera.
            /// </summary>
            public void resetCamera()
            {
                renderer.GetActiveCamera().SetPosition(0.5, 1, 0);
                renderer.GetActiveCamera().SetFocalPoint(0, 0, 0);
                renderer.GetActiveCamera().SetViewUp(0, 0, 1);
                renderer.ResetCamera();
                renWin.Render();
            }

            /// <summary>
            /// Remove bone mask
            /// </summary>
            public void removeMask()
            {
                imask.Dispose();
                imask = vtkImageData.New();
            }

            /// <summary>
            /// Get data dimensions
            /// </summary>
            /// <returns>Data dimensions.</returns>
            public int[] getDims()
            {
                int[] dims = idata.GetExtent();
                return dims;
            }

            /// <summary>
            /// Get VOI from the data
            /// </summary>
            /// <returns> VOI</returns>
            public vtkImageData getVOI(int[] extent = null, int[] orientation = null)
            {
                //Empty output data
                vtkImageData voi;

                //If no VOI is specified, full data is returned
                if (extent == null)
                {
                    voi = idata;
                }
                else
                {
                    //Extract VOI
                    vtkExtractVOI extractor = vtkExtractVOI.New();
                    extractor.SetInput(idata);
                    extractor.SetVOI(extent[0], extent[1], extent[2], extent[3], extent[4], extent[5]);
                    extractor.Update();
                    voi = extractor.GetOutput();
                }
                //If order of the axes is specified, the return array is permuted
                if (orientation != null)
                {
                    vtkImagePermute permuter = vtkImagePermute.New();
                    permuter.SetInput(voi);
                    permuter.SetFilteredAxes(orientation[0], orientation[1], orientation[2]);
                    permuter.Update();
                    voi = permuter.GetOutput();
                }
                return voi;
            }

            public void center_crop(int size = 400)
            {                
                vtkImageData tmp = Processing.center_crop(idata,size);
                idata.Dispose();
                idata = vtkImageData.New();
                idata.DeepCopy(tmp);
                tmp.Dispose();
            }
        }

        /// <summary>
        /// Render input data to new window.
        /// </summary>
        /// <param name="inputData">vtkImageData to be rendered.</param>
        public static void RenderToNewWindow(vtkImageData inputData)
        {
            // Render cropped volume
            var renwin = vtkRenderWindow.New();
            var vol = new renderPipeLine();
            vol.connectWindow(renwin);
            vol.connectDataFromMemory(inputData);
            vol.renderVolume();
        }
    }
}
