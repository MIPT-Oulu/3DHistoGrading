using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Kitware.VTK;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace HistoGrading.Components
{
    /// <summary>
    /// Utility functions that are used in the software.
    /// </summary>
    public class Functions
    {
        /// <summary>
        /// Load image files into vtkImageData.
        /// </summary>
        /// <param name="path">Path to images.</param>
        /// <param name="extension">Image extension.</param>
        /// <returns></returns>
        public static vtkImageData VTKLoader(string path, string extension)
        {   
            /*DEPRECATED!!*/
            //Output
            vtkImageData data = vtkImageData.New();
            //Get files from path
            DirectoryInfo d = new DirectoryInfo(@path);
            FileInfo[] files = d.GetFiles();

            vtkStringArray allfiles = vtkStringArray.New();
            //Iterate over files and read image data
            foreach (FileInfo file in files)
            {
                //Fullfile
                string fullfile = Path.Combine(path, file.Name);
                allfiles.InsertNextValue(fullfile);
            }
            if (extension == ".png")
            {
                vtkPNGReader reader = vtkPNGReader.New();
                reader.SetFileNames(allfiles);
                reader.Update();
                data = reader.GetOutput();
                reader.Dispose();
            }
            if (extension == ".jpg")
            {
                vtkJPEGReader reader = vtkJPEGReader.New();
                reader.SetFileNames(allfiles);
                reader.Update();
                data = reader.GetOutput();
                reader.Dispose();
            }
            if (extension == ".bmp")
            {
                vtkBMPReader reader = vtkBMPReader.New();
                reader.SetFileNames(allfiles);
                reader.Update();
                data = reader.GetOutput();
                reader.Dispose();
            }
            data.SetScalarTypeToUnsignedChar();
            data.Update();
            return data;
        }

        /// <summary>
        /// Gets a 2D slice from the 3D data.
        /// </summary>
        /// <param name="volume">Full 3D data.</param>
        /// <param name="sliceN">Number of slice to be selected.</param>
        /// <param name="axis">Axis on which selection will be made.</param>
        /// <returns></returns>
        public static vtkImageData volumeSlicer(vtkImageData volume, int[] sliceN, int axis)
        {
            //Initialize VOI extractor and permuter.
            //Permuter will correct the orientation of the output image
            vtkExtractVOI slicer = vtkExtractVOI.New();
            vtkImagePermute permuter = vtkImagePermute.New();
            //Connect data to slicer
            slicer.SetInput(volume);
            slicer.Update();

            //Get data dimensions
            int[] dims = slicer.GetOutput().GetExtent();

            //Get slice

            //Coronal plane
            if (axis == 0)
            {
                //Set VOI
                slicer.SetVOI(sliceN[0], sliceN[0], dims[2], dims[3], dims[4], dims[5]);
                slicer.Update();
                //Permute image (not necessary here)
                permuter.SetInputConnection(slicer.GetOutputPort());
                permuter.SetFilteredAxes(1, 2, 0);
                permuter.Update();
            }
            //Sagittal plane
            if (axis == 1)
            {
                //Set VOI
                slicer.SetVOI(dims[0], dims[1], sliceN[1], sliceN[1], dims[4], dims[5]);
                slicer.Update();
                //Permute image
                permuter.SetInputConnection(slicer.GetOutputPort());
                permuter.SetFilteredAxes(0, 2, 1);
                permuter.Update();
            }
            //Transverse plane, XY
            if (axis == 2)
            {
                //Set VOI
                slicer.SetVOI(dims[0], dims[1], dims[2], dims[3], sliceN[2], sliceN[2]);
                slicer.Update();
                //Permute image
                permuter.SetInputConnection(slicer.GetOutputPort());
                permuter.SetFilteredAxes(0, 1, 2);
                permuter.Update();
            }
            //slicer.Update();

            vtkImageData output = vtkImageData.New();
            output.DeepCopy(permuter.GetOutput());
            permuter.Dispose();
            slicer.Dispose();
            //Return copy of the slice
            return output;
        }

        /// <summary>
        /// Create scalar copy of vtkImageData.
        /// </summary>
        /// <param name="data">Input data.</param>
        /// <returns>Copied data.</returns>
        public static vtkImageData scalarCopy(vtkImageData data)
        {
            /*DEPRECATED!!*/
            //Get data extent
            int[] dims = data.GetExtent();
            vtkImageData newdata = vtkImageData.New();
            newdata.SetExtent(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
            for (int h = dims[0]; h <= dims[1]; h++)
            {
                for (int w = dims[2]; w <= dims[3]; w++)
                {
                    for (int d = dims[4]; d <= dims[5]; d++)
                    {
                        double scalar = data.GetScalarComponentAsDouble(h, w, d, 0);
                        newdata.SetScalarComponentFromDouble(h, w, d, 0, scalar);
                    }

                }
            }
            return newdata;
        }

        /// <summary>
        /// Read all images in folder containing file from given path
        /// Uses ParaLoader class to get the data in vtk format.
        /// </summary>
        /// <param name="path">Directory that includes images to be loaded.</param>
        /// <returns></returns>
        public static vtkImageData loadVTK(string path)
        {
            //Declare loader
            ParaLoader loader = new ParaLoader();
            //Set input path to loader
            loader.setInput(path);
            //Load data
            loader.Load();
            //Extract data to variable
            vtkImageData data = vtkImageData.New();
            data.DeepCopy(loader.GetData());
            loader.Dispose();
            return data;
        }

        /// <summary>
        /// Find all files which correspond to selected file.
        /// When reading multiple files, files must start with the same name, which ends in a digit,
        /// as the selected file and have the same extension.
        /// </summary>
        /// <param name="file">Name for the file to be checked.</param>
        /// <returns></returns>
        public static List<string> getFiles(string file)
        {
            //Get file name and extension
            string fileName = Path.GetFileName(file);
            string extension = Path.GetExtension(file);

            //Get the starting string of the file
            //if the files are numbered, the starting string is recorded and used
            //to find all files with similar names

            //Empty string
            string fstart = "";
            //Length of file name
            int nameL = fileName.Length;
            //Loop from the start of the extension to start of the file name
            //and check for numbers
            for (int k = 5; k < nameL; k++)
            {
                char c = fileName[nameL - k];
                if (Char.IsNumber(c) == false)
                {
                    for (int kk = 0; kk < (nameL - k); kk++)
                    {
                        fstart += fileName[kk];
                    }

                    break;
                }
            }

            //Get directory info
            string path = Path.GetDirectoryName(file);
            DirectoryInfo d = new DirectoryInfo(@path);

            //Get all files
            FileInfo[] allFiles = d.GetFiles();

            //Empty list for found files
            List<string> names = new List<string>();

            //Loop over the files and find files corresponding to the selected files
            foreach (FileInfo curFile in allFiles)
            {
                //Length of current file name
                int L = curFile.Name.Length;
                //Compare to selected file and check if contains numbers/has same extension
                if (curFile.Name.StartsWith(fstart) && Char.IsNumber(curFile.Name[L - 5]) && curFile.Name.EndsWith(extension))
                {
                    names.Add(Path.Combine(path, curFile.Name));
                }
            }

            //No numbered files found, use selected image
            if (names.Count == 0)
            {
                names.Add(fileName);
            }

            return names;
        }

        /// <summary>
        /// Function for correctly mapping the pixel values, copied from CNTK examples.
        /// </summary>
        /// <param name="pixelFormat">Format of the color data.</param>
        /// <param name="stride">Stride length.</param>
        /// <returns></returns>
        public static Func<int, int, int, int> GetPixelMapper(PixelFormat pixelFormat, int stride)
        {
            switch (pixelFormat)
            {
                /*
                case PixelFormat.Format32bppArgb:
                    return (h, w, c) => h * stride + w * 4 + c;  // bytes are B-G-R-A
                case PixelFormat.Format24bppRgb:
                    return (h, w, c) => h * stride + w * 3 + c;  // bytes are B-G-R
                */
                case PixelFormat.Format8bppIndexed:
                default:
                    return (h, w, c) => h * stride + w;
            }
        }
        
        /// <summary>
        /// Also does not work
        /// </summary>
        /// <param name="input"></param>
        /// <param name="path"></param>
        /// <param name="name"></param>
        public static void saveVTKPNG(vtkImageData input, string path, string name)
        {
            //Data dimensions
            int[] dims = input.GetExtent();            
            int d = dims[5] - dims[4] + 1;
            //Array of indices
            int[] inds = new int[] { 0, 0, 0 };

            for (int k = 0; k<d;k++)
            {
                //Set file name
                string dirname = path + "\\" + name;
                if (k == 0) { Directory.CreateDirectory(dirname); }
                string fname = dirname + "\\" + name + "\\" + name + "_" + String.Format("{0}", k).PadLeft(6, '0') + ".png";

                //Get slice
                inds[2] += k;
                vtkImageData tmp = Functions.volumeSlicer(input, inds, 2);

                //New png writer
                vtkPNGWriter writer = vtkPNGWriter.New();
                writer.SetInput(tmp);
                writer.WriteToMemoryOff();
                writer.SetFileDimensionality(2);
                writer.SetFileName(fname);                
                writer.Write();
                writer.Dispose();
            }
        }

        /// <summary>
        /// Does not work
        /// </summary>
        /// <param name="input"></param>
        /// <param name="path"></param>
        /// <param name="name"></param>
        /// <param name="extension"></param>
        public static void saveVTK(vtkImageData input, string path, string name, string extension = "png")
        {
            //Get dimensiosn
            int[] dims = input.GetExtent();
            int h = dims[1] - dims[0] + 1;
            int w = dims[3] - dims[2] + 1;
            int d = dims[5] - dims[4] + 1;
            //Convert to bytedata
            byte[] bytedata = DataTypes.vtkToByte(input);
            //Iterate over z-axis
            for(int kz = 0; kz<d; kz++)
            {         
                //Empty array for slice
                byte[] _tmp = new byte[h * w];
                //Iterate over x and y axes
                Parallel.For(0, w, (int kx) =>
                {
                    //Iterate over x and y axes
                    Parallel.For(0, h, (int ky) =>
                    {
                        int pos = kz * (h * w) + ky * w + kx;
                        int _pos = ky * w + kx;
                        _tmp[_pos] = bytedata[pos];
                    });
                });
                //Convert slice to Mat
                Mat image = new Mat(h, w, MatType.CV_8UC1, _tmp);
                _tmp = null;                
                //Save image
                string dirname = path + "\\" + name;
                if (kz == 0) { Directory.CreateDirectory(dirname); }
                string fname = dirname + "\\" + name + "_" + String.Format("{0}",kz).PadLeft(6, '0') + "." + extension;
                Console.WriteLine(dirname);
                Console.WriteLine(fname);
                Image _image = image.ToBitmap();
                image.Dispose();
                _tmp = null;
                _image.Save(fname,ImageFormat.Png);
                _image.Dispose();                                
            }
        }

        /// <summary>
        /// Prompts a folderbrowserdialog with given description.
        /// </summary>
        /// <param name="description">Description to be displayed on top of the window.</param>
        /// <returns>Selected path or empty string.</returns>
        public static string GetDirectory(string description)
        {
            var dlg = new FolderBrowserDialog { Description = description };
            DialogResult result = dlg.ShowDialog();

            return DirectoryResult(dlg.SelectedPath, result); // Check that path was selected.
        }

        /// <summary>
        /// Prompts a folderbrowserdialog with given description.
        /// </summary>
        /// <param name="description">Description to be displayed on top of the window.</param>
        /// <returns>Selected path or empty string.</returns>
        public static string GetFile(string description)
        {
            var dlg = new OpenFileDialog { Title = description };
            DialogResult result = dlg.ShowDialog();

            return DirectoryResult(dlg.FileName, result); // Check that path was selected.
        }

        /// <summary>
        /// Test to see if a path was selected during folderbrowserdialog.
        /// </summary>
        /// <param name="selectedPath">Path selected by user.</param>
        /// <param name="result">Result of the dialog. E.g. OK or cancel</param>
        /// <returns>Path or empty string.</returns>
        public static string DirectoryResult(string selectedPath, DialogResult result)
        {
            return result == DialogResult.OK ? selectedPath : string.Empty;
        }

        /// <summary>
        /// Crop volume to selected length.
        /// </summary>
        /// <typeparam name="T">Data type can be selected by user.</typeparam>
        /// <param name="volume">Volume to be cropped.</param>
        /// <param name="dims">Cropping dimensions. Format: x1, x2, y1, y2, z1, z2.</param>
        /// <returns>Cropped volume.</returns>
        public static T[,,] Crop3D<T>(T[,,] volume, int[] dims)
        {
            if (dims.Length != 6)
                throw new Exception("Invalid number of dimensions on dims variable. Include 6 dimensions.");
            T[,,] croppedVolume = new T[dims[1] - dims[0] + 1, dims[3] - dims[2] + 1, dims[5] - dims[4] + 1];

            Parallel.For(dims[0], dims[1] + 1, x =>
            {
                Parallel.For(dims[2], dims[3] + 1, y =>
                {
                    Parallel.For(dims[4], dims[5] + 1, z =>
                    {
                        croppedVolume[x - dims[0], y - dims[2], z - dims[4]] = volume[x, y, z];
                    });
                });
            });
            return croppedVolume;
        }

        public static void get_bbox(out int min_x, out int max_x, out int min_y, out int max_y, Mat input, double threshold = 70.0, double max_val = 255.0, int ks=25)
        {
            //Threshold input
            Mat BW = input.Threshold(threshold, max_val, ThresholdTypes.Binary);
            //Generate cirular kernel for closing
            byte[] circle = new byte[ks * ks];
            for(int k = 0; k<ks*ks; k++)
            {
                int y = k / ks;
                int x = k % ks;
                double dy = (double)y - (double)ks / 2;
                double dx = (double)x - (double)ks / 2;

                double val = dy * dy + dx * dx;
                if(val < ks*ks)
                {
                    circle[k] = 255;
                }
            }
            var strct = InputArray.Create(new Mat(ks, ks, MatType.CV_8UC1,circle));
            BW = BW.Dilate(strct);
            BW = BW.Erode(strct);

            var edges = BW.FindContoursAsMat(RetrievalModes.CComp, ContourApproximationModes.ApproxSimple);
            if (edges.Count() > 0)
            {

                Rect bbox = new Rect();
                int curArea = 0;

                foreach (MatOfPoint contour in edges)
                {
                    var brect = contour.BoundingRect();
                    var area = brect.Height * brect.Width;
                    if (area > curArea)
                    {
                        bbox = brect;
                        curArea = area;
                    }
                }

                min_x = bbox.Left;
                max_x = bbox.Right;
                min_y = bbox.Top;
                max_y = bbox.Bottom;
            }
            else
            {
                min_x = 0;
                max_x = 0;
                min_y = 0;
                max_y = 0;
            }
        }

        public static double get_sample_center(vtkImageData slice, double threshold = 70.0, double scale = 1.0)
        {
            //Get slice dimensions, x and y axes swapped in vtk
            int[] dims = slice.GetExtent();            
            int w = dims[1] - dims[0] + 1;
            int h = dims[3] - dims[2] + 1;

            //Convert data to byte array
            byte[] bytedata = DataTypes.vtkToByte(slice);
            /*
            //Set the top 1/3rd to zero
            Parallel.For(0, h*w, (int k) =>
            {
                int y = k / w;
                if(y>2*h/3)
                {                    
                    bytedata[k] = 0;
                }
            });
            */
            Mat image = new Mat(h, w, MatType.CV_8UC1, bytedata);            
            //Resize image
            if(scale != 1.0)
            {
                OpenCvSharp.Size size = new OpenCvSharp.Size(scale * w, scale * h);
                image = image.Resize(size);
            }            
            Mat BW = image.Threshold(threshold, 255.0, ThresholdTypes.Binary);
            Mat inds = BW.FindNonZero();

            using (var window = new Window("BW", WindowMode.AutoSize, BW))
            {
                Cv2.WaitKey();
            }

            var ellip = Cv2.FitEllipse(inds);
            var theta = ellip.Angle;
            if(theta > 90)
            {
                theta -= 180;
            }

            return (double)theta;
            
            /*
            //Get enclosing circle
            Point2f xy; float r;
            inds.MinEnclosingCircle(out xy, out r);

            int[] output = new int[] { (int)(xy.X/scale), (int)(xy.Y / scale) };

            return output;
            */
        }

        public static double get_angle(int[] data, int step = 1, bool radians = false)
        {
            //Compute the mean of the points
            double mu = 0.0;
            double nom = 1e-9;
            double idxmu = 0.0;

            for (int k = 0; k < data.Length; k++)
            {
                mu += data[k];
                nom += 1.0;
                idxmu += k * step;
            }
            mu /= nom;
            idxmu /= nom;

            //Generate points object for fitting
            List<Point2f> points = new List<Point2f>();
            for (int k = 0; k < data.Length; k++)
            {
                points.Add(new Point2f((float)(k - idxmu), (float)(data[k]-mu)));    
            }

            //Fit line
            Line2D line = Cv2.FitLine(points, DistanceTypes.L2, 0, 0.01, 0.01);

            double slope = line.Vy / (line.Vx + 1e-9);

            double theta = Math.Atan(slope);

            if(radians == true)
            {
                return theta;
            }
            else
            {
                theta *= 180.0 / Math.PI;
                return theta;
            }
            
        }

        public static int[,] get_surface_index_from_tiles(double[,,] array, double threshold)
        {
            int h = array.GetLength(0);
            int w = array.GetLength(1);
            int d = array.GetLength(2);

            int[,] idx = new int[h, w];

            Parallel.For(0,h, (int y) =>
            {
                Parallel.For(0, w, (int x) =>
                {
                    for(int k = d-1; k > -1; k-=1)
                    {
                        double val = array[y, x, k];
                        if(val > threshold)
                        {
                            idx[y, x] = k;
                            break;
                        }
                    }
                });
            });

            return idx;
        }

        public static double[] get_tile_angles(int[,] idx, int[] steps)
        {
            //Surface coordinates to points                       

            List<Point2f> pointsx = new List<Point2f>();
            List<Point2f> pointsy = new List<Point2f>();

            for (int y = 0; y < idx.GetLength(0); y++)
            {
                for (int x = 0; x < idx.GetLength(1); x++)
                {
                    pointsx.Add(new Point2f((float)(x * steps[1]), (float)(idx[y, x])));
                    pointsy.Add(new Point2f((float)(y * steps[0]), (float)(idx[y, x])));
                }
            }

            //Surface fit
            Line2D linex = Cv2.FitLine(pointsx, DistanceTypes.L2, 0, 0.01, 0.01);
            Line2D liney = Cv2.FitLine(pointsy, DistanceTypes.L2, 0, 0.01, 0.01);

            //Get angles as degrees
            double thetax = Math.Atan(linex.Vy / (linex.Vx + 1e-9)) * 180.0/Math.PI;
            double thetay = Math.Atan(liney.Vy / (liney.Vx + 1e-9)) * 180.0 / Math.PI;

            return new double[] { thetax, thetay};
        }

        public static vtkImageData vtk_average(vtkImageData input1, vtkImageData input2)
        {
            //Average the input arrays
            vtkImageMathematics math1 = vtkImageMathematics.New();
            math1.SetOperationToMultiplyByK();
            math1.SetConstantK(0.5);
            math1.SetInput1(input1);
            math1.Update();

            vtkImageMathematics math2 = vtkImageMathematics.New();
            math2.SetOperationToMultiplyByK();
            math2.SetConstantK(0.5);
            math2.SetInput1(input2);
            math2.Update();

            vtkImageMathematics math = vtkImageMathematics.New();
            math.SetOperationToAdd();
            math.SetInput1(math1.GetOutput());
            math.SetInput2(math2.GetOutput());
            math.Update();

            return math.GetOutput();
        }

        public static vtkImageData get_surface_voi(vtkImageData input, int n_tiles = 64, double threshold = 70.0, double mult = 0.05)
        {
            //Get data dimensions
            int[] dims = input.GetExtent();

            //Get average tiles
            double[,,] tiles; int[] steps;

            threshold *= mult;

            Processing.average_tiles(out tiles, out steps, input, 16);
            //Get surface indices
            int[,] idx = Functions.get_surface_index_from_tiles(tiles, threshold);

            //Find minimum and maximum index
            int min = 65000; int max = 0;
            for (int k1 = 0; k1 < idx.GetLength(0); k1++)
            {
                for (int k2 = 0; k2 < idx.GetLength(1); k2++)
                {
                    if (idx[k1, k2] < min) { min = idx[k1, k2]; }
                    if (idx[k1, k2] > max) { max = idx[k1, k2]; }
                }
            }

            //Crop VOI around minima and maxima
            int[] outextent = new int[] { dims[0], dims[1], dims[2], dims[3], Math.Max(min - 50, dims[4]), Math.Min(max + 50, dims[5]) };

            vtkExtractVOI cropper = vtkExtractVOI.New();
            cropper.SetInput(input);
            cropper.SetVOI(outextent[0], outextent[1], outextent[2], outextent[3], outextent[4], outextent[5]);
            cropper.Update();

            vtkImageData tmpvtk = cropper.GetOutput();

            
            //Get surface orientation
            double[] angles = Functions.get_tile_angles(idx, steps);
            angles = new double[] { angles[0], -angles[1] };
            int[] axes = new int[] { 0, 1 };
            
            
            //Reorient the surface
            for (int k=0; k < angles.Length; k++)
            {
                tmpvtk = Processing.rotate_sample(tmpvtk, angles[k], axes[k], 0);
            }
            
            
            //Detect surface indices and compute mean and standard deviation images
            double[,] mu; double[,] std; vtkImageData output;
            Processing.get_voi_mu_std(out output, out mu, out std, tmpvtk, 25,80.0);
            
            
            //Invert the reorienting            
            for (int k = 0; k < angles.Length; k++)
            {
                output = Processing.rotate_sample(output, -angles[k], axes[k], 0);
            }
            

            return output;
            

        }

        public static int[,] get_surface(vtkImageData mask, double threshold = 0.0)
        {
            //Get data dimensions
            int[] dims = mask.GetExtent();
            int h = dims[1] - dims[0] + 1;
            int w = dims[3] - dims[2] + 1;
            int d = dims[5] - dims[4] + 1;

            //Get strides
            int strideh = w;
            int stridew = 1;
            int strided = h * w;

            //Convert mask to byte data
            byte[] bytedata = DataTypes.vtkToByte(mask);

            //Empty array for surface indices
            int[,] indices = new int[h, w];
            //Find the mask surface
            Parallel.For(0, h, (int y) =>
            {
                Parallel.For(0, w, (int x) =>
                {
                    for(int z = d-1; z > 0; z-=1)
                    {
                        int pos = z * strided + y * strideh + x * stridew;
                        byte val = bytedata[pos];
                        if((double)val>threshold)
                        {
                            indices[y, x] = z;            
                            break;
                        }
                    }
                });
            });

            //Return indices
            return indices;
        }

        public static vtkImageData get_deep_ac(vtkImageData sample, int[,] BCI, int[,] depth, int offset = 10)
        {
            //vtk to byte
            byte[] bytedata = DataTypes.vtkToByte(sample);

            //Get dims
            int[] dims = sample.GetExtent();
            int h = dims[1] - dims[0] + 1;
            int w = dims[3] - dims[2] + 1;
            int d = dims[5] - dims[4] + 1;

            //Output array
            byte[] output = new byte[bytedata.Length];

            Parallel.For(0, h, (int y) =>
            {
                Parallel.For(0, w, (int x) =>
                {
                    for(int z = BCI[y, x] + depth[y, x]; z > BCI[y,x] + offset; z-=1)
                    {
                        int pos = z * (h * w) + y * w + x;
                        output[pos] = bytedata[pos];
                    }
                });
            });

            bytedata = null;

            //Convert output to vtkImageData
            return DataTypes.byteToVTK1D(output, dims);

        }

        public static vtkImageData get_calcified_cartilage(vtkImageData sample, int[,] BCI, int offset = 10)
        {
            //vtk to byte
            byte[] bytedata = DataTypes.vtkToByte(sample);

            //Get dims
            int[] dims = sample.GetExtent();
            int h = dims[1] - dims[0] + 1;
            int w = dims[3] - dims[2] + 1;
            int d = dims[5] - dims[4] + 1;

            //Output array
            byte[] output = new byte[bytedata.Length];

            Parallel.For(0, h, (int y) =>
            {
                Parallel.For(0, w, (int x) =>
                {
                    for ( int z = BCI[y, x] + offset; z > 0; z-=1)
                    {
                        int pos = z * (h * w) + y * w + x;
                        output[pos] = bytedata[pos];
                    }
                });
            });

            bytedata = null;

            return DataTypes.byteToVTK1D(output,dims);
        }

        public static vtkImageData get_cartilage_surface_tmp(vtkImageData sample, int[,] surface, int depth = 50)
        {
            //vtk to byte
            byte[] bytedata = DataTypes.vtkToByte(sample);

            //Get dims
            int[] dims = sample.GetExtent();
            int h = dims[1] - dims[0] + 1;
            int w = dims[3] - dims[2] + 1;
            int d = dims[5] - dims[4] + 1;

            //Output array
            byte[] output = new byte[bytedata.Length];

            Parallel.For(0, h, (int y) =>
            {
                Parallel.For(0, w, (int x) =>
                {
                    for (int z = surface[y, x]; z > Math.Max(surface[y, x] - depth,0); z-=1)
                    {
                        int pos = z * (h * w) + y * w + x;
                        output[pos] = bytedata[pos];
                    }
                });
            });
            bytedata = null;
            return DataTypes.byteToVTK1D(output, dims);
        }

        public static vtkImageData get_cartilage_surface(vtkImageData sample, int[,] surface, int depth = 25)
        {
            //Get dims
            int[] dims0 = sample.GetExtent();
            int h0 = dims0[1] - dims0[0] + 1;
            int w0 = dims0[3] - dims0[2] + 1;
            int d0 = dims0[5] - dims0[4] + 1;

            
            //Get surface orientation and minimum and maximum indices

            List<Point2f> pointsx = new List<Point2f>();
            List<Point2f> pointsy = new List<Point2f>();

            int zmin = 65000;
            int zmax = 0;

            for (int y = 24; y<surface.GetLength(0)-24; y+=4)
            {
                for (int x = 24; x < surface.GetLength(1)-24; x+=4)
                {
                    pointsx.Add(new Point2f((float)x, (float)surface[y, x]));
                    pointsy.Add(new Point2f((float)y, (float)surface[y, x]));
                    if(surface[y, x] > zmax) { zmax = surface[y, x]; }
                    if(surface[y, x] < zmin) { zmin = surface[y, x]; }
                }
            }

            //Line fit
            Line2D linex = Cv2.FitLine(pointsx, DistanceTypes.L2, 0, 0.01, 0.01);
            Line2D liney = Cv2.FitLine(pointsy, DistanceTypes.L2, 0, 0.01, 0.01);

            //Get angles as degrees
            double thetax = Math.Atan(linex.Vy / (linex.Vx + 1e-9)) * 180.0 / Math.PI;
            double thetay = Math.Atan(liney.Vy / (liney.Vx + 1e-9)) * 180.0 / Math.PI;
            double[] angles = new double[] { thetax, thetay};
            int[] axes = new int[] {0, 1};

            //Crop and rotate
            int[] voidims = new int[] { dims0[0], dims0[1] , dims0[2] , dims0[3] , Math.Max(zmin-50, dims0[4]), Math.Min(zmax + 50, dims0[5]) };
            vtkExtractVOI cropper = vtkExtractVOI.New();
            cropper.SetInput(sample);
            cropper.SetVOI(voidims[0], voidims[1], voidims[2], voidims[3], voidims[4], voidims[5]);
            cropper.Update();

            vtkImageData tmpvtk = cropper.GetOutput();

            for(int k = 0; k<angles.Length; k++)
            {
                tmpvtk = Processing.rotate_sample(tmpvtk,angles[k],axes[k],0);
            }
            
            //Threshold the voi
            vtkImageData BW0 = Processing.otsu3D(tmpvtk, 0);
            vtkImageData BW1 = Processing.otsu3D(tmpvtk, 1);

            //Threshold and get new surface
            int[,] tmpsurf = get_surface(vtk_average(BW0, BW1));
            

            //Get new dimensions
            int[] dims1 = tmpvtk.GetExtent();
            int h1 = dims1[1] - dims1[0] + 1;
            int w1 = dims1[3] - dims1[2] + 1;
            int d1 = dims1[5] - dims1[4] + 1;

            //Input to byte data
            byte[] bytedata = DataTypes.vtkToByte(tmpvtk);

            //Empty byte array for output
            byte[] output = new byte[bytedata.Length];

            Parallel.For(0, h1, (int y) =>
            {
                Parallel.For(0, w1, (int x) =>
                {
                    for(int z = tmpsurf[y,x]; z > tmpsurf[y,x]-depth; z-=1)
                    {
                        if (z >= 0)
                        {
                            int pos = z * (h1 * w1) + y * w1 + x;
                            output[pos] = bytedata[pos];
                        }                        
                    }
                });
            });

            //Convert back to vtk image data
            tmpvtk = DataTypes.byteToVTK1D(output, dims1);

            
            
            //Return to original rotation
            for (int k = 0; k < angles.Length; k++)
            {
                tmpvtk = Processing.rotate_sample(tmpvtk, -angles[k], axes[k], 0);
            }
            
            

            return tmpvtk;
        }

        public static void get_analysis_vois(out vtkImageData dcartilage, out vtkImageData ccartilage, out vtkImageData scartilage,
            vtkImageData input, vtkImageData BCImask, int surf_depth = 25, int bone_depth = 0, double cartilage_depth = 0.6)
        {
            //Get input dimensions
            int[] dims = input.GetExtent();

            int h = dims[1] - dims[0] + 1;
            int w = dims[3] - dims[2] + 1;
            int d = dims[5] - dims[4] + 1;

            //Get strides
            int strideh = w;
            int stridew = 1;
            int strided = h * w;

            //Threshold input array for surface detection
            vtkImageData BW0 = Processing.otsu3D(input, 0);
            vtkImageData BW1 = Processing.otsu3D(input, 1);            

            //Get Surface indices
            int[,] surface = get_surface(vtk_average(BW0,BW1),0.0);

            BW0.Dispose();
            BW1.Dispose();

            //Get BCI indices
            int[,] BCI = get_surface(BCImask,0);

            //Empty array for depth
            int[,] depth = new int[BCI.GetLength(0),BCI.GetLength(1)];
                        
            //Compute depth            
            for (int y = 0; y < BCI.GetLength(0);y++)
            {
                //Compute depth
                for (int x = 0; x < BCI.GetLength(1); x++)
                {
                    int val = (int)((surface[y, x] - BCI[y, x]) * cartilage_depth);
                    depth[y, x] = val;                    
                }
            }

            //Get deep cartilage VOI
            dcartilage = get_deep_ac(input, BCI, depth);

            //Get calcified cartilage VOI
            ccartilage = get_calcified_cartilage(input, BCI);

            //Get cartilage surface
            scartilage = get_cartilage_surface_tmp(input,surface);
        }

        public static Mat largest_connected_component(Mat bw)
        {
            int h = bw.Height; int w = bw.Width;
            int[,] label_array = new int[h, w];
            int N = bw.ConnectedComponents(out label_array, PixelConnectivity.Connectivity8);
            int[] sums = new int[N-1];
            List<byte[,]> labels = new List<byte[,]>();
            
            //Count pixels in each componenet and make new binary objects
            for (int tmp = 1; tmp < N; tmp++)
            {
                byte[,] tmparray = new byte[h, w];
                Parallel.For(0,h, (int ky) =>
                {
                    Parallel.For(0, w, (int kx) =>
                    {
                        if (label_array[ky, kx] == tmp)
                        {
                            sums[tmp - 1] += 1;
                            tmparray[ky, kx] = 255;
                        }
                    });
                });
                labels.Add(tmparray);
            }

            //Find the component with largest area
            int idx = 0; int area = 0;
            for(int k = 0; k<N-1; k++)
            {
                if(sums[k]>area)
                {
                    idx = k; area = sums[k];
                }
            }

            return new Mat(h, w, MatType.CV_8UC1, labels.ElementAt(idx));

        }

        public static vtkImageData auto_rotate(vtkImageData vtkdata, double ori_thresh = 5)
        {
            //Get sample dimensions
            int[] dims = vtkdata.GetExtent();
            int h = dims[1] - dims[0] + 1;
            int w = dims[3] - dims[2] + 1;
            int d = dims[5] - dims[4] + 1;
            /*
            byte[] mult_vector = new byte[h * w * d];
            Parallel.For(0, h, (int ky) =>
            {
                Parallel.For(0, w, (int kx) =>
                {
                    Parallel.For(0, d, (int kz) =>
                    {
                        int pos = kz * (h * w) + ky * w + kx;
                        mult_vector[pos] = 1;
                    });
                });
            });
            //Set top of the sample to zero
            vtkImageData multiplier = DataTypes.byteToVTK1D(mult_vector, dims);
            vtkImageMathematics math = vtkImageMathematics.New();
            math.SetInput1(vtkdata);
            math.SetInput2(multiplier);
            math.SetOperationToMultiply();
            math.Update();
            */
            //Downsample the array by a factor of 10
            double s = 0.1;
            vtkImageData scaled = Processing.rescale_sample(vtkdata, s);
            //math.Dispose();


            //Get sample orientation
            double[] angles = Processing.grad_descent(scaled);
            angles = new double[] { angles[0], angles[1] };
            int[] axes = new int[] { 0, 1 };

            scaled.Dispose();

            Console.WriteLine("Angles: {0} | {1}", angles[0], angles[1]);

            for (int k = 0; k < angles.Length; k++)
            {
                if (Math.Abs(angles[k]) > ori_thresh)
                {
                    vtkdata = Processing.rotate_sample(vtkdata, angles[k], axes[k], 1);
                }
            }

            return vtkdata;
        }

        public static double dice_score_2d(byte[] image1, byte[] image2, double threshold = 0.0)
        {
            //Data dimensions
            int N = image1.Length;            
            //Parameters
            double intersection_sum = 0.0;
            double m1_sum = 0.0;
            double m2_sum = 0.0;
            double e = 1e-9;

            //Iterate over image data
            for(int k = 0; k < N; k++)
            {
                double val1 = (double)image1[k];
                double val2 = (double)image2[k];                    
                if (val1 > threshold) { val1 = 1.0; }
                else { val1 = 0.0; }
                if (val2 > threshold) { val2 = 1.0; }
                else { val2 = 0.0; }
                intersection_sum += val1 * val2;
                m1_sum += val1;
                m2_sum += val2;                
            }

            double score = (2 * intersection_sum) / (m1_sum + m2_sum + e);

            return score;
        }
    }

    /// <summary>
    /// Image loader, reads images in parallel 
    /// </summary>
    public class ParaLoader : IDisposable
    {
        //Methods for disposing the object
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        protected virtual void Dispose(bool disposing)
        {
            Disposed = true;
        }
        protected bool Disposed { get; private set; }

        //Declarations

        //Empty byte array
        byte[,,] data;
        //Empty image data
        vtkImageData vtkdata = vtkImageData.New();
        //Data dimensions
        int[] input_dims = new int[3];
        int[] output_dims = new int[6];
        //Empty list for files
        List<string> files;

        /// <summary>
        /// Set input files
        /// </summary>
        /// <param name="file">File path.</param>
        public void setInput(string file, int[] dims = null)
        {
            //Get files
            files = Functions.getFiles(file);
            //Read image and get dimensions
            Mat _tmp = new Mat(file, ImreadModes.GrayScale);
            input_dims[0] = _tmp.Height;
            input_dims[1] = _tmp.Width;
            input_dims[2] = files.Count;
            //Set output dimensions
            if (dims == null)
            {
                output_dims[0] = 0; output_dims[1] = input_dims[0];
                output_dims[2] = 0; output_dims[3] = input_dims[1];
                output_dims[4] = 0; output_dims[5] = input_dims[2];
            }
            else
            {
                output_dims = dims;
            }
            //Clear temp file
            _tmp.Dispose();

            //Set data extent. Data extent is set, so z-axis is along the
            //first dimension, and y-axis is along the last dimension.
            //This will be reversed when the data gets converted to vtkImagedata.
            data = new byte[output_dims[5] - output_dims[4], output_dims[1] - output_dims[0], output_dims[3] - output_dims[2]];            
        }

        /*
        /// <summary>
        /// Read image from file idx. The image is read using OpenCV, and converted to Bitmap.
        /// Bitmap is then read to the bytearray.
        /// </summary>
        /// <param name="idx">File index.</param>
        private void readImage(int idx)
        {
            //Read image from file idx. The image is read using OpenCV, and converted to Bitmap.
            //Bitmap is then read to the bytearray.
            Mat _tmp = new Mat(files[idx], ImreadModes.GrayScale);           
            
            Bitmap _image = BitmapConverter.ToBitmap(_tmp);
            //Lock bits
            Rectangle _rect = new Rectangle(0, 0, input_dims[1], input_dims[0]);
            BitmapData _bmpData =
                _image.LockBits(_rect, ImageLockMode.ReadOnly, _image.PixelFormat);

            //Get the address of first line
            IntPtr _ptr = _bmpData.Scan0;

            //Declare new array for gray scale values
            int _bytes = Math.Abs(_bmpData.Stride) * _bmpData.Height;
            byte[] _grayValues = new byte[_bytes];

            //Copy the rgb values to the new array
            Marshal.Copy(_ptr, _grayValues, 0, _bytes);

            //Method for correct pixel mapping
            Func<int, int, int, int> mapPixel = Functions.GetPixelMapper(_image.PixelFormat, _bmpData.Stride);

            //Read bits to byte array in parallel
            //Remember the data orientation
            Parallel.For(output_dims[0], output_dims[1], (int h) =>
            {
                Parallel.For(output_dims[2], output_dims[3], (int w) =>
                {
                    data[idx - output_dims[4], h - output_dims[0], w - output_dims[2]] = _grayValues[mapPixel(h, w, 0)];
                });
            });            
        }
        */

        /// <summary>
        /// Read image from file idx. The image is read using OpenCV, and converted to byte array.
        /// </summary>
        /// <param name="idx">File index.</param>
        private void readImage(int idx)
        {
            //Read image from file idx. The image is read using OpenCV, and converted to byte array.            
            Mat _tmp = new Mat(files[idx], ImreadModes.GrayScale);
            byte[] _vals = new byte[_tmp.Width * _tmp.Height];
            IntPtr pointer = _tmp.Data;
            Marshal.Copy(pointer, _vals, 0, _tmp.Width * _tmp.Height);
            
            //Read bits to byte array in parallel
            //Remember the data orientation
            Parallel.For(output_dims[0], output_dims[1], (int h) =>
            {
                Parallel.For(output_dims[2], output_dims[3], (int w) =>
                {
                    int pos = h * (_tmp.Width) + w;
                    data[idx - output_dims[4], h - output_dims[0], w - output_dims[2]] = _vals[pos];
                });
            });

            _tmp.Dispose();
            _vals = null;
        }

        /// <summary>
        /// Load all images in parallel
        /// </summary>
        public void Load()
        {
            //Loop over files
            Parallel.For(output_dims[4], output_dims[5], (int d) =>
            {
                readImage(d);
            });
        }

        /// <summary>
        /// Extract data as vtkImageData
        /// </summary>
        /// <returns>Converted data as vtkImageData variable.</returns>
        public vtkImageData GetData()
        {
            //Conver byte data to vtkImageData
            vtkdata = DataTypes.byteToVTK(data);
            /*
            //flip the data along z axis
            vtkImageFlip flipper = vtkImageFlip.New();
            flipper.SetInput(vtkdata);
            flipper.SetFilteredAxis(2);
            flipper.Update();
            */
            return vtkdata; //flipper.GetOutput();
        }
    }

}
