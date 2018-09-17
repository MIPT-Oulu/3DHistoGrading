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
                slicer.SetVOI(sliceN[0] - 1, sliceN[0], dims[2], dims[3], dims[4], dims[5]);
                slicer.Update();
                //Permute image (not necessary here)
                permuter.SetInputConnection(slicer.GetOutputPort());
                permuter.SetFilteredAxes(1, 2, 0);
                permuter.Update();
            }
            //Transverse plane YZ
            if (axis == 1)
            {
                //Set VOI
                slicer.SetVOI(dims[0], dims[1], sliceN[1] - 1, sliceN[1], dims[4], dims[5]);
                slicer.Update();
                //Permute image
                permuter.SetInputConnection(slicer.GetOutputPort());
                permuter.SetFilteredAxes(0, 2, 1);
                permuter.Update();
            }
            //Transverse plane, XZ
            if (axis == 2)
            {
                //Set VOI
                slicer.SetVOI(dims[0], dims[1], dims[2], dims[3], sliceN[2] - 1, sliceN[2]);
                slicer.Update();
                //Permute image
                permuter.SetInputConnection(slicer.GetOutputPort());
                permuter.SetFilteredAxes(0, 1, 2);
                permuter.Update();
            }
            //slicer.Update();

            //Return copy of the slice
            return permuter.GetOutput();
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
            vtkImageData data = loader.GetData();
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
    }

    /// <summary>
    /// Image loader, reads images in parallel 
    /// </summary>
    public class ParaLoader
    {
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
            data = new byte[output_dims[5] - output_dims[4], output_dims[3] - output_dims[2], output_dims[1] - output_dims[0]];
        }

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
                    data[idx - output_dims[4], w - output_dims[2], h - output_dims[0]] = _grayValues[mapPixel(h, w, 0)];
                });
            });
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
            return vtkdata;
        }

    }
}
