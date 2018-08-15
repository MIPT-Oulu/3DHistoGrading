using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Kitware.VTK;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace HistoGrading.Components
{
    class Functions
    {
        //Loader
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

        //Slicer
        public static vtkImageData volumeSlicer(vtkImageData volume, int[] sliceN, int axis)
        {
            /*Gets a 2D slice from the 3D data*/

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

        //Prerocessing
        public static vtkImageData scalarCopy(vtkImageData data)
        {
            /*DEPERCATED!!*/
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

        public static vtkImageData loadVTK(string path)
        {
            /*Read all images in folder containing file from given path
             *Uses ParaLoader class to get the data in vtk format*/

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

        //Find files
        public static List<string> getFiles(string file)
        {
            /*Find all files which correspond to selected file.
             *When reading multiple files, files must start with the same name, which ends in a digit,
             *as the selected file and have the same extension.*/

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

        //Image loader, reads images in parallel
        private class ParaLoader
        {
            //Declarations

            //Empty byte array
            byte[,,] data;
            //Empty image data
            vtkImageData vtkdata = vtkImageData.New();
            //Data dimensiosn
            int[] dims = new int[] { 0, 0, 0 };
            //Empty list for files
            List<string> files;

            //Set input files
            public void setInput(string file)
            {
                //Get files
                files = getFiles(file);
                //Read image and get dimensions
                Mat _tmp = new Mat(file, ImreadModes.GrayScale);
                dims[0] = _tmp.Height;
                dims[1] = _tmp.Width;
                dims[2] = files.Count;
                //Clear temp file
                _tmp.Dispose();

                //Set data extent. Data extent is set, so z-axis is along the
                //fisrt dimension, and y-axis is along the last dimension.
                //This will be reversed when the data gets converted to vtkImagedata.
                data = new byte[dims[2], dims[1], dims[0]];
            }

            //Read image
            private void readImage(int idx)
            {
                //Read image from file idx. The image is read using OpenCV, and converted to Bitmap.
                //Bitmap is then read to the bytearray.
                Mat _tmp = new Mat(files[idx], ImreadModes.GrayScale);
                Bitmap _image = BitmapConverter.ToBitmap(_tmp);
                //Lock bits
                Rectangle _rect = new Rectangle(0, 0, dims[1], dims[0]);
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
                Func<int, int, int, int> mapPixel = GetPixelMapper(_image.PixelFormat, _bmpData.Stride);

                //Read bits to byte array in parallel
                //Remember the data orientation
                Parallel.For(0, dims[0], (int h) =>
                {
                    Parallel.For(0, dims[1], (int w) =>
                    {
                        data[idx, w, h] = _grayValues[mapPixel(h, w, 0)];
                    });
                });
            }

            //Load all images in parallel
            public void Load()
            {
                //Loop over files
                Parallel.For(0, dims[2], (int d) =>
                {
                    readImage(d);
                });
            }

            //Extract data as vtkImageData
            public vtkImageData GetData()
            {
                //Character array for conversion
                vtkUnsignedCharArray charArray = vtkUnsignedCharArray.New();
                //Pin byte array
                GCHandle pinnedArray = GCHandle.Alloc(data, GCHandleType.Pinned);
                //Set character array input
                charArray.SetArray(pinnedArray.AddrOfPinnedObject(), dims[0] * dims[1] * dims[2], 1);
                //Set vtkdata properties and connect array
                //Data from char array
                vtkdata.GetPointData().SetScalars(charArray);
                //Number of scalars/pixel
                vtkdata.SetNumberOfScalarComponents(1);
                //Data extent, 1st and last axis are swapped from the char array
                //Data is converted back to original orientation
                vtkdata.SetExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1);
                //Scalar type
                vtkdata.SetScalarTypeToUnsignedChar();
                vtkdata.Update();

                //Clear memory
                data = null;
                //Return vtk data
                return vtkdata;
            }

            //Function for correctly mapping the pixel values, copied from CNTK examples
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
        }
    }
}
