using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LBPLibrary;
using System.Windows.Forms;
using System.Globalization;

using Accord.Math;

using Kitware.VTK;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace HistoGrading.Components
{
    /// <summary>
    /// Class that includes functions used for calculating OA grades from cartilage surface images.
    /// </summary>
    public class Grading
    {

        /// <summary>
        /// Loads default model weights.
        /// Default model file is on project folder named "weights.dat"
        /// </summary>
        /// <param name="mod">Model containing grading variables</param>
        /// <param name="param">Excel containing LBP parameters</param>
        /// <param name="model_path">Path to model</param>
        /// <param name="param_path">Path to parameters</param>
        /// <returns>Grading model and LBP parameters</returns>
        public static string LoadModel(out Model mod, out Parameters param, string model_path, string param_path)
        {            
            // Path to files
            string path =
                    new DirectoryInfo(Directory.GetCurrentDirectory()) // Get current directory
                    .Parent.Parent.Parent.Parent.FullName; // Move to correct location and add file name

            // Load parameters from .csv
            var paramList = DataTypes.ReadCSV(path + param_path).ToInt32();
            var paramFlat = new int[paramList.Length];
            for (int i = 0; i < paramList.Length; i++)
            {
                paramFlat[i] = paramList[0, i];
            }

            // Set parameters
            param = new Parameters()
            {
                W_stand = new int[] { paramFlat[0], paramFlat[1] , paramFlat[2] , paramFlat[3] },
                Neighbours = paramFlat[4],
                LargeRadius = paramFlat[5],
                Radius = paramFlat[6],
                W_c = paramFlat[7],
                W_r = new int[] { paramFlat[8], paramFlat[9] }
            };

            // Read weights from .dat file
            var reader = new BinaryWriterApp(path + model_path);
            try
            {
                reader.ReadWeights();
            }
            catch (Exception)
            {
                throw new Exception("Could not find weights.dat! Check that default model is on correct folder.");
            }

            // Set model variables
            mod = new Model();
            mod.nComp = reader.ncomp;
            mod.eigenVectors = reader.eigenVectors;
            mod.singularValues = reader.singularValues;
            mod.weights = reader.weights;
            mod.mean = reader.mean;

            return path;
        }

        /*
        /// <summary>
        /// Calculates OA grade prediction from cartilage surface.
        /// </summary>
        /// <param name="volume">Data array.</param>
        /// <param name="filename">Path for saving results.</param>
        /// <param name="VOIcoordinates">Coordinates for extracted VOI.</param>
        /// <returns>Returns string containing the OA grade</returns>
        public static string PredictSurface(ref Rendering.renderPipeLine volume, string filename, out int[,] VOIcoordinates)
        {
            // Initialize Grading form
            var grading = new GradingForm(); grading.Show();

            // Default variables
            int threshold = 80;
            int[] size = { 400, 30 };

            // Load default model
            string path = LoadModel(out Model mod, @"\Default\weights.dat");
            grading.UpdateModel(); grading.Show();

            // Surface extraction
            Processing.VOIExtraction(ref volume, threshold, size, "surface",
                out VOIcoordinates, out byte[,,] surface);

            // Mean and std images
            //Processing.MeanAndStd(surface, out double[,] meanImage, out double[,] stdImage);
            vtkImageData surf = new vtkImageData();
            Processing.MeanAndStd(surf, out double[,] meanImage, out double[,] stdImage);
            // Show images to user
            grading.UpdateMean(
                DataTypes.DoubleToBitmap(meanImage),
                DataTypes.DoubleToBitmap(stdImage),
                DataTypes.DoubleToBitmap(Elementwise.Add(meanImage, stdImage)));
            grading.Show();

            // LBP features
            // Get default parameters
            Parameters param = new Parameters();
            grading.UpdateParameters(param);
            int[,] features = LBP(meanImage.Add(stdImage), param, 
                out double[,] LBPIL, out double[,] LBPIS, out double[,] LBPIR);
            // Show LBP images to user
            grading.UpdateLBP(
                DataTypes.DoubleToBitmap(LBPIL),
                DataTypes.DoubleToBitmap(LBPIS),
                DataTypes.DoubleToBitmap(LBPIR));
            grading.Show();

            // Calculate PCA and regression
            FeaturesToGrade(features, mod, path, filename, out string grade);
            grading.UpdateGrade(grade); grading.Show();

            // Save results
            SaveResult(grade, path, filename);
            grading.UpdateGrade(grade); grading.Show();

            return "OA grade (surface): " + grade;
        }
        */
        
        /*    
        /// <summary>
        /// Calculates OA grade prediction from bone-cartilage interface.
        /// Above interface deep cartilage volume is extracted, calcified cartilage below interface.
        /// </summary>
        /// <param name="volume">Data array.</param>
        /// <param name="filename">Path for saving results.</param>
        /// <param name="deepCoordinates">Coordinates for extracted deep cartilage VOI.</param>
        /// <param name="calcifiedCoordinates">Coordinates for extracted calcified cartilage VOI.</param>
        /// <returns>Returns string containing the OA grade</returns>
        public static string PredictBCI(ref Rendering.renderPipeLine volume, string filename, out int[,] deepCoordinates, out int[,] calcifiedCoordinates)
        {
            // Initialize Grading form
            var grading = new GradingForm(); grading.Show();

            // Default variables
            int threshold = 80;
            int[] size = { 400, 30 };

            // Load default model
            string path = LoadModel(out Model mod, @"\Default\weights.dat");
            grading.UpdateModel(); grading.Show();

            // Deep cartilage extraction
            Processing.VOIExtraction(ref volume, threshold, size, "deep",
                out deepCoordinates, out byte[,,] deepSurface);
            // Calcified cartilage extraction
            Processing.VOIExtraction(ref volume, threshold, size, "calcified",
                out calcifiedCoordinates, out byte[,,] calcifiedSurface);

            // Mean and std images
            vtkImageData surf = new vtkImageData();
            //Processing.MeanAndStd(deepSurface, out double[,] meanImage, out double[,] stdImage);
            Processing.MeanAndStd(surf, out double[,] meanImage, out double[,] stdImage);
            //Processing.MeanAndStd(calcifiedSurface, out double[,] meanccImage, out double[,] stdccImage);
            Processing.MeanAndStd(surf, out double[,] meanccImage, out double[,] stdccImage);
            // Show images to user
            grading.UpdateMean(
                DataTypes.DoubleToBitmap(meanImage),
                DataTypes.DoubleToBitmap(stdImage),
                DataTypes.DoubleToBitmap(Elementwise.Add(meanImage, stdImage)));
            grading.Show();

            // LBP features
            // Get default parameters
            Parameters param = new Parameters();
            grading.UpdateParameters(param);
            int[,] features = LBP(meanImage.Add(stdImage), param,
                out double[,] LBPIL, out double[,] LBPIS, out double[,] LBPIR);
            // Show LBP images to user
            grading.UpdateLBP(
                DataTypes.DoubleToBitmap(LBPIL),
                DataTypes.DoubleToBitmap(LBPIS),
                DataTypes.DoubleToBitmap(LBPIR));
            grading.Show();

            // Calculate PCA and regression
            FeaturesToGrade(features, mod, path, filename, out string grade);
            grading.UpdateGrade(grade); grading.Show();

            // Save results
            SaveResult(grade, path, filename);

            return "OA grade (surface): " + grade;
        }
        */
        /// <summary>
        /// Compares predicted grades to reference grades.
        /// </summary>
        /// <param name="grades">Predicted grades.</param>
        /// <returns>Sum of absolute differences.</returns>
        public static double CompareGrades(double[] grades)
        {
            // Load actual grades
            string filename = new DirectoryInfo(Directory.GetCurrentDirectory()) // Get current directory
                .Parent.Parent.Parent.Parent.FullName + @"\Default\sample_grades.csv"; // Move to correct location and add file name

            int[] actualGrades =
                LBPLibrary.Functions.ArrayToVector(
                LBPLibrary.Functions.ReadCSV(filename))
                .ToInt32();

            // Difference between actual grades
            double[] loss = grades.Subtract(actualGrades.ToDouble());

            return Matrix.Sum(Elementwise.Abs(loss)); // Absolute sum
        }

        /// <summary>
        /// Calculates LBP features using LBPLibrary Nuget package.
        /// Takes grayscale image as input.
        /// Currently software inputs sum of mean and standard images of surface VOI.
        /// </summary>
        /// <returns>Feature array.</returns>
        public static int[,] LBP(double[,] inputImage, Parameters param, out double[,] LBPIL, out double[,] LBPIS, out double[,] LBPIR, string zone)
        {
            // Grayscale standardization
            var standrd = new LocalStandardization(param.W_stand[0], param.W_stand[1], param.W_stand[2], param.W_stand[3]);
            standrd.Standardize(ref inputImage, param.Method); // standardize given image

            // Replace NaN values
            bool nans = false;
            var mean = LBPLibrary.Functions.Mean(inputImage);
            Parallel.For(0, inputImage.GetLength(0), i =>
            {
                Parallel.For(0, inputImage.GetLength(1), j =>
                {
                    if (double.IsNaN(inputImage[i, j]))
                    {
                        inputImage[i, j] = mean;
                        nans = true;
                    }
                });
            });
            if (nans)
                Console.WriteLine("Input includes NaN values!");

            ////Visualize input image
            //double min = 1e9; double max = -1e9;
            //for (int kx = 0; kx < inputImage.GetLength(1); kx++)
            //{
            //    for (int ky = 0; ky < inputImage.GetLength(0); ky++)
            //    {
            //        double val = inputImage[ky, kx];
            //        if (val > max) { max = val; }
            //        if (val < min) { min = val; }
            //    }
            //}

            //byte[,] valim = new byte[inputImage.GetLength(0), inputImage.GetLength(1)];
            //for (int kx = 0; kx < inputImage.GetLength(1); kx++)
            //{
            //    for (int ky = 0; ky < inputImage.GetLength(0); ky++)
            //    {
            //        double val = inputImage[ky, kx];
            //        valim[ky, kx] = (byte)(255.0 * (val - min) / (max - min));
            //    }
            //}
            //Console.WriteLine("inputImage min | max: {0} | {1}", min, max);
            //Mat valmat = new Mat(inputImage.GetLength(0), inputImage.GetLength(1), MatType.CV_8UC1, valim);
            //using (Window win = new Window("inputimage", WindowMode.AutoSize, image: valmat))
            //{
            //    Cv2.WaitKey();
            //}

            // LBP calculation
            LBPApplication.PipelineMRELBP(inputImage, param,
                    out LBPIL, out LBPIS, out LBPIR, out int[] histL, out int[] histS, out int[] histR, out int[] histCenter);

            // Concatenate histograms
            int[] f = Matrix.Concatenate(histCenter, Matrix.Concatenate(histL, Matrix.Concatenate(histS, histR)));
            int[,] features = new int[0, 0];

            return Matrix.Concatenate(features, f);
        }

        /// <summary>
        /// Calculates OA grade from LBP features using pretrained PCA an regression with given model.
        /// </summary>
        /// <param name="features">MRELBP features.</param>
        /// <param name="mod">Model that includes PCA eigenvectors, mean feature, and regression weights.</param>
        /// <param name="path">Path to save results.</param>
        /// <param name="samplename">Name of analysed sample. This is saved with estimated grade to results.csv</param>
        /// <param name="grade"></param>
        public static void FeaturesToGrade(int[,] features, Model mod, string path, string samplename, out string grade)
        {
            // Centering
            double[,] dataAdjust = Processing.SubtractMean(features.ToDouble(), mod.mean);
            
            // Whitening and PCA matrix
            int w = mod.eigenVectors.GetLength(0);
            double[,] transform = new double[w, mod.nComp];
            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < mod.nComp; j++)
                {
                    transform[i, j] = mod.eigenVectors[i, j] / mod.singularValues[j];
                }
            }
            double[,] PCA = dataAdjust.Dot(transform);

            // Regression
            double[] grades = PCA.Dot(mod.weights).Add(1.5);

            // Convert estimated grade to string
            /*
            if (grades[0] < 1)
                grade = grades[0].ToString("0.##", CultureInfo.InvariantCulture);
            else if (grades[0] < 0)
                grade = "0.00";
            else if (grades[0] > 3)
                grade = "3.00";
            else
                grade = grades[0].ToString("####.##", CultureInfo.InvariantCulture);
            */
            grade = grades[0].ToString("###0.##", CultureInfo.InvariantCulture);
        }

        /// <summary>
        /// Save results to .csv file.
        /// Check that file is not opened.
        /// </summary>
        /// <param name="grade">OA grade.</param>
        /// <param name="path">Save path.</param>
        /// <param name="filename">Sample name.</param>
        private static void SaveResult(string grade, string path, string filename)
        {
            var result = DialogResult.OK;
            while (result == DialogResult.OK)
            {
                try
                {
                    File.AppendAllText(
                        path + @"\Default\results.csv", filename + ";"
                        + grade + "\n");
                    break;
                }
                catch (Exception)
                {
                    result = MessageBox.Show(
                        "Results file is opened. Please close the file before continuing."
                        , "Error!", MessageBoxButtons.OKCancel);
                }
            }
        }

        public static string grade_voi(string zone, string sample, double[,] mean, double[,] sd, string model_path, string param_path)
        {
            // Initialize Grading form
            var grading = new GradingForm(); grading.Show();

            // Load grading model
            string path = LoadModel(out Model mod, out Parameters param, model_path, param_path);

            // Show images to user
            grading.UpdateModel(); grading.Show();
            grading.UpdateMean(
                DataTypes.DoubleToBitmap(mean),
                DataTypes.DoubleToBitmap(sd),
                DataTypes.DoubleToBitmap(Elementwise.Add(mean, sd)));
            grading.Show();

            var meansd = Elementwise.Add(mean, sd);

            //Get LBP features
            grading.UpdateParameters(param);
            int[,] features = LBP(meansd, param, out double[,] LBPIL, out double[,] LBPIS, out double[,] LBPIR,zone);

            grading.UpdateLBP(
                DataTypes.DoubleToBitmap(LBPIL),
                DataTypes.DoubleToBitmap(LBPIS),
                DataTypes.DoubleToBitmap(LBPIR));
            grading.Show();
            
            //Get grade
            // Calculate PCA and regression
            FeaturesToGrade(features, mod, path, sample, out string grade);
            Console.WriteLine(grade);
            grading.UpdateGrade(grade); grading.Show();
            SaveResult(grade, path, sample);

            return grade;
        }
    }

    /// <summary>
    /// Class <c>Model</c> for saving model variables. Loaded from .dat file
    /// using BinaryWriterApp class of LBPLibrary. See also:
    /// <seealso cref="Grading.LoadModel(out Model)"/>
    /// </summary>
    public class Model
    {
        /// <summary>
        /// Number of principal components used.
        /// </summary>
        public int nComp;
        /// <summary>
        /// Eigenvectors of pretrained PCA analysis including number of components selected.
        /// </summary>
        public float[,] eigenVectors;
        /// <summary>
        /// Singularvalues of pretrained PCA analysis. Square root of Eigenvalues.
        /// </summary>
        public float[] singularValues;
        /// <summary>
        /// Feature weights from pretrained linear regression.
        /// </summary>
        public double[] weights;
        /// <summary>
        /// Mean feature vector.
        /// </summary>
        public double[] mean;
    }
}
