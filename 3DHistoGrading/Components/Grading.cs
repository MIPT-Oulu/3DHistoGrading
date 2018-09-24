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
using Accord.Math.Decompositions;
using Accord.Statistics;

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
        /// <param name="mod">Model containing all variables</param>
        /// <returns>Model path</returns>
        public static string LoadModel(ref Model mod)
        {
            // Path to model (weights.dat)
            string path =
                    new DirectoryInfo(Directory.GetCurrentDirectory()) // Get current directory
                    .Parent.Parent.Parent.Parent.FullName; // Move to correct location and add file name

            // Read weights from .dat file
            var reader = new BinaryWriterApp(path + @"\Default\weights.dat");
            try
            {
                reader.ReadWeights();
            }
            catch (Exception)
            {
                throw new Exception("Could not find weights.dat! Check that default model is on correct folder.");
            }

            // Update model variables
            mod.nComp = reader.ncomp;
            mod.eigenVectors = reader.eigenVectors;
            mod.singularValues = reader.singularValues;
            mod.weights = reader.weights;
            mod.mean = reader.mean;

            return path;
        }

        /// <summary>
        /// Calculates OA grade prediction.
        /// </summary>
        /// <param name="mod">Loaded model.</param>
        /// <param name="features">LBP features.</param>
        /// <param name="volume">Data array.</param>
        /// <param name="filename">Path for saving results.</param>
        /// <returns>Returns string containing the OA grade</returns>
        public static string Predict(Model mod, ref int[,] features, ref Rendering.renderPipeLine volume, string filename)
        {
            // Initialize Grading form
            var grading = new GradingForm();
            grading.Show();
            

            // Default variables
            int threshold = 80;
            int[] size = { 400, 30 };

            // Load default model
            string path = LoadModel(ref mod);
            grading.UpdateModel(); grading.Show();

            // Surface extraction
            Processing.SurfaceExtraction(ref volume, threshold, size, out int[,] surfacecoordinates, out byte[,,] surface);

            // Mean and std images
            Processing.MeanAndStd(surface, out double[,] meanImage, out double[,] stdImage);
            // Show images to user
            grading.UpdateMean(
                DataTypes.DoubleToBitmap(meanImage),
                DataTypes.DoubleToBitmap(stdImage));
            grading.Show();

            // LBP features
            // Get default parameters
            Parameters param = new Parameters();
            grading.UpdateParameters(param);
            features = LBP(meanImage.Add(stdImage), param, out double[,] LBPIL, out double[,] LBPIS, out double[,] LBPIR);
            // Show LBP images to user
            grading.UpdateLBP(
                DataTypes.DoubleToBitmap(LBPIL),
                DataTypes.DoubleToBitmap(LBPIS),
                DataTypes.DoubleToBitmap(LBPIR));
            grading.Show();

            // PCA
            double[,] dataAdjust = Processing.SubtractMean(features.ToDouble(), mod.mean);
            double[,] PCA = dataAdjust.Dot(mod.eigenVectors.ToDouble());

            // Regression
            double[] grades = PCA.Dot(mod.weights).Add(1.5);
            string grade = grades[0].ToString("####.##", CultureInfo.InvariantCulture).PadLeft(1, '0'));

            // Save results
            SaveResult(grade, path, filename);
            grading.UpdateGrade(grade); grading.Show();

            return "OA grade: " + grade;
            //double sum = CompareGrades(grade);
            //return "Sum of differences between pretrained model and actual grade: " + sum.ToString("###.###", CultureInfo.InvariantCulture);
        }

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
        public static int[,] LBP(double[,] inputImage, Parameters param, out double[,] LBPIL, out double[,] LBPIS, out double[,] LBPIR)
        {
            // Grayscale standardization
            var standrd = new LocalStandardization(param.W_stand[0], param.W_stand[1], param.W_stand[2], param.W_stand[3]);
            standrd.Standardize(ref inputImage, param.Method); // standardize given image

            // LBP calculation
            LBPApplication.PipelineMRELBP(inputImage, param,
                    out LBPIL, out LBPIS, out LBPIR, out int[] histL, out int[] histS, out int[] histR, out int[] histCenter);

            // Concatenate histograms
            int[] f = Matrix.Concatenate(histCenter, Matrix.Concatenate(histL, Matrix.Concatenate(histS, histR)));
            int[,] features = new int[0, 0];

            return Matrix.Concatenate(features, f);
        }

        /// <summary>
        /// Save results to .csv file.
        /// Check that file is not opened.
        /// </summary>
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
    }

    /// <summary>
    /// Class <c>Model</c> for saving model variables. Loaded from .dat file
    /// using BinaryWriterApp class of LBPLibrary. See also:
    /// <seealso cref="Grading.LoadModel(ref Model)"/>
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
