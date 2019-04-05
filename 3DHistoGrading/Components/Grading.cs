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
            mod.weightsLog = reader.weightsLog;
            mod.intercept = reader.intercept;
            mod.interceptLog = reader.interceptLog;
            mod.mean = reader.mean;
            

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

            return path;
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
        public static double[,] LBP(double[,] inputImage, Parameters param, out double[,] LBPIL, out double[,] LBPIS, out double[,] LBPIR, string zone)
        {
            // LBP calculation
            LBPApplication.PipelineMRELBP(inputImage, param,
                    out LBPIL, out LBPIS, out LBPIR, out int[] histL, out int[] histS, out int[] histR, out int[] histCenter);

            // Concatenate histograms
            int[] f = Matrix.Concatenate(histCenter, Matrix.Concatenate(histL, Matrix.Concatenate(histS, histR)));

            // Normalize
            double[] fScaled = Elementwise.Divide(f, f.Sum());
            double[,] features = new double[0, 0];

            return Matrix.Concatenate(features, fScaled);
        }

        /// <summary>
        /// Calculates OA grade from LBP features using pretrained PCA an regression with given model.
        /// </summary>
        /// <param name="features">MRELBP features.</param>
        /// <param name="mod">Model that includes PCA eigenvectors, mean feature, and regression weights.</param>
        /// <param name="path">Path to save results.</param>
        /// <param name="samplename">Name of analysed sample. This is saved with estimated grade to results.csv</param>
        /// <param name="grade"></param>
        public static bool FeaturesToGrade(double[,] features, Model mod, string path, string samplename, out string grade)
        {
            // Centering
            double[,] dataAdjust = Processing.SubtractMean(features, mod.mean);
            
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
            double[] grades = PCA.Dot(mod.weights).Add(mod.intercept);
            double[] logistic = PCA.Dot(mod.weightsLog).Add(mod.interceptLog);
            bool log;
            if (logistic[0] > 0.5)
            {
                Console.WriteLine("Logistic regression estimated sample as degenerated.");
                log = true;
            }
            else
            {
                Console.WriteLine("Logistic regression estimated sample as healthy / mildly degenerated.");
                log = false;
            }

            // Convert estimated grade to string
            if (grades[0] < 0)
                grade = "0.00";
            else if (grades[0] > 3)
                grade = "3.00";
            else
                grade = grades[0].ToString("###0.##", CultureInfo.InvariantCulture);
            return log;
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

        /// <summary>
        /// Grading pipeline. Outputs the grades for each VOI as a string and results can be viewed from external grading window.
        /// </summary>
        /// <param name="zone">Name of the zone to be graded.</param>
        /// <param name="sample">Sample name.</param>
        /// <param name="mean">Mean image from the zone's VOI along depth-axis.</param>
        /// <param name="sd">Standard deviation image from the zone's VOI along depth-axis.</param>
        /// <param name="model_path">Path to grading model. Binary data file including values for PCA and regression.</param>
        /// <param name="param_path">Path to LBP parameters.</param>
        /// <returns></returns>
        public static string grade_voi(string zone, string sample, double[,] mean, double[,] sd, string model_path, string param_path)
        {
            // Initialize Grading form
            var grading = new GradingForm(); grading.Show();
            grading.Text = zone;

            // Load grading model
            string path = LoadModel(out Model mod, out Parameters param, model_path, param_path);

            // Sum mean and std images
            var meansd = Elementwise.Add(mean, sd);

            // Grayscale standardization
            var standrd = new LocalStandardization(param.W_stand[0], param.W_stand[1], param.W_stand[2], param.W_stand[3]);
            standrd.Standardize(ref meansd, param.Method); // standardize given image

            // Replace NaN values from standardized image
            bool nans = false;
            var mu = LBPLibrary.Functions.Mean(meansd);
            Parallel.For(0, meansd.GetLength(0), i =>
            {
                Parallel.For(0, meansd.GetLength(1), j =>
                {
                    if (double.IsNaN(meansd[i, j]))
                    {
                        meansd[i, j] = 0;
                        nans = true;
                    }
                });
            });
            if (nans)
                Console.WriteLine("Input includes NaN values!");

            // Show images to user
            grading.UpdateModel(zone); grading.Show();
            grading.UpdateMean(
                DataTypes.DoubleToBitmap(mean),
                DataTypes.DoubleToBitmap(sd),
                DataTypes.DoubleToBitmap(meansd));
            grading.Show();

            //Get LBP features
            grading.UpdateParameters(param);
            double[,] features = LBP(meansd, param, out double[,] LBPIL, out double[,] LBPIS, out double[,] LBPIR,zone);

            grading.UpdateLBP(
                DataTypes.DoubleToBitmap(LBPIL),
                DataTypes.DoubleToBitmap(LBPIS),
                DataTypes.DoubleToBitmap(LBPIR));
            grading.Show();
            
            //Get grade
            // Calculate PCA and regression
            bool degenerated = FeaturesToGrade(features, mod, path, sample, out string grade);
            Console.WriteLine(grade);
            grading.UpdateGrade(grade, degenerated); grading.Show();
            SaveResult(grade, path, sample);

            return grade;
        }
    }

    /// <summary>
    /// Class <c>Model</c> for saving model variables. Loaded from .dat file
    /// using BinaryWriterApp class of LBPLibrary. See also:
    /// <seealso cref="Grading.LoadModel(out Model, out Parameters, string, string)"/>
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
        /// Pretrained linear and logistic regression weights. Readed with ReadWeights method from .dat file.
        /// </summary>
        public double[] weights, weightsLog;
        /// <summary>
        /// Pretrained linear and logistic regression intercept term.
        /// </summary>
        public double intercept, interceptLog;
        /// <summary>
        /// Mean feature vector.
        /// </summary>
        public double[] mean;
    }
}
