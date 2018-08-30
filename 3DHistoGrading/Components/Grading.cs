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
        /// <returns>State of loading model</returns>
        public static string LoadModel(ref Model mod)
        {
            // Path to model (weights.dat)
            string filename =
                    new DirectoryInfo(Directory.GetCurrentDirectory()) // Get current directory
                    .Parent.Parent.Parent.Parent.FullName + @"\Default\weights.dat"; // Move to correct location and add file name

            // Read weights from .dat file
            var reader = new BinaryWriterApp(filename);
            reader.ReadWeights();

            // Update model variables
            mod.nComp = reader.ncomp;
            mod.eigenVectors = reader.eigenVectors;
            mod.singularValues = reader.singularValues;
            mod.weights = reader.weights;

            return "Model loaded";
        }

        /// <summary>
        /// Calculates OA grade prediction.
        /// </summary>
        /// <param name="mod">Loaded model.</param>
        /// <param name="features">LBP features.</param>
        /// <returns>Returns string containing the OA grade</returns>
        public static string Predict(Model mod, ref int[,] features)
        {
            // Check if model is not loaded
            if (mod.nComp == 0 || mod.singularValues == null || mod.eigenVectors == null || mod.weights == null)
                return "Model not loaded";

            //
            // LBP features
            //
            if (features.Length == 0) // Calculate if doesn't exist already
            {
                // Load LBP features
                string filename = 
                    new DirectoryInfo(Directory.GetCurrentDirectory()) // Get current directory
                    .Parent.Parent.Parent.Parent.FullName + @"\Default\sample_features.csv"; // Move to correct location and add file name

                features = LBPLibrary.Functions
                    .ReadCSV(filename)
                    .ToInt32();
            }
            //
            //if (features.Length == 0) // Calculate if doesn't exist already
            //    features = LBP();

            // PCA
            double[,] dataAdjust = SubtractMean(features.ToDouble());
            double[,] PCA = dataAdjust.Dot(mod.eigenVectors.ToDouble());

            // Regression
            double[] grade = PCA.Dot(mod.weights).Add(1.5);

            double sum = CompareGrades(grade);

            //return "OA grade (sample 1): " + grade[0].ToString("####.##", CultureInfo.InvariantCulture);
            return "Sum of differences between pretrained model and actual grade: " + sum.ToString("###.###", CultureInfo.InvariantCulture);
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
        /// Currently asks user to input  directories for surface images and save paths.
        /// When surfaceimages can be calculated in GUI, this should be modified.
        /// </summary>
        /// <returns>Feature array.</returns>
        public static int[,] LBP()
        {
            // Select load path
            string path = null;
            var fbd = new FolderBrowserDialog() { Description = "Select the directory to load images" };
            if (fbd.ShowDialog() == DialogResult.OK)
                path = fbd.SelectedPath;
            else
                return new int[0, 0];

            string meanpath = null, stdpath = null, savepath = null;
            // Select mean image path
            var meanfile = new OpenFileDialog() { Title = "Select mean image to be calculated" };
            if (meanfile.ShowDialog() == DialogResult.OK)
                meanpath = meanfile.FileName;
            else
                return new int[0, 0];

            // Select std image path
            var stdfile = new OpenFileDialog() { Title = "Select std image to be calculated" };
            if (stdfile.ShowDialog() == DialogResult.OK)
                stdpath = stdfile.FileName;
            else
                return new int[0, 0];

            // Select save path
            fbd = new FolderBrowserDialog() { Description = "Select the directory to save results" };
            if (fbd.ShowDialog() == DialogResult.OK)
                savepath = fbd.SelectedPath;
            else
                return new int[0, 0];

            // Requires mean and std images from surface volume
            Parameters param = new Parameters() { Meanstd = true, ImageType = ".dat" };
            // Calculate single LBP image
            RunLBP run = new RunLBP()
            {
                path = path,
                savepath = savepath,
                param = param,

                meanpath = meanpath,
                stdpath = stdpath
            };

            // Calculate single image
            run.CalculateSingle();

            // Calculate batch
            //run.CalculateBatch();

            int[,] features = run.features;
            return features;
        }

        /// <summary>
        /// Computes mean along each column of the array
        /// and subtracts it along the columns.
        /// </summary>
        /// <param name="array">Array to be calculated.</param>
        /// <returns>Subtracted array.</returns>
        public static double[,] SubtractMean(double[,] array)
        {
            int w = array.GetLength(0), l = array.GetLength(1);
            double[,] dataAdjust = new double[0, 0];
            double[] means = new double[w];

            for (int i = 0; i < w; i++)
            {
                double[] vector = 
                    LBPLibrary.Functions.ArrayToVector(
                    LBPLibrary.Functions.GetSubMatrix(array, i, i, 0, l - 1));

                means[i] = vector.Average();
                vector = Elementwise.Subtract(vector, means[i]);
                

                dataAdjust = Matrix.Concatenate(dataAdjust, vector);
            }

            return dataAdjust;
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
    }
}
