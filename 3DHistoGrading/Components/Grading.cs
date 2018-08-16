using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LBPLibrary;
using System.Windows.Forms;

using Accord.Math;

namespace HistoGrading.Components
{
    public class Grading
    {
        // Load model
        // Default model file should be loaded
        public static void LoadModel(ref Model mod)
        {
            // Path to model (weights.dat)
            string filename =
                Directory.GetParent(Directory.GetParent(Directory.GetParent( // Move up 3 directories to project folder
                Directory.GetCurrentDirectory()
                ).FullName).FullName).FullName + @"\weights.dat"; // Get path from DirectoryInfo object

            // Read weights from .dat file
            var reader = new BinaryWriterApp(filename);
            reader.ReadWeights();

            // Update model variables
            mod.nComp = reader.ncomp;
            mod.eigenVectors = reader.eigenVectors;
            mod.singularValues = reader.singularValues;
            mod.weights = reader.weights;
        }

        // Predict
        // Prediction using the model
        public static double Predict(Model mod, ref int[,] features)
        {
            // LBP features
            //features = LBP();

            // PCA
            double[,] PCA = features
                .Transpose()
                .Dot(mod.eigenVectors.ToDouble());
            double mean = LBPLibrary.Functions.Mean(PCA);
            PCA = PCA.Subtract(mean);
            // Regression
            double[] grade = PCA.Dot(mod.weights).Abs();
            return grade[0];
        }

        public static int[,] LBP()
        {

            // Select load path
            string path = null;
            var fbd = new FolderBrowserDialog() { Description = "Select the directory to load images" };
            if (fbd.ShowDialog() == DialogResult.OK)
                path = fbd.SelectedPath;
            else
            {
                Console.WriteLine("No directory selected.\n");
                return new int[0, 0];
            }

            string meanpath = null, stdpath = null, savepath = null;
            //// Select mean image path
            //var meanfile = new OpenFileDialog() { Title = "Select mean image to be calculated" };
            //if (meanfile.ShowDialog() == DialogResult.OK)
            //    meanpath = meanfile.FileName;
            //else
            //{
            //    Console.WriteLine("No directory selected.\n");
            //    return new int[0, 0];
            //}

            //// Select std image path
            //var stdfile = new OpenFileDialog() { Title = "Select std image to be calculated" };
            //if (stdfile.ShowDialog() == DialogResult.OK)
            //    stdpath = stdfile.FileName;
            //else
            //{
            //    Console.WriteLine("No directory selected.\n");
            //    return new int[0, 0];
            //}

            //// Select save path
            //var fbd = new FolderBrowserDialog() { Description = "Select the directory to save results" };
            //if (fbd.ShowDialog() == DialogResult.OK)
            //    savepath = fbd.SelectedPath;
            //else
            //{
            //    Console.WriteLine("No save path selected.\n");
            //    return new int[0, 0];
            //}

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

            //// Calculate single image
            //run.CalculateSingle();

            // Calculate batch
            run.CalculateBatch();

            return run.features;
        }
    }

    public class Model
    {
        public int nComp;
        public float[,] eigenVectors;
        public float[] singularValues;
        public double[] weights;
    }
}
