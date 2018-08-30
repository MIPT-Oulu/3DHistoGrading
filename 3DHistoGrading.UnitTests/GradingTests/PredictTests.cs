using System;
using LBPLibrary;
using HistoGrading.Components;
using System.IO;

using Xunit;
using Accord.Math;

namespace _3DHistoGrading.UnitTests
{
    public class PredictTests
    {
        TestImage testImg = new TestImage(); // Initialize testimage function
        //BinaryWriterApp lbpreader = new BinaryWriterApp(Directory.GetCurrentDirectory() + @"\Test.dat");
        //string load = @"C:\temp\test\load";
        //string save = @"C:\temp\test\save";

        //public PredictTests()
        //{
        //    Directory.CreateDirectory(@"C:\temp\test\load");
        //    Directory.CreateDirectory(@"C:\temp\test\save");
        //}

        [Fact]
        public void Predict_NoModel_ReturnsError()
        {
            // Grading variables
            Model model = new Model();
            int[,] features = new int[0, 0];

            string state = Grading.Predict(model, ref features);

            Assert.Equal("Model not loaded", state);
        }

        [Fact]
        public void Predict_DefaultModelAndFeatures_EqualsReference()
        {
            // Change current directory
            Directory.SetCurrentDirectory(Directory.GetCurrentDirectory() + @"\dll");
            // Grading variables
            Model model = new Model();
            int[,] features = new int[0, 0];
            // Load LBP features
            string filename = new DirectoryInfo(Directory.GetCurrentDirectory()) // Get current directory
                .Parent.Parent.Parent.Parent.FullName + @"\Default\sample_features.csv"; // Move to correct location and add file name
            features = Functions
                .ReadCSV(filename)
                .ToInt32();
            // Load model
            string state = Grading.LoadModel(ref model);

            // Predict grade
            state = Grading.Predict(model, ref features);

            Assert.Equal("Sum of differences between pretrained model and actual grade: 12,484", state);
        }

        [Fact]
        public void Subtractmean_SubtractFromTestImage_ReturnsCorrectvalues()
        {
            testImg.New("Quarters", new int[] { 6, 6 });

            double[,] imageAdjust = Grading.SubtractMean(testImg.Image.ToDouble());

            TestImage.DisplayArray(imageAdjust);
            TestImage.DisplayArray(testImg.Image);
            double[,] refArray = new double[6, 6] // Here, actually columns are written out
                {{ -1, -1, -1, -1, -1, -1},
                { -1, -1, -1, -1, -1, -1},
                { -1, -1, -1, -1, -1, -1},
                { 1, 1, 1, 1, 1, 1},
                { 1, 1, 1, 1, 1, 1},
                { 1, 1, 1, 1, 1, 1} };
            Console.WriteLine("test2");
            Assert.Equal(refArray, imageAdjust);
        }
    }
}
