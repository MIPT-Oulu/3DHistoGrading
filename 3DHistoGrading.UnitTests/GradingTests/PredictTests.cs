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

        [Fact]
        public void Subtractmean_SubtractFromTestImage_ReturnsCorrectvalues()
        {
            testImg.New("Quarters", new int[] { 6, 6 });

            double[,] imageAdjust = Processing.SubtractMean(testImg.Image.ToDouble());

            double[,] refArray = new double[6, 6] // Here, actually columns are written out
                {{ -1, -1, -1, -1, -1, -1},
                { -1, -1, -1, -1, -1, -1},
                { -1, -1, -1, -1, -1, -1},
                { 1, 1, 1, 1, 1, 1},
                { 1, 1, 1, 1, 1, 1},
                { 1, 1, 1, 1, 1, 1} };
            Assert.Equal(refArray, imageAdjust);
        }

        [Fact]
        public void LBP_FeaturesFromTestArray_EqualsReference()
        {
            testImg.New("Quarters", new int[] { 40, 40 });

            int[,] features = Grading.LBP(testImg.Image.ToDouble());

            int[,] refArray = new int[1, 32] // Here, actually columns are written out
                {{ 162, 162, 0, 0, 0, 14, 4, 14, 0, 0, 0, 292, 0, 10, 18, 94, 56, 94, 18, 10, 0, 24, 5, 36, 9, 12, 16, 12, 9, 37, 5, 183} }
                .Transpose();
            Assert.Equal(refArray, features);
        }

        //[Fact]
        //public void Predict_DefaultModelAndFeatures_EqualsReference()
        //{
        //    // Change current directory
        //    Directory.SetCurrentDirectory(Directory.GetCurrentDirectory() + @"\dll");
        //    // Grading variables
        //    Model model = new Model();
        //    int[,] features = new int[0, 0];
        //    // Load LBP features
        //    string filename = new DirectoryInfo(Directory.GetCurrentDirectory()) // Get current directory
        //        .Parent.Parent.Parent.Parent.FullName + @"\Default\sample_features.csv"; // Move to correct location and add file name
        //    features = LBPLibrary.Functions
        //        .ReadCSV(filename)
        //        .ToInt32();
        //    // Load model
        //    string state = Grading.LoadModel(ref model);

        //    // Predict grade
        //    state = Grading.Predict(model, ref features);

        //    Assert.Equal("Sum of differences between pretrained model and actual grade: 12.484", state);
        //}

        //[Fact]
        //public void Predict_DefaultModelNoFeatures_EqualsReference()
        //{
        //    // Change current directory
        //    //Directory.SetCurrentDirectory(Directory.GetCurrentDirectory() + @"\dll");
        //    // Grading variables
        //    Model model = new Model();
        //    int[,] features = new int[0, 0];
        //    // Load model
        //    string state = Grading.LoadModel(ref model);

        //    // Predict grade
        //    state = Grading.Predict(model, ref features);

        //    Assert.Equal("Sum of differences between pretrained model and actual grade: 12.484", state);
        //}
    }
}
