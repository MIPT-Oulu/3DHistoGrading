using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LBPLibrary;
using HistoGrading.Components;
using _3DHistoGrading.UnitTests;
using HistoGrading;
using System.Windows.Forms;

using Xunit;
//using NUnit.Framework;
//using NUnit.Extensions.Forms;

namespace _3DHistoGrading.UnitTests
{
    public class PredictTests
    {
        TestImage testImg = new TestImage(); // Initialize testimage function
        BinaryWriterApp lbpreader = new BinaryWriterApp(Directory.GetCurrentDirectory() + @"\Test.dat");
        string load = @"C:\temp\test\load";
        string save = @"C:\temp\test\save";

        public PredictTests()
        {
            Directory.CreateDirectory(@"C:\temp\test\load");
            Directory.CreateDirectory(@"C:\temp\test\save");
        }

        [Fact]
        public void Predict_NoModel_ReturnsError()
        {
            // Grading variables
            Model model = new Model();
            int[,] features = new int[0, 0];

            string state = Grading.Predict(model, ref features);

            Assert.Equal("Model not loaded", state);
        }
    }
}
