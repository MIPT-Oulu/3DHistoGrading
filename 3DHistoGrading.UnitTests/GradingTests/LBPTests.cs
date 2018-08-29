using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using LBPLibrary;
using HistoGrading.Components;
using _3DHistoGrading.UnitTests;

namespace _3DHistoGrading.UnitTests
{
    public class LBPTests
    {
        TestImage testImg = new TestImage(); // Initialize testimage function
        BinaryWriterApp lbpreader = new BinaryWriterApp(Directory.GetCurrentDirectory() + @"\Test.dat");
        string load = @"C:\temp\test\load";
        string save = @"C:\temp\test\save";

        public LBPTests()
        {
            Directory.CreateDirectory(@"C:\temp\test\load");
            Directory.CreateDirectory(@"C:\temp\test\save");
        }

        [Fact]
        public void RunLBP_DefaultInput_OutputsDefaultParameters()
        {
            var param = new Parameters();
            testImg.New("Quarters", new int[] { 28, 28 });

            //Grading.LoadModel();

        }

        [Fact]
        public void Model_DefaultInput_NullParameters()
        {
            Model model = new Model();

            Assert.Null(model.eigenVectors);
            Assert.Null(model.singularValues);
            Assert.Equal(0, model.nComp);
            Assert.Null(model.weights);
        }
    }
}
