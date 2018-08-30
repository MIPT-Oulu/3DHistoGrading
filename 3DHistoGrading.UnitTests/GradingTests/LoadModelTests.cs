using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using LBPLibrary;
using HistoGrading.Components;

namespace _3DHistoGrading.UnitTests
{
    public class LoadModelTests
    {
        [Fact]
        public void Load_NoModel_ReturnsError()
        {
            // Grading variables
            Model model = new Model();
            int[,] features = new int[0, 0];

            string state = Grading.LoadModel(ref model);

            Assert.Equal("Model loaded", state);
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
