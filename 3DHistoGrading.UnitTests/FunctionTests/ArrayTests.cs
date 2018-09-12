using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using HistoGrading.Components;
using System.Windows.Forms;
using Accord.Math;
using System.Drawing;

using Kitware.VTK;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace _3DHistoGrading.UnitTests.FunctionTests
{
    public class ArrayTests
    {
        TestImage testImg = new TestImage(); // Initialize testimage function

        [Fact]
        public void VectorToVolume_Samplevector_Returns3DVolume()
        {
            testImg.New("Quarters", new int[] { 27, 27 });
            float[] vector = LBPLibrary.Functions.ArrayToVector(testImg.Image);

            float[,,] volume = DataTypes.VectorToVolume(vector, new int[] { 3, 3, 3 });

            float[,,] refArray = new float[3, 3, 3] // Here, actually columns are written out
                { {{ -1, -1, -1},
                { 1, 1, 1},
                { 1, 1, 1,} } ,
                {{ -1, -1, -1},
                { 1, 1, 1},
                { 1, 1, 1,} } ,
                {{ -1, -1, -1},
                { 1, 1, 1},
                { 1, 1, 1,} }};
            Assert.Equal(refArray, volume);
        }
    }
}
