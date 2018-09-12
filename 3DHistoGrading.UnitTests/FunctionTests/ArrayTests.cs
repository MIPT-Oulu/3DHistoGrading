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
            testImg.New("Running Numbers", new int[] { 9, 3 });
            float[] vector = LBPLibrary.Functions.ArrayToVector(testImg.Image);

            float[,,] volume = DataTypes.VectorToVolume(vector, new int[] { 3, 3, 3 });

            float[,,] refArray = new float[3, 3, 3] // Here, actually columns are written out
                { {{ 0, 1, 2},
                { 1, 2, 3},
                { 2, 3, 4,} } ,
                {{ 3, 4, 5},
                { 4, 5, 6},
                { 5, 6, 7,} } ,
                {{ 6, 7, 8},
                { 7, 8, 9},
                { 8, 9, 10,} }};
            Assert.Equal(refArray, volume);
        }

        [Fact]
        public void VolumeToSlice_SampleVolume_Returns2DSlice()
        {
            testImg.New("Quarters", new int[] { 9, 3 });
            float[] vector = LBPLibrary.Functions.ArrayToVector(testImg.Image);

            float[,,] volume = DataTypes.VectorToVolume(vector, new int[] { 3, 3, 3 });
            float[,] slice = DataTypes.VolumeToSlice(volume, 2, 1);

            float[,] refArray = new float[3, 3] // Here, actually columns are written out
                { { 1, 3, 3},
                { 2, 4, 4},
                { 2, 4, 4,} };
            Assert.Equal(refArray, slice);
        }
    }
}
