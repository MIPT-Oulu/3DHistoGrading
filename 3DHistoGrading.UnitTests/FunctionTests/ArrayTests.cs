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

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        [InlineData(2)]
        public void VolumeToSlice_SampleVolume_Returns2DSlice(int axis)
        {
            testImg.New("Quarters", new int[] { 9, 3 });
            float[] vector = LBPLibrary.Functions.ArrayToVector(testImg.Image);

            float[,,] volume = DataTypes.VectorToVolume(vector, new int[] { 3, 3, 3 });
            float[,] slice = DataTypes.VolumeToSlice(volume, 2, axis);

            float[,] refArray = new float[3, 3];
            switch (axis)
            {
                case 0:
                    refArray = new float[3, 3] // Here, actually columns are written out
                        { { 2, 4, 4},
                        { 2, 4, 4},
                        { 2, 4, 4,} };
                    break;
                case 1:
                    refArray = new float[3, 3] // Here, actually columns are written out
                        { { 1, 3, 3},
                        { 2, 4, 4},
                        { 2, 4, 4,} };
                    break;
                case 2:
                    refArray = new float[3, 3] // Here, actually columns are written out
                        { { 3, 3, 3},
                        { 3, 4, 4},
                        { 4, 4, 4,} };
                    break;
                default:
                    break;
            }
            Assert.Equal(refArray, slice);
        }

        [Fact]
        public void VolumeToSlice_WrongAxis_ThrowsException()
        {
            testImg.New("Quarters", new int[] { 9, 3 });
            float[] vector = LBPLibrary.Functions.ArrayToVector(testImg.Image);

            float[,,] volume = DataTypes.VectorToVolume(vector, new int[] { 3, 3, 3 });

            Exception ex = Assert.Throws<Exception>(
                delegate { float[,] slice = DataTypes.VolumeToSlice(volume, 2, 4); });
            Assert.Equal("Invalid axis given. Give axis as an integer between 0 and 2.", ex.Message);

        }

        [Fact]
        public void ByteToFloat_NoStandard_EqualsReference()
        {
            testImg.New("Quarters", new int[] { 2, 2 });
            byte[] vector = LBPLibrary.Functions.ArrayToVector(testImg.Image).ToByte();

            float[] floatVector = DataTypes.byteToFloat(vector);

            float[] refArray = new float[] { 1, 2, 3, 4 };
            Assert.Equal(refArray, floatVector);
        }

        [Fact]
        public void ByteToFloat_NoStandardDev_EqualsReference()
        {
            testImg.New("Quarters", new int[] { 2, 2 });
            byte[] vector = LBPLibrary.Functions.ArrayToVector(testImg.Image).ToByte();

            float[] floatVector = DataTypes.byteToFloat(vector, 2.5F);

            float[] refArray = new float[] { -1.5F, -0.5F, 0.5F, 1.5F };
            Assert.Equal(refArray, floatVector);
        }

        [Fact]
        public void ByteToFloat_AllInputs_EqualsReference()
        {
            testImg.New("Quarters", new int[] { 2, 2 });
            byte[] vector = LBPLibrary.Functions.ArrayToVector(testImg.Image).ToByte();

            float[] floatVector = DataTypes.byteToFloat(vector, 2.5F, 3);

            float[] refArray = new float[] { -0.5F, -0.166666672F, 0.166666672F, 0.5F };
            Assert.Equal(refArray, floatVector);
        }
    }
}
