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
using NUnit.Framework;
using NUnit.Extensions.Forms;

using Kitware.VTK;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace _3DHistoGrading.UnitTests.FunctionTests
{
    [TestFixture]
    public class LoadTests
    {
        TestImage testImg = new TestImage(); // Initialize testimage function

        [Fact]
        public void Directory_NoOK_ReturnsEmptyString()
        {
            string selectedPath = "Testpath";
            var check = new DialogResult();
            

            string result = Functions.DirectoryResult(selectedPath, check);

            Xunit.Assert.Empty(result);
        }
    }
}
