using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LBPLibrary;
using System.IO;
using Accord.Math;
using OpenCvSharp;

namespace OrientSample
{
    class Program
    {
        static void Main(string[] args)
        {
            // Load slice
            string path = new DirectoryInfo(Directory.GetCurrentDirectory()).Parent.Parent.FullName
                + @"\reference.png";
            Mat im = new Mat(path, ImreadModes.GrayScale);
            Orient.ShowImage(im);

            // Threshold
            var mask = new Mat();
            Cv2.Threshold(im, mask, 80.0, 1.0, ThresholdTypes.Binary);
            im = im.Mul(mask);
            Orient.ShowImage(im);

            // Linear fit for edge
            Orient.LinearFit(im, out Line2D line, out Mat imline);
            Orient.ShowImage(imline);

            // Get angle
            double angle = (Math.Atan(line.Vy / line.Vx)) * 180 / Math.PI; // Angle from x-axis
        }
    }
}
