using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using System.IO;

using LBPLibrary;
using Accord.Math;
using OpenCvSharp;


namespace OrientSample
{
    class Orient
    {
        public static void ShowImage(Mat im)
        {
            using (var window = new Window("window", image: im, flags: WindowMode.AutoSize))
            {
                Cv2.WaitKey();
            }
        }

        /// <summary>
        /// Fits a line to sample from left side. Uses 21 points from 1/3 to 2/3 of sample height.
        /// </summary>
        /// <param name="im">Sample image.</param>
        /// <param name="line">Fitted line parameters.</param>
        public static void LinearFit(Mat im, out Line2D line, out Mat imline)
        {
            // Get fit range
            var size = im.Size();
            int[] range = new int[] { im.Size().Height / 3, im.Size().Height * 2 / 3 };

            // Get points for fit
            List<Point2f> points = new List<Point2f>();
            for (int j = range[0]; j <= range[1]; j+= (int)Math.Floor((range[1] - range[0]) / 20.0))
            {
                for (int i = 0; i < im.Size().Width; i++)
                {
                    if (im.At<float>(i, j) > 0)
                    {
                        points.Add(new Point2f(i, j));
                        break;
                    }
                }
            }

            // Linear fit
            line = Cv2.FitLine(points, DistanceTypes.L2, 0, 0.01, 0.01);

            // Draw line
            int lefty = (int)Math.Round((-line.X1 * line.Vy / line.Vx) + line.Y1);
            int righty = (int)Math.Round((size.Width - line.X1) * line.Vy / line.Vx + line.Y1);
            Cv2.Line(im, new Point(size.Width - 1, righty), new Point(0, lefty), new Scalar(255, 255, 255), 2);
            imline = im;
        }
    }
}
