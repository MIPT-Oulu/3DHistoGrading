using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace HistoGrading
{
    public partial class GradingForm : Form
    {
        public GradingForm()
        {
            InitializeComponent();
            Refresh();
        }

        public void UpdateModel()
        {
            progressBar1.Value = 5;
            progressLabel.Text = "Progress: Default grading model loaded.";
            Refresh();
        }

        public void UpdateSurface()
        {
            progressBar1.Value = 40;
            progressLabel.Text = "Progress: Cartilage surface and volume extracted.";
            Refresh();
        }

        public void UpdateMean(Bitmap meanIm, Bitmap stdIm)
        {
            meanPicture.SizeMode = PictureBoxSizeMode.StretchImage;
            meanPicture.Image = meanIm;
            stdPicture.SizeMode = PictureBoxSizeMode.StretchImage;
            stdPicture.Image = stdIm;
            progressBar1.Value = 60;
            progressLabel.Text = "Progress: Mean and Standard deviation images calculated.";
            Refresh();
        }

        public void UpdateParameters(LBPLibrary.Parameters param)
        {
            string paramText =
                "LBP parameters\n\n" +
                "Small radius: " + param.Radius.ToString() + "\n" +
                "Large radius: " + param.LargeRadius.ToString() + "\n" +
                "Neighbours: " + param.Neighbours.ToString() + "\n" +
                "\nFilters\n\n" +
                "Center: " + param.W_c.ToString() + "\n" +
                "Small: " + param.W_r[0].ToString() + "\n" +
                "Large: " + param.W_r[1].ToString() + "\n" +
                "\nStandardization\n\n" +
                "Kernel sizes: " + param.W_stand[0].ToString() + ", " + param.W_stand[1].ToString() + "\n" +
                "Standard deviations: " + param.W_stand[2].ToString() + ", " + param.W_stand[3].ToString() + "\n";

            parameterTip.SetToolTip(parameterLabel, paramText);
            Refresh();
        }

        public void UpdateLBP(Bitmap small, Bitmap large, Bitmap radial)
        {
            smallPicture.SizeMode = PictureBoxSizeMode.StretchImage;
            smallPicture.Image = small;
            largePicture.SizeMode = PictureBoxSizeMode.StretchImage;
            largePicture.Image = large;
            radialPicture.SizeMode = PictureBoxSizeMode.StretchImage;
            radialPicture.Image = radial;
            progressBar1.Value = 90;
            progressLabel.Text = "Progress: LBP features calculated.";
            Refresh();
        }

        public void UpdateGrade(string grade)
        {
            progressBar1.Value = 100;
            progressLabel.Text = "Done: Grade estimated (" + grade + ").";
            Refresh();
        }
    }
}
