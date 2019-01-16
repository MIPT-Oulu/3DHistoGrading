using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Threading;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace HistoGrading
{
    /// <summary>
    /// Form that visualizes results of LBP and grading.
    /// </summary>
    public partial class GradingForm : Form
    {
        // Grading form should update with its own thread in the future.

        /// <summary>
        /// Form that displays results of sample grading.
        /// </summary>
        public GradingForm()
        {
            InitializeComponent();
            Refresh();
        }

        /// <summary>
        /// Update that model is loaded.
        /// </summary>
        public void UpdateModel(string zonetext)
        {
            progressBar1.Value = 10;
            progressLabel.Text = "Progress: Default grading model loaded.";
            gradeLabel.Text = zonetext;
            UseWaitCursor = true;
            Refresh();
        }

        /// <summary>
        /// Update after surface volume is extracted.
        /// </summary>
        public void UpdateSurface()
        {
            progressBar1.Value = 40;
            progressLabel.Text = "Progress: Sample volume extracted.";
            Refresh();
        }

        /// <summary>
        /// Update after mean and std images are calculated
        /// </summary>
        /// <param name="meanIm">Mean image.</param>
        /// <param name="stdIm">Standard deviation image.</param>
        /// <param name="meanstdIm">Mean + standard deviation image.</param>
        public void UpdateMean(Bitmap meanIm, Bitmap stdIm, Bitmap meanstdIm)
        {
            progressBar1.Value = 60;
            progressLabel.Text = "Progress: Mean and Standard deviation images calculated.";
            meanPicture.SizeMode = PictureBoxSizeMode.StretchImage;
            meanPicture.Image = meanIm;
            stdPicture.SizeMode = PictureBoxSizeMode.StretchImage;
            stdPicture.Image = stdIm;
            meanstdPicture.SizeMode = PictureBoxSizeMode.StretchImage;
            meanstdPicture.Image = meanstdIm;
            Refresh();
        }

        /// <summary>
        /// Update used grading parameters.
        /// </summary>
        /// <param name="param">Class including LBP variables.</param>
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

        /// <summary>
        /// Update when LBP images are calculated.
        /// </summary>
        /// <param name="small">LBP image with small radius.</param>
        /// <param name="large">LBP image with large radius.</param>
        /// <param name="radial">LBP image with small radius subtracted from large radius.</param>
        public void UpdateLBP(Bitmap small, Bitmap large, Bitmap radial)
        {
            progressBar1.Value = 90;
            progressLabel.Text = "Progress: LBP features calculated.";
            smallPicture.SizeMode = PictureBoxSizeMode.StretchImage;
            smallPicture.Image = small;
            largePicture.SizeMode = PictureBoxSizeMode.StretchImage;
            largePicture.Image = large;
            radialPicture.SizeMode = PictureBoxSizeMode.StretchImage;
            radialPicture.Image = radial;
            Refresh();
        }

        /// <summary>
        /// Update final estimated grade.
        /// </summary>
        /// <param name="grade">Estimated grade.</param>
        public void UpdateGrade(string grade)
        {
            progressBar1.Value = 100;
            progressLabel.Text = "Done: Grade estimated (" + grade + ").";
            UseWaitCursor = false;
            Refresh();
        }

        private void label2_Click(object sender, EventArgs e)
        {

        }
    }
}
