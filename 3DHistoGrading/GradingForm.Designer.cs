namespace HistoGrading
{
    partial class GradingForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.progressBar1 = new System.Windows.Forms.ProgressBar();
            this.progressLabel = new System.Windows.Forms.Label();
            this.meanPicture = new System.Windows.Forms.PictureBox();
            this.stdPicture = new System.Windows.Forms.PictureBox();
            this.smallPicture = new System.Windows.Forms.PictureBox();
            this.largePicture = new System.Windows.Forms.PictureBox();
            this.radialPicture = new System.Windows.Forms.PictureBox();
            this.gradeLabel = new System.Windows.Forms.Label();
            this.parameterLabel = new System.Windows.Forms.Label();
            this.parameterTip = new System.Windows.Forms.ToolTip(this.components);
            this.meanstdLabel = new System.Windows.Forms.Label();
            this.meanstdPicture = new System.Windows.Forms.PictureBox();
            ((System.ComponentModel.ISupportInitialize)(this.meanPicture)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.stdPicture)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.smallPicture)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.largePicture)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.radialPicture)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.meanstdPicture)).BeginInit();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(115, 161);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(65, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "Mean image";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(336, 161);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(127, 13);
            this.label2.TabIndex = 2;
            this.label2.Text = "Standard deviation image";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(376, 499);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(57, 13);
            this.label3.TabIndex = 7;
            this.label3.Text = "Large LBP";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(115, 499);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(55, 13);
            this.label4.TabIndex = 8;
            this.label4.Text = "Small LBP";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(636, 499);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(60, 13);
            this.label5.TabIndex = 9;
            this.label5.Text = "Radial LBP";
            // 
            // progressBar1
            // 
            this.progressBar1.Location = new System.Drawing.Point(459, 80);
            this.progressBar1.Name = "progressBar1";
            this.progressBar1.Size = new System.Drawing.Size(300, 23);
            this.progressBar1.TabIndex = 10;
            // 
            // progressLabel
            // 
            this.progressLabel.AutoSize = true;
            this.progressLabel.Location = new System.Drawing.Point(456, 64);
            this.progressLabel.Name = "progressLabel";
            this.progressLabel.Size = new System.Drawing.Size(54, 13);
            this.progressLabel.TabIndex = 11;
            this.progressLabel.Text = "Progress: ";
            // 
            // meanPicture
            // 
            this.meanPicture.Location = new System.Drawing.Point(38, 177);
            this.meanPicture.Name = "meanPicture";
            this.meanPicture.Size = new System.Drawing.Size(200, 200);
            this.meanPicture.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
            this.meanPicture.TabIndex = 12;
            this.meanPicture.TabStop = false;
            // 
            // stdPicture
            // 
            this.stdPicture.Location = new System.Drawing.Point(299, 177);
            this.stdPicture.Name = "stdPicture";
            this.stdPicture.Size = new System.Drawing.Size(200, 200);
            this.stdPicture.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
            this.stdPicture.TabIndex = 13;
            this.stdPicture.TabStop = false;
            // 
            // smallPicture
            // 
            this.smallPicture.Location = new System.Drawing.Point(38, 525);
            this.smallPicture.Name = "smallPicture";
            this.smallPicture.Size = new System.Drawing.Size(200, 200);
            this.smallPicture.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
            this.smallPicture.TabIndex = 14;
            this.smallPicture.TabStop = false;
            // 
            // largePicture
            // 
            this.largePicture.Location = new System.Drawing.Point(299, 525);
            this.largePicture.Name = "largePicture";
            this.largePicture.Size = new System.Drawing.Size(200, 200);
            this.largePicture.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
            this.largePicture.TabIndex = 15;
            this.largePicture.TabStop = false;
            // 
            // radialPicture
            // 
            this.radialPicture.Location = new System.Drawing.Point(559, 525);
            this.radialPicture.Name = "radialPicture";
            this.radialPicture.Size = new System.Drawing.Size(200, 200);
            this.radialPicture.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
            this.radialPicture.TabIndex = 16;
            this.radialPicture.TabStop = false;
            // 
            // gradeLabel
            // 
            this.gradeLabel.AutoSize = true;
            this.gradeLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 20F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.gradeLabel.Location = new System.Drawing.Point(32, 49);
            this.gradeLabel.Name = "gradeLabel";
            this.gradeLabel.Size = new System.Drawing.Size(110, 31);
            this.gradeLabel.TabIndex = 17;
            this.gradeLabel.Text = "Grading";
            // 
            // parameterLabel
            // 
            this.parameterLabel.AutoSize = true;
            this.parameterLabel.Location = new System.Drawing.Point(35, 115);
            this.parameterLabel.Name = "parameterLabel";
            this.parameterLabel.Size = new System.Drawing.Size(154, 13);
            this.parameterLabel.TabIndex = 18;
            this.parameterLabel.Text = "Parameters (hover mouse over)";
            this.parameterTip.SetToolTip(this.parameterLabel, "Parameters:");
            // 
            // parameterTip
            // 
            this.parameterTip.AutoPopDelay = 15000;
            this.parameterTip.InitialDelay = 500;
            this.parameterTip.ReshowDelay = 100;
            // 
            // meanstdLabel
            // 
            this.meanstdLabel.AutoSize = true;
            this.meanstdLabel.Location = new System.Drawing.Point(634, 161);
            this.meanstdLabel.Name = "meanstdLabel";
            this.meanstdLabel.Size = new System.Drawing.Size(62, 13);
            this.meanstdLabel.TabIndex = 19;
            this.meanstdLabel.Text = "Mean + Std";
            // 
            // meanstdPicture
            // 
            this.meanstdPicture.Location = new System.Drawing.Point(559, 177);
            this.meanstdPicture.Name = "meanstdPicture";
            this.meanstdPicture.Size = new System.Drawing.Size(200, 200);
            this.meanstdPicture.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
            this.meanstdPicture.TabIndex = 20;
            this.meanstdPicture.TabStop = false;
            // 
            // GradingForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 737);
            this.Controls.Add(this.meanstdPicture);
            this.Controls.Add(this.meanstdLabel);
            this.Controls.Add(this.parameterLabel);
            this.Controls.Add(this.gradeLabel);
            this.Controls.Add(this.radialPicture);
            this.Controls.Add(this.largePicture);
            this.Controls.Add(this.smallPicture);
            this.Controls.Add(this.stdPicture);
            this.Controls.Add(this.meanPicture);
            this.Controls.Add(this.progressLabel);
            this.Controls.Add(this.progressBar1);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Name = "GradingForm";
            this.Text = "GradingForm";
            ((System.ComponentModel.ISupportInitialize)(this.meanPicture)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.stdPicture)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.smallPicture)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.largePicture)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.radialPicture)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.meanstdPicture)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.ProgressBar progressBar1;
        private System.Windows.Forms.Label progressLabel;
        private System.Windows.Forms.PictureBox meanPicture;
        private System.Windows.Forms.PictureBox stdPicture;
        private System.Windows.Forms.PictureBox smallPicture;
        private System.Windows.Forms.PictureBox largePicture;
        private System.Windows.Forms.PictureBox radialPicture;
        private System.Windows.Forms.Label gradeLabel;
        private System.Windows.Forms.Label parameterLabel;
        private System.Windows.Forms.ToolTip parameterTip;
        private System.Windows.Forms.Label meanstdLabel;
        private System.Windows.Forms.PictureBox meanstdPicture;
    }
}