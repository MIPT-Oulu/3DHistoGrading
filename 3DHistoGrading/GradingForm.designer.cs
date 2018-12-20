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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(GradingForm));
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
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            ((System.ComponentModel.ISupportInitialize)(this.meanPicture)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.stdPicture)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.smallPicture)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.largePicture)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.radialPicture)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.meanstdPicture)).BeginInit();
            this.tableLayoutPanel1.SuspendLayout();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Location = new System.Drawing.Point(109, 182);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(83, 16);
            this.label1.TabIndex = 1;
            this.label1.Text = "Mean image";
            // 
            // label2
            // 
            this.label2.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(352, 182);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(121, 16);
            this.label2.TabIndex = 2;
            this.label2.Text = "Standard deviation";
            this.label2.Click += new System.EventHandler(this.label2_Click);
            // 
            // label3
            // 
            this.label3.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.Location = new System.Drawing.Point(377, 513);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(71, 16);
            this.label3.TabIndex = 7;
            this.label3.Text = "Large LBP";
            // 
            // label4
            // 
            this.label4.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.label4.AutoSize = true;
            this.label4.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label4.Location = new System.Drawing.Point(116, 513);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(70, 16);
            this.label4.TabIndex = 8;
            this.label4.Text = "Small LBP";
            // 
            // label5
            // 
            this.label5.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.label5.AutoSize = true;
            this.label5.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label5.Location = new System.Drawing.Point(637, 513);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(76, 16);
            this.label5.TabIndex = 9;
            this.label5.Text = "Radial LBP";
            // 
            // progressBar1
            // 
            this.progressBar1.BackColor = System.Drawing.SystemColors.GrayText;
            this.progressBar1.Dock = System.Windows.Forms.DockStyle.Top;
            this.progressBar1.ForeColor = System.Drawing.SystemColors.ButtonShadow;
            this.progressBar1.Location = new System.Drawing.Point(23, 23);
            this.progressBar1.Name = "progressBar1";
            this.progressBar1.Size = new System.Drawing.Size(256, 34);
            this.progressBar1.TabIndex = 10;
            // 
            // progressLabel
            // 
            this.progressLabel.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.progressLabel.AutoSize = true;
            this.progressLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.progressLabel.Location = new System.Drawing.Point(23, 2);
            this.progressLabel.Name = "progressLabel";
            this.progressLabel.Size = new System.Drawing.Size(69, 16);
            this.progressLabel.TabIndex = 11;
            this.progressLabel.Text = "Progress: ";
            // 
            // meanPicture
            // 
            this.meanPicture.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.meanPicture.Location = new System.Drawing.Point(23, 201);
            this.meanPicture.Name = "meanPicture";
            this.meanPicture.Size = new System.Drawing.Size(256, 249);
            this.meanPicture.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
            this.meanPicture.TabIndex = 12;
            this.meanPicture.TabStop = false;
            // 
            // stdPicture
            // 
            this.stdPicture.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.stdPicture.Location = new System.Drawing.Point(285, 201);
            this.stdPicture.Name = "stdPicture";
            this.stdPicture.Size = new System.Drawing.Size(256, 249);
            this.stdPicture.TabIndex = 13;
            this.stdPicture.TabStop = false;
            // 
            // smallPicture
            // 
            this.smallPicture.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.smallPicture.Location = new System.Drawing.Point(23, 532);
            this.smallPicture.Name = "smallPicture";
            this.smallPicture.Size = new System.Drawing.Size(256, 249);
            this.smallPicture.TabIndex = 14;
            this.smallPicture.TabStop = false;
            // 
            // largePicture
            // 
            this.largePicture.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.largePicture.Location = new System.Drawing.Point(285, 532);
            this.largePicture.Name = "largePicture";
            this.largePicture.Size = new System.Drawing.Size(256, 249);
            this.largePicture.TabIndex = 15;
            this.largePicture.TabStop = false;
            // 
            // radialPicture
            // 
            this.radialPicture.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.radialPicture.Location = new System.Drawing.Point(547, 532);
            this.radialPicture.Name = "radialPicture";
            this.radialPicture.Size = new System.Drawing.Size(256, 249);
            this.radialPicture.TabIndex = 16;
            this.radialPicture.TabStop = false;
            // 
            // gradeLabel
            // 
            this.gradeLabel.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.gradeLabel.AutoSize = true;
            this.gradeLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 30F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.gradeLabel.Location = new System.Drawing.Point(285, 48);
            this.gradeLabel.Name = "gradeLabel";
            this.gradeLabel.Size = new System.Drawing.Size(223, 46);
            this.gradeLabel.TabIndex = 17;
            this.gradeLabel.Text = "Zone grade";
            // 
            // parameterLabel
            // 
            this.parameterLabel.AutoSize = true;
            this.parameterLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.parameterLabel.Location = new System.Drawing.Point(547, 20);
            this.parameterLabel.Name = "parameterLabel";
            this.parameterLabel.Size = new System.Drawing.Size(197, 16);
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
            this.meanstdLabel.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.meanstdLabel.AutoSize = true;
            this.meanstdLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.meanstdLabel.Location = new System.Drawing.Point(568, 182);
            this.meanstdLabel.Name = "meanstdLabel";
            this.meanstdLabel.Size = new System.Drawing.Size(214, 16);
            this.meanstdLabel.TabIndex = 19;
            this.meanstdLabel.Text = "Normalized Mean + Std (LBP input)";
            // 
            // meanstdPicture
            // 
            this.meanstdPicture.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.meanstdPicture.Location = new System.Drawing.Point(547, 201);
            this.meanstdPicture.Name = "meanstdPicture";
            this.meanstdPicture.Size = new System.Drawing.Size(256, 249);
            this.meanstdPicture.TabIndex = 20;
            this.meanstdPicture.TabStop = false;
            // 
            // tableLayoutPanel1
            // 
            this.tableLayoutPanel1.ColumnCount = 5;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 20F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 33.33333F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 33.33334F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 33.33334F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 22F));
            this.tableLayoutPanel1.Controls.Add(this.meanstdLabel, 3, 2);
            this.tableLayoutPanel1.Controls.Add(this.meanstdPicture, 3, 3);
            this.tableLayoutPanel1.Controls.Add(this.label2, 2, 2);
            this.tableLayoutPanel1.Controls.Add(this.label1, 1, 2);
            this.tableLayoutPanel1.Controls.Add(this.label4, 1, 4);
            this.tableLayoutPanel1.Controls.Add(this.meanPicture, 1, 3);
            this.tableLayoutPanel1.Controls.Add(this.smallPicture, 1, 5);
            this.tableLayoutPanel1.Controls.Add(this.largePicture, 2, 5);
            this.tableLayoutPanel1.Controls.Add(this.radialPicture, 3, 5);
            this.tableLayoutPanel1.Controls.Add(this.stdPicture, 2, 3);
            this.tableLayoutPanel1.Controls.Add(this.label5, 3, 4);
            this.tableLayoutPanel1.Controls.Add(this.label3, 2, 4);
            this.tableLayoutPanel1.Controls.Add(this.gradeLabel, 2, 1);
            this.tableLayoutPanel1.Controls.Add(this.progressBar1, 1, 1);
            this.tableLayoutPanel1.Controls.Add(this.parameterLabel, 3, 1);
            this.tableLayoutPanel1.Controls.Add(this.progressLabel, 1, 0);
            this.tableLayoutPanel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanel1.Location = new System.Drawing.Point(0, 0);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 7;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 20F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 13.33003F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 10.00003F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 33.3349F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 10.00014F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 33.3349F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 20F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(830, 807);
            this.tableLayoutPanel1.TabIndex = 21;
            // 
            // GradingForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.WindowFrame;
            this.ClientSize = new System.Drawing.Size(830, 807);
            this.Controls.Add(this.tableLayoutPanel1);
            this.ForeColor = System.Drawing.SystemColors.InactiveCaption;
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximumSize = new System.Drawing.Size(1080, 1080);
            this.MinimumSize = new System.Drawing.Size(100, 100);
            this.Name = "GradingForm";
            this.Text = "Grading";
            ((System.ComponentModel.ISupportInitialize)(this.meanPicture)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.stdPicture)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.smallPicture)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.largePicture)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.radialPicture)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.meanstdPicture)).EndInit();
            this.tableLayoutPanel1.ResumeLayout(false);
            this.tableLayoutPanel1.PerformLayout();
            this.ResumeLayout(false);

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
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
    }
}