namespace HistoGrading
{
    partial class MainForm
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainForm));
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.panel1 = new System.Windows.Forms.Panel();
            this.saveButton = new System.Windows.Forms.Button();
            this.rotate_button = new System.Windows.Forms.Button();
            this.getVoiButton = new System.Windows.Forms.Button();
            this.label3 = new System.Windows.Forms.Label();
            this.cropButton = new System.Windows.Forms.Button();
            this.segmentButton = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.viewLabel = new System.Windows.Forms.Label();
            this.predict = new System.Windows.Forms.Button();
            this.sagittalButton = new System.Windows.Forms.Button();
            this.coronalButton = new System.Windows.Forms.Button();
            this.transverseButton = new System.Windows.Forms.Button();
            this.volumeButton = new System.Windows.Forms.Button();
            this.resetButton = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.gmaxLabel = new System.Windows.Forms.Label();
            this.gminBar = new System.Windows.Forms.HScrollBar();
            this.gmaxBar = new System.Windows.Forms.HScrollBar();
            this.maskButton = new System.Windows.Forms.Button();
            this.fileButton = new System.Windows.Forms.Button();
            this.panel2 = new System.Windows.Forms.Panel();
            this.gradeLabel = new System.Windows.Forms.Label();
            this.sliceLabel = new System.Windows.Forms.Label();
            this.sliceBar = new System.Windows.Forms.VScrollBar();
            this.renderWindowControl = new Kitware.VTK.RenderWindowControl();
            this.panel3 = new System.Windows.Forms.Panel();
            this.label4 = new System.Windows.Forms.Label();
            this.mainProgress = new System.Windows.Forms.ProgressBar();
            this.maskLabel = new System.Windows.Forms.Label();
            this.fileLabel = new System.Windows.Forms.Label();
            this.fileDialog = new System.Windows.Forms.OpenFileDialog();
            this.tableLayoutPanel1.SuspendLayout();
            this.panel1.SuspendLayout();
            this.panel2.SuspendLayout();
            this.panel3.SuspendLayout();
            this.SuspendLayout();
            // 
            // tableLayoutPanel1
            // 
            this.tableLayoutPanel1.ColumnCount = 3;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 170F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 40F));
            this.tableLayoutPanel1.Controls.Add(this.panel1, 0, 1);
            this.tableLayoutPanel1.Controls.Add(this.panel2, 0, 0);
            this.tableLayoutPanel1.Controls.Add(this.sliceBar, 2, 1);
            this.tableLayoutPanel1.Controls.Add(this.renderWindowControl, 1, 1);
            this.tableLayoutPanel1.Controls.Add(this.panel3, 0, 2);
            this.tableLayoutPanel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanel1.Location = new System.Drawing.Point(0, 0);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 3;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 40F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 25F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(1604, 1500);
            this.tableLayoutPanel1.TabIndex = 0;
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.saveButton);
            this.panel1.Controls.Add(this.rotate_button);
            this.panel1.Controls.Add(this.getVoiButton);
            this.panel1.Controls.Add(this.label3);
            this.panel1.Controls.Add(this.cropButton);
            this.panel1.Controls.Add(this.segmentButton);
            this.panel1.Controls.Add(this.label2);
            this.panel1.Controls.Add(this.viewLabel);
            this.panel1.Controls.Add(this.predict);
            this.panel1.Controls.Add(this.sagittalButton);
            this.panel1.Controls.Add(this.coronalButton);
            this.panel1.Controls.Add(this.transverseButton);
            this.panel1.Controls.Add(this.volumeButton);
            this.panel1.Controls.Add(this.resetButton);
            this.panel1.Controls.Add(this.label1);
            this.panel1.Controls.Add(this.gmaxLabel);
            this.panel1.Controls.Add(this.gminBar);
            this.panel1.Controls.Add(this.gmaxBar);
            this.panel1.Controls.Add(this.maskButton);
            this.panel1.Controls.Add(this.fileButton);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel1.Location = new System.Drawing.Point(3, 45);
            this.panel1.Margin = new System.Windows.Forms.Padding(3, 5, 3, 5);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(164, 1254);
            this.panel1.TabIndex = 0;
            // 
            // saveButton
            // 
            this.saveButton.Cursor = System.Windows.Forms.Cursors.Hand;
            this.saveButton.Enabled = false;
            this.saveButton.ForeColor = System.Drawing.SystemColors.ControlText;
            this.saveButton.Location = new System.Drawing.Point(15, 1012);
            this.saveButton.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.saveButton.Name = "saveButton";
            this.saveButton.Size = new System.Drawing.Size(140, 75);
            this.saveButton.TabIndex = 16;
            this.saveButton.Text = "Save Results";
            this.saveButton.UseVisualStyleBackColor = true;
            this.saveButton.Click += new System.EventHandler(this.saveButton_Click);
            // 
            // rotate_button
            // 
            this.rotate_button.Cursor = System.Windows.Forms.Cursors.Hand;
            this.rotate_button.Enabled = false;
            this.rotate_button.ForeColor = System.Drawing.SystemColors.ControlText;
            this.rotate_button.Location = new System.Drawing.Point(15, 534);
            this.rotate_button.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.rotate_button.Name = "rotate_button";
            this.rotate_button.Size = new System.Drawing.Size(140, 62);
            this.rotate_button.TabIndex = 15;
            this.rotate_button.Text = "Auto Rotate";
            this.rotate_button.UseVisualStyleBackColor = true;
            this.rotate_button.Click += new System.EventHandler(this.rotate_button_Click);
            // 
            // getVoiButton
            // 
            this.getVoiButton.Cursor = System.Windows.Forms.Cursors.Hand;
            this.getVoiButton.Enabled = false;
            this.getVoiButton.ForeColor = System.Drawing.SystemColors.ControlText;
            this.getVoiButton.Location = new System.Drawing.Point(14, 757);
            this.getVoiButton.Name = "getVoiButton";
            this.getVoiButton.Size = new System.Drawing.Size(141, 65);
            this.getVoiButton.TabIndex = 14;
            this.getVoiButton.Text = "Get VOIs";
            this.getVoiButton.UseVisualStyleBackColor = true;
            this.getVoiButton.Click += new System.EventHandler(this.getVoiButton_Click);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(9, 489);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(87, 20);
            this.label3.TabIndex = 13;
            this.label3.Text = "Processing";
            // 
            // cropButton
            // 
            this.cropButton.Cursor = System.Windows.Forms.Cursors.Hand;
            this.cropButton.Enabled = false;
            this.cropButton.ForeColor = System.Drawing.SystemColors.ControlText;
            this.cropButton.Location = new System.Drawing.Point(14, 605);
            this.cropButton.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.cropButton.Name = "cropButton";
            this.cropButton.Size = new System.Drawing.Size(141, 74);
            this.cropButton.TabIndex = 12;
            this.cropButton.Text = "Crop Sample";
            this.cropButton.UseVisualStyleBackColor = true;
            this.cropButton.Click += new System.EventHandler(this.cropButton_Click);
            // 
            // segmentButton
            // 
            this.segmentButton.Cursor = System.Windows.Forms.Cursors.Hand;
            this.segmentButton.Enabled = false;
            this.segmentButton.ForeColor = System.Drawing.SystemColors.ControlText;
            this.segmentButton.Location = new System.Drawing.Point(15, 686);
            this.segmentButton.Name = "segmentButton";
            this.segmentButton.Size = new System.Drawing.Size(141, 65);
            this.segmentButton.TabIndex = 11;
            this.segmentButton.Text = "BCI Segmentation";
            this.segmentButton.UseVisualStyleBackColor = true;
            this.segmentButton.Click += new System.EventHandler(this.segmentButton_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(10, 858);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(66, 20);
            this.label2.TabIndex = 3;
            this.label2.Text = "Grading";
            // 
            // viewLabel
            // 
            this.viewLabel.AutoSize = true;
            this.viewLabel.Location = new System.Drawing.Point(10, 171);
            this.viewLabel.Name = "viewLabel";
            this.viewLabel.Size = new System.Drawing.Size(43, 20);
            this.viewLabel.TabIndex = 10;
            this.viewLabel.Text = "Viewing";
            // 
            // predict
            // 
            this.predict.Cursor = System.Windows.Forms.Cursors.Hand;
            this.predict.Enabled = false;
            this.predict.ForeColor = System.Drawing.SystemColors.ControlText;
            this.predict.Location = new System.Drawing.Point(14, 883);
            this.predict.Name = "predict";
            this.predict.Size = new System.Drawing.Size(141, 65);
            this.predict.TabIndex = 2;
            this.predict.Text = "Grade sample";
            this.predict.UseVisualStyleBackColor = true;
            this.predict.Click += new System.EventHandler(this.predict_Click);
            // 
            // sagittalButton
            // 
            this.sagittalButton.Cursor = System.Windows.Forms.Cursors.Hand;
            this.sagittalButton.Enabled = false;
            this.sagittalButton.ForeColor = System.Drawing.SystemColors.ControlText;
            this.sagittalButton.Location = new System.Drawing.Point(10, 418);
            this.sagittalButton.Margin = new System.Windows.Forms.Padding(3, 5, 3, 5);
            this.sagittalButton.Name = "sagittalButton";
            this.sagittalButton.Size = new System.Drawing.Size(148, 40);
            this.sagittalButton.TabIndex = 9;
            this.sagittalButton.Text = "Sagittal, YZ";
            this.sagittalButton.UseVisualStyleBackColor = true;
            this.sagittalButton.Click += new System.EventHandler(this.sagittalButton_Click);
            // 
            // coronalButton
            // 
            this.coronalButton.Cursor = System.Windows.Forms.Cursors.Hand;
            this.coronalButton.Enabled = false;
            this.coronalButton.ForeColor = System.Drawing.SystemColors.ControlText;
            this.coronalButton.Location = new System.Drawing.Point(10, 369);
            this.coronalButton.Margin = new System.Windows.Forms.Padding(3, 5, 3, 5);
            this.coronalButton.Name = "coronalButton";
            this.coronalButton.Size = new System.Drawing.Size(148, 40);
            this.coronalButton.TabIndex = 8;
            this.coronalButton.Text = "Coronal, XZ";
            this.coronalButton.UseVisualStyleBackColor = true;
            this.coronalButton.Click += new System.EventHandler(this.coronalButton_Click);
            // 
            // transverseButton
            // 
            this.transverseButton.Cursor = System.Windows.Forms.Cursors.Hand;
            this.transverseButton.Enabled = false;
            this.transverseButton.ForeColor = System.Drawing.SystemColors.ControlText;
            this.transverseButton.Location = new System.Drawing.Point(10, 320);
            this.transverseButton.Margin = new System.Windows.Forms.Padding(3, 5, 3, 5);
            this.transverseButton.Name = "transverseButton";
            this.transverseButton.Size = new System.Drawing.Size(148, 40);
            this.transverseButton.TabIndex = 7;
            this.transverseButton.Text = "Transverse, XY";
            this.transverseButton.UseVisualStyleBackColor = true;
            this.transverseButton.Click += new System.EventHandler(this.transverseButton_Click);
            // 
            // volumeButton
            // 
            this.volumeButton.Cursor = System.Windows.Forms.Cursors.Hand;
            this.volumeButton.Enabled = false;
            this.volumeButton.ForeColor = System.Drawing.SystemColors.ControlText;
            this.volumeButton.Location = new System.Drawing.Point(10, 275);
            this.volumeButton.Margin = new System.Windows.Forms.Padding(3, 5, 3, 5);
            this.volumeButton.Name = "volumeButton";
            this.volumeButton.Size = new System.Drawing.Size(148, 40);
            this.volumeButton.TabIndex = 6;
            this.volumeButton.Text = "Volume";
            this.volumeButton.UseVisualStyleBackColor = true;
            this.volumeButton.Click += new System.EventHandler(this.volumeButton_Click);
            // 
            // resetButton
            // 
            this.resetButton.Cursor = System.Windows.Forms.Cursors.Hand;
            this.resetButton.Enabled = false;
            this.resetButton.ForeColor = System.Drawing.SystemColors.ControlText;
            this.resetButton.Location = new System.Drawing.Point(10, 195);
            this.resetButton.Margin = new System.Windows.Forms.Padding(3, 5, 3, 5);
            this.resetButton.Name = "resetButton";
            this.resetButton.Size = new System.Drawing.Size(148, 71);
            this.resetButton.TabIndex = 5;
            this.resetButton.Text = "Reset Camera";
            this.resetButton.UseVisualStyleBackColor = true;
            this.resetButton.Click += new System.EventHandler(this.resetButton_Click);
            // 
            // label1
            // 
            this.label1.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(10, 1202);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(72, 20);
            this.label1.TabIndex = 4;
            this.label1.Text = "Gray min";
            // 
            // gmaxLabel
            // 
            this.gmaxLabel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.gmaxLabel.AutoSize = true;
            this.gmaxLabel.Location = new System.Drawing.Point(10, 1139);
            this.gmaxLabel.Name = "gmaxLabel";
            this.gmaxLabel.Size = new System.Drawing.Size(76, 20);
            this.gmaxLabel.TabIndex = 2;
            this.gmaxLabel.Text = "Gray max";
            // 
            // gminBar
            // 
            this.gminBar.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.gminBar.Cursor = System.Windows.Forms.Cursors.Hand;
            this.gminBar.Enabled = false;
            this.gminBar.Location = new System.Drawing.Point(-3, 1223);
            this.gminBar.Maximum = 255;
            this.gminBar.Name = "gminBar";
            this.gminBar.Size = new System.Drawing.Size(165, 24);
            this.gminBar.TabIndex = 3;
            this.gminBar.Scroll += new System.Windows.Forms.ScrollEventHandler(this.gminBar_Scroll);
            // 
            // gmaxBar
            // 
            this.gmaxBar.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.gmaxBar.Cursor = System.Windows.Forms.Cursors.Hand;
            this.gmaxBar.Enabled = false;
            this.gmaxBar.Location = new System.Drawing.Point(-3, 1160);
            this.gmaxBar.Maximum = 255;
            this.gmaxBar.Name = "gmaxBar";
            this.gmaxBar.Size = new System.Drawing.Size(166, 24);
            this.gmaxBar.TabIndex = 2;
            this.gmaxBar.Value = 255;
            this.gmaxBar.Scroll += new System.Windows.Forms.ScrollEventHandler(this.gmaxBar_Scroll);
            // 
            // maskButton
            // 
            this.maskButton.Cursor = System.Windows.Forms.Cursors.Hand;
            this.maskButton.Enabled = false;
            this.maskButton.ForeColor = System.Drawing.SystemColors.ControlText;
            this.maskButton.Location = new System.Drawing.Point(10, 85);
            this.maskButton.Margin = new System.Windows.Forms.Padding(3, 5, 3, 5);
            this.maskButton.Name = "maskButton";
            this.maskButton.Size = new System.Drawing.Size(148, 71);
            this.maskButton.TabIndex = 1;
            this.maskButton.Text = "Load Mask";
            this.maskButton.UseVisualStyleBackColor = true;
            this.maskButton.Click += new System.EventHandler(this.maskButton_Click);
            // 
            // fileButton
            // 
            this.fileButton.Cursor = System.Windows.Forms.Cursors.Hand;
            this.fileButton.ForeColor = System.Drawing.SystemColors.ControlText;
            this.fileButton.BackColor = System.Drawing.SystemColors.ControlLight;
            this.fileButton.Location = new System.Drawing.Point(10, 6);
            this.fileButton.Margin = new System.Windows.Forms.Padding(3, 5, 3, 5);
            this.fileButton.Name = "fileButton";
            this.fileButton.Size = new System.Drawing.Size(148, 71);
            this.fileButton.TabIndex = 0;
            this.fileButton.Text = "Load Volume";
            this.fileButton.UseVisualStyleBackColor = true;
            this.fileButton.Click += new System.EventHandler(this.fileButton_Click);
            // 
            // panel2
            // 
            this.panel2.BackColor = System.Drawing.SystemColors.WindowFrame;
            this.tableLayoutPanel1.SetColumnSpan(this.panel2, 3);
            this.panel2.Controls.Add(this.gradeLabel);
            this.panel2.Controls.Add(this.sliceLabel);
            this.panel2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel2.Enabled = false;
            this.panel2.Location = new System.Drawing.Point(3, 5);
            this.panel2.Margin = new System.Windows.Forms.Padding(3, 5, 3, 5);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(1598, 30);
            this.panel2.TabIndex = 1;
            // 
            // gradeLabel
            // 
            this.gradeLabel.AutoSize = true;
            this.gradeLabel.Dock = System.Windows.Forms.DockStyle.Left;
            this.gradeLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.gradeLabel.Location = new System.Drawing.Point(0, 0);
            this.gradeLabel.Name = "gradeLabel";
            this.gradeLabel.Size = new System.Drawing.Size(78, 20);
            this.gradeLabel.TabIndex = 3;
            this.gradeLabel.Text = "No Grade";
            // 
            // sliceLabel
            // 
            this.sliceLabel.AutoSize = true;
            this.sliceLabel.Dock = System.Windows.Forms.DockStyle.Right;
            this.sliceLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.sliceLabel.Location = new System.Drawing.Point(1533, 0);
            this.sliceLabel.Name = "sliceLabel";
            this.sliceLabel.Size = new System.Drawing.Size(65, 20);
            this.sliceLabel.TabIndex = 2;
            this.sliceLabel.Text = "No data";
            // 
            // sliceBar
            // 
            this.sliceBar.Cursor = System.Windows.Forms.Cursors.Hand;
            this.sliceBar.Dock = System.Windows.Forms.DockStyle.Fill;
            this.sliceBar.Enabled = false;
            this.sliceBar.Location = new System.Drawing.Point(1564, 40);
            this.sliceBar.Name = "sliceBar";
            this.sliceBar.Size = new System.Drawing.Size(40, 1264);
            this.sliceBar.TabIndex = 2;
            this.sliceBar.Value = 50;
            this.sliceBar.Scroll += new System.Windows.Forms.ScrollEventHandler(this.sliceBar_Scroll);
            // 
            // renderWindowControl
            // 
            this.renderWindowControl.AddTestActors = false;
            this.renderWindowControl.Cursor = System.Windows.Forms.Cursors.Hand;
            this.renderWindowControl.Dock = System.Windows.Forms.DockStyle.Fill;
            this.renderWindowControl.Enabled = false;
            this.renderWindowControl.Location = new System.Drawing.Point(176, 46);
            this.renderWindowControl.Margin = new System.Windows.Forms.Padding(6);
            this.renderWindowControl.Name = "renderWindowControl";
            this.renderWindowControl.Size = new System.Drawing.Size(1382, 1500);
            this.renderWindowControl.TabIndex = 3;
            this.renderWindowControl.TestText = null;
            this.renderWindowControl.Load += new System.EventHandler(this.renderWindowControl_Load);
            // 
            // panel3
            // 
            this.tableLayoutPanel1.SetColumnSpan(this.panel3, 3);
            this.panel3.Controls.Add(this.label4);
            this.panel3.Controls.Add(this.mainProgress);
            this.panel3.Controls.Add(this.maskLabel);
            this.panel3.Controls.Add(this.fileLabel);
            this.panel3.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel3.Location = new System.Drawing.Point(3, 1307);
            this.panel3.Name = "panel3";
            this.panel3.Size = new System.Drawing.Size(1598, 24);
            this.panel3.TabIndex = 4;
            // 
            // label4
            // 
            this.label4.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(1345, 4);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(76, 20);
            this.label4.TabIndex = 3;
            this.label4.Text = "Progress:";
            // 
            // mainProgress
            // 
            this.mainProgress.Dock = System.Windows.Forms.DockStyle.Right;
            this.mainProgress.Location = new System.Drawing.Point(1433, 0);
            this.mainProgress.Name = "mainProgress";
            this.mainProgress.Size = new System.Drawing.Size(165, 24);
            this.mainProgress.TabIndex = 2;
            // 
            // maskLabel
            // 
            this.maskLabel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.maskLabel.AutoSize = true;
            this.maskLabel.Location = new System.Drawing.Point(337, 4);
            this.maskLabel.Name = "maskLabel";
            this.maskLabel.Size = new System.Drawing.Size(129, 20);
            this.maskLabel.TabIndex = 1;
            this.maskLabel.Text = "No Mask Loaded";
            // 
            // fileLabel
            // 
            this.fileLabel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.fileLabel.AutoSize = true;
            this.fileLabel.Location = new System.Drawing.Point(0, 4);
            this.fileLabel.Name = "fileLabel";
            this.fileLabel.Size = new System.Drawing.Size(126, 20);
            this.fileLabel.TabIndex = 0;
            this.fileLabel.Text = "No Data Loaded";
            // 
            // fileDialog
            // 
            this.fileDialog.FileName = "openFileDialog1";
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.WindowFrame;
            this.ClientSize = new System.Drawing.Size(1604, 1500);
            this.Controls.Add(this.tableLayoutPanel1);
            this.ForeColor = System.Drawing.SystemColors.InactiveCaption;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Margin = new System.Windows.Forms.Padding(3, 5, 3, 5);
            this.Name = "MainForm";
            this.Text = "MIPT-Histological-Grading";
            this.tableLayoutPanel1.ResumeLayout(false);
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.panel2.ResumeLayout(false);
            this.panel2.PerformLayout();
            this.panel3.ResumeLayout(false);
            this.panel3.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Button fileButton;
        private System.Windows.Forms.Button maskButton;
        private System.Windows.Forms.VScrollBar sliceBar;
        private System.Windows.Forms.HScrollBar gminBar;
        private System.Windows.Forms.HScrollBar gmaxBar;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label gmaxLabel;
        private System.Windows.Forms.Button resetButton;
        private Kitware.VTK.RenderWindowControl renderWindowControl;
        private System.Windows.Forms.Button sagittalButton;
        private System.Windows.Forms.Button coronalButton;
        private System.Windows.Forms.Button transverseButton;
        private System.Windows.Forms.Button volumeButton;
        private System.Windows.Forms.Label viewLabel;
        private System.Windows.Forms.OpenFileDialog fileDialog;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Button predict;
        private System.Windows.Forms.Button segmentButton;
        private System.Windows.Forms.Button cropButton;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Button getVoiButton;
        private System.Windows.Forms.Button rotate_button;
        private System.Windows.Forms.Button saveButton;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.Label gradeLabel;
        private System.Windows.Forms.Label sliceLabel;
        private System.Windows.Forms.Label maskLabel;
        private System.Windows.Forms.Label fileLabel;
        private System.Windows.Forms.Panel panel3;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.ProgressBar mainProgress;
    }
}

