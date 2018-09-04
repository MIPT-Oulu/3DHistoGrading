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
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.panel1 = new System.Windows.Forms.Panel();
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
            this.sliceLabel = new System.Windows.Forms.Label();
            this.maskLabel = new System.Windows.Forms.Label();
            this.fileLabel = new System.Windows.Forms.Label();
            this.sliceBar = new System.Windows.Forms.VScrollBar();
            this.renderWindowControl = new Kitware.VTK.RenderWindowControl();
            this.fileDialog = new System.Windows.Forms.OpenFileDialog();
            this.tableLayoutPanel1.SuspendLayout();
            this.panel1.SuspendLayout();
            this.panel2.SuspendLayout();
            this.SuspendLayout();
            // 
            // tableLayoutPanel1
            // 
            this.tableLayoutPanel1.ColumnCount = 3;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 113F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 45F));
            this.tableLayoutPanel1.Controls.Add(this.panel1, 0, 1);
            this.tableLayoutPanel1.Controls.Add(this.panel2, 0, 0);
            this.tableLayoutPanel1.Controls.Add(this.sliceBar, 2, 1);
            this.tableLayoutPanel1.Controls.Add(this.renderWindowControl, 1, 1);
            this.tableLayoutPanel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanel1.Location = new System.Drawing.Point(0, 0);
            this.tableLayoutPanel1.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 2;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 57F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 365F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(785, 493);
            this.tableLayoutPanel1.TabIndex = 0;
            // 
            // panel1
            // 
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
            this.panel1.Location = new System.Drawing.Point(2, 60);
            this.panel1.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(109, 430);
            this.panel1.TabIndex = 0;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(7, 296);
            this.label2.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(44, 13);
            this.label2.TabIndex = 3;
            this.label2.Text = "Grading";
            // 
            // viewLabel
            // 
            this.viewLabel.AutoSize = true;
            this.viewLabel.Location = new System.Drawing.Point(7, 157);
            this.viewLabel.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.viewLabel.Name = "viewLabel";
            this.viewLabel.Size = new System.Drawing.Size(30, 13);
            this.viewLabel.TabIndex = 10;
            this.viewLabel.Text = "View";
            // 
            // predict
            // 
            this.predict.Location = new System.Drawing.Point(9, 312);
            this.predict.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.predict.Name = "predict";
            this.predict.Size = new System.Drawing.Size(94, 42);
            this.predict.TabIndex = 2;
            this.predict.Text = "Grade sample";
            this.predict.UseVisualStyleBackColor = true;
            this.predict.Click += new System.EventHandler(this.predict_Click);
            // 
            // sagittalButton
            // 
            this.sagittalButton.Enabled = false;
            this.sagittalButton.Location = new System.Drawing.Point(5, 266);
            this.sagittalButton.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.sagittalButton.Name = "sagittalButton";
            this.sagittalButton.Size = new System.Drawing.Size(99, 26);
            this.sagittalButton.TabIndex = 9;
            this.sagittalButton.Text = "Sagittal, YZ";
            this.sagittalButton.UseVisualStyleBackColor = true;
            this.sagittalButton.Click += new System.EventHandler(this.sagittalButton_Click);
            // 
            // coronalButton
            // 
            this.coronalButton.Enabled = false;
            this.coronalButton.Location = new System.Drawing.Point(5, 235);
            this.coronalButton.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.coronalButton.Name = "coronalButton";
            this.coronalButton.Size = new System.Drawing.Size(99, 26);
            this.coronalButton.TabIndex = 8;
            this.coronalButton.Text = "Coronal, XZ";
            this.coronalButton.UseVisualStyleBackColor = true;
            this.coronalButton.Click += new System.EventHandler(this.coronalButton_Click);
            // 
            // transverseButton
            // 
            this.transverseButton.Enabled = false;
            this.transverseButton.Location = new System.Drawing.Point(5, 205);
            this.transverseButton.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.transverseButton.Name = "transverseButton";
            this.transverseButton.Size = new System.Drawing.Size(99, 26);
            this.transverseButton.TabIndex = 7;
            this.transverseButton.Text = "Transverse, XY";
            this.transverseButton.UseVisualStyleBackColor = true;
            this.transverseButton.Click += new System.EventHandler(this.transverseButton_Click);
            // 
            // volumeButton
            // 
            this.volumeButton.Enabled = false;
            this.volumeButton.Location = new System.Drawing.Point(5, 174);
            this.volumeButton.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.volumeButton.Name = "volumeButton";
            this.volumeButton.Size = new System.Drawing.Size(99, 26);
            this.volumeButton.TabIndex = 6;
            this.volumeButton.Text = "Volume";
            this.volumeButton.UseVisualStyleBackColor = true;
            this.volumeButton.Click += new System.EventHandler(this.volumeButton_Click);
            // 
            // resetButton
            // 
            this.resetButton.Enabled = false;
            this.resetButton.Location = new System.Drawing.Point(7, 107);
            this.resetButton.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.resetButton.Name = "resetButton";
            this.resetButton.Size = new System.Drawing.Size(99, 46);
            this.resetButton.TabIndex = 5;
            this.resetButton.Text = "Reset Camera";
            this.resetButton.UseVisualStyleBackColor = true;
            this.resetButton.Click += new System.EventHandler(this.resetButton_Click);
            // 
            // label1
            // 
            this.label1.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(7, 396);
            this.label1.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(48, 13);
            this.label1.TabIndex = 4;
            this.label1.Text = "Gray min";
            // 
            // gmaxLabel
            // 
            this.gmaxLabel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.gmaxLabel.AutoSize = true;
            this.gmaxLabel.Location = new System.Drawing.Point(7, 355);
            this.gmaxLabel.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.gmaxLabel.Name = "gmaxLabel";
            this.gmaxLabel.Size = new System.Drawing.Size(51, 13);
            this.gmaxLabel.TabIndex = 2;
            this.gmaxLabel.Text = "Gray max";
            // 
            // gminBar
            // 
            this.gminBar.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.gminBar.Enabled = false;
            this.gminBar.Location = new System.Drawing.Point(-2, 410);
            this.gminBar.Maximum = 255;
            this.gminBar.Name = "gminBar";
            this.gminBar.Size = new System.Drawing.Size(110, 24);
            this.gminBar.TabIndex = 3;
            this.gminBar.Scroll += new System.Windows.Forms.ScrollEventHandler(this.gminBar_Scroll);
            // 
            // gmaxBar
            // 
            this.gmaxBar.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.gmaxBar.Enabled = false;
            this.gmaxBar.Location = new System.Drawing.Point(-2, 369);
            this.gmaxBar.Maximum = 255;
            this.gmaxBar.Name = "gmaxBar";
            this.gmaxBar.Size = new System.Drawing.Size(110, 24);
            this.gmaxBar.TabIndex = 2;
            this.gmaxBar.Value = 255;
            this.gmaxBar.Scroll += new System.Windows.Forms.ScrollEventHandler(this.gmaxBar_Scroll);
            // 
            // maskButton
            // 
            this.maskButton.Enabled = false;
            this.maskButton.Location = new System.Drawing.Point(7, 55);
            this.maskButton.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.maskButton.Name = "maskButton";
            this.maskButton.Size = new System.Drawing.Size(99, 46);
            this.maskButton.TabIndex = 1;
            this.maskButton.Text = "Load Mask";
            this.maskButton.UseVisualStyleBackColor = true;
            this.maskButton.Click += new System.EventHandler(this.maskButton_Click);
            // 
            // fileButton
            // 
            this.fileButton.Location = new System.Drawing.Point(7, 4);
            this.fileButton.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.fileButton.Name = "fileButton";
            this.fileButton.Size = new System.Drawing.Size(99, 46);
            this.fileButton.TabIndex = 0;
            this.fileButton.Text = "Load Volume";
            this.fileButton.UseVisualStyleBackColor = true;
            this.fileButton.Click += new System.EventHandler(this.fileButton_Click);
            // 
            // panel2
            // 
            this.panel2.BackColor = System.Drawing.SystemColors.Control;
            this.tableLayoutPanel1.SetColumnSpan(this.panel2, 3);
            this.panel2.Controls.Add(this.sliceLabel);
            this.panel2.Controls.Add(this.maskLabel);
            this.panel2.Controls.Add(this.fileLabel);
            this.panel2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel2.Enabled = false;
            this.panel2.Location = new System.Drawing.Point(2, 3);
            this.panel2.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(781, 51);
            this.panel2.TabIndex = 1;
            // 
            // sliceLabel
            // 
            this.sliceLabel.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.sliceLabel.AutoSize = true;
            this.sliceLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.sliceLabel.Location = new System.Drawing.Point(449, 30);
            this.sliceLabel.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.sliceLabel.Name = "sliceLabel";
            this.sliceLabel.Size = new System.Drawing.Size(65, 20);
            this.sliceLabel.TabIndex = 2;
            this.sliceLabel.Text = "No data";
            // 
            // maskLabel
            // 
            this.maskLabel.AutoSize = true;
            this.maskLabel.Location = new System.Drawing.Point(7, 30);
            this.maskLabel.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.maskLabel.Name = "maskLabel";
            this.maskLabel.Size = new System.Drawing.Size(89, 13);
            this.maskLabel.TabIndex = 1;
            this.maskLabel.Text = "No Mask Loaded";
            // 
            // fileLabel
            // 
            this.fileLabel.AutoSize = true;
            this.fileLabel.Location = new System.Drawing.Point(7, 8);
            this.fileLabel.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.fileLabel.Name = "fileLabel";
            this.fileLabel.Size = new System.Drawing.Size(86, 13);
            this.fileLabel.TabIndex = 0;
            this.fileLabel.Text = "No Data Loaded";
            // 
            // sliceBar
            // 
            this.sliceBar.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.sliceBar.Enabled = false;
            this.sliceBar.Location = new System.Drawing.Point(740, 57);
            this.sliceBar.Name = "sliceBar";
            this.sliceBar.Size = new System.Drawing.Size(45, 436);
            this.sliceBar.TabIndex = 2;
            this.sliceBar.Value = 50;
            this.sliceBar.Scroll += new System.Windows.Forms.ScrollEventHandler(this.sliceBar_Scroll);
            // 
            // renderWindowControl
            // 
            this.renderWindowControl.AddTestActors = false;
            this.renderWindowControl.Dock = System.Windows.Forms.DockStyle.Fill;
            this.renderWindowControl.Enabled = false;
            this.renderWindowControl.Location = new System.Drawing.Point(116, 60);
            this.renderWindowControl.Name = "renderWindowControl";
            this.renderWindowControl.Size = new System.Drawing.Size(621, 430);
            this.renderWindowControl.TabIndex = 3;
            this.renderWindowControl.TestText = null;
            this.renderWindowControl.Load += new System.EventHandler(this.renderWindowControl_Load);
            // 
            // fileDialog
            // 
            this.fileDialog.FileName = "openFileDialog1";
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(785, 493);
            this.Controls.Add(this.tableLayoutPanel1);
            this.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.Name = "MainForm";
            this.Text = "CTVisualization";
            this.tableLayoutPanel1.ResumeLayout(false);
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.panel2.ResumeLayout(false);
            this.panel2.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Button fileButton;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.Label maskLabel;
        private System.Windows.Forms.Label fileLabel;
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
        private System.Windows.Forms.Label sliceLabel;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Button predict;
    }
}

