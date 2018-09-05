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
            this.viewLabel = new System.Windows.Forms.Label();
            this.transverse2Button = new System.Windows.Forms.Button();
            this.transverse1Button = new System.Windows.Forms.Button();
            this.coronalButton = new System.Windows.Forms.Button();
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
            this.voiButton = new System.Windows.Forms.Button();
            this.tableLayoutPanel1.SuspendLayout();
            this.panel1.SuspendLayout();
            this.panel2.SuspendLayout();
            this.SuspendLayout();
            // 
            // tableLayoutPanel1
            // 
            this.tableLayoutPanel1.ColumnCount = 3;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 150F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 60F));
            this.tableLayoutPanel1.Controls.Add(this.panel1, 0, 1);
            this.tableLayoutPanel1.Controls.Add(this.panel2, 0, 0);
            this.tableLayoutPanel1.Controls.Add(this.sliceBar, 2, 1);
            this.tableLayoutPanel1.Controls.Add(this.renderWindowControl, 1, 1);
            this.tableLayoutPanel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanel1.Location = new System.Drawing.Point(0, 0);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 2;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 70F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 449F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(1143, 759);
            this.tableLayoutPanel1.TabIndex = 0;
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.voiButton);
            this.panel1.Controls.Add(this.viewLabel);
            this.panel1.Controls.Add(this.transverse2Button);
            this.panel1.Controls.Add(this.transverse1Button);
            this.panel1.Controls.Add(this.coronalButton);
            this.panel1.Controls.Add(this.volumeButton);
            this.panel1.Controls.Add(this.resetButton);
            this.panel1.Controls.Add(this.label1);
            this.panel1.Controls.Add(this.gmaxLabel);
            this.panel1.Controls.Add(this.gminBar);
            this.panel1.Controls.Add(this.gmaxBar);
            this.panel1.Controls.Add(this.maskButton);
            this.panel1.Controls.Add(this.fileButton);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel1.Location = new System.Drawing.Point(3, 73);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(144, 683);
            this.panel1.TabIndex = 0;
            // 
            // viewLabel
            // 
            this.viewLabel.AutoSize = true;
            this.viewLabel.Location = new System.Drawing.Point(9, 217);
            this.viewLabel.Name = "viewLabel";
            this.viewLabel.Size = new System.Drawing.Size(37, 17);
            this.viewLabel.TabIndex = 10;
            this.viewLabel.Text = "View";
            // 
            // transverse2Button
            // 
            this.transverse2Button.Location = new System.Drawing.Point(6, 351);
            this.transverse2Button.Name = "transverse2Button";
            this.transverse2Button.Size = new System.Drawing.Size(132, 32);
            this.transverse2Button.TabIndex = 9;
            this.transverse2Button.Text = "Transverse, YZ";
            this.transverse2Button.UseVisualStyleBackColor = true;
            this.transverse2Button.Click += new System.EventHandler(this.transverse2Button_Click);
            // 
            // transverse1Button
            // 
            this.transverse1Button.Location = new System.Drawing.Point(6, 313);
            this.transverse1Button.Name = "transverse1Button";
            this.transverse1Button.Size = new System.Drawing.Size(132, 32);
            this.transverse1Button.TabIndex = 8;
            this.transverse1Button.Text = "Transverse, XZ";
            this.transverse1Button.UseVisualStyleBackColor = true;
            this.transverse1Button.Click += new System.EventHandler(this.transverse1Button_Click);
            // 
            // coronalButton
            // 
            this.coronalButton.Location = new System.Drawing.Point(6, 275);
            this.coronalButton.Name = "coronalButton";
            this.coronalButton.Size = new System.Drawing.Size(132, 32);
            this.coronalButton.TabIndex = 7;
            this.coronalButton.Text = "Coronal";
            this.coronalButton.UseVisualStyleBackColor = true;
            this.coronalButton.Click += new System.EventHandler(this.coronalButton_Click);
            // 
            // volumeButton
            // 
            this.volumeButton.Location = new System.Drawing.Point(6, 237);
            this.volumeButton.Name = "volumeButton";
            this.volumeButton.Size = new System.Drawing.Size(132, 32);
            this.volumeButton.TabIndex = 6;
            this.volumeButton.Text = "Volume";
            this.volumeButton.UseVisualStyleBackColor = true;
            this.volumeButton.Click += new System.EventHandler(this.volumeButton_Click);
            // 
            // resetButton
            // 
            this.resetButton.Location = new System.Drawing.Point(9, 139);
            this.resetButton.Name = "resetButton";
            this.resetButton.Size = new System.Drawing.Size(132, 57);
            this.resetButton.TabIndex = 5;
            this.resetButton.Text = "Reset Camera";
            this.resetButton.UseVisualStyleBackColor = true;
            this.resetButton.Click += new System.EventHandler(this.resetButton_Click);
            // 
            // label1
            // 
            this.label1.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(10, 642);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(65, 17);
            this.label1.TabIndex = 4;
            this.label1.Text = "Gray min";
            // 
            // gmaxLabel
            // 
            this.gmaxLabel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.gmaxLabel.AutoSize = true;
            this.gmaxLabel.Location = new System.Drawing.Point(10, 591);
            this.gmaxLabel.Name = "gmaxLabel";
            this.gmaxLabel.Size = new System.Drawing.Size(68, 17);
            this.gmaxLabel.TabIndex = 2;
            this.gmaxLabel.Text = "Gray max";
            // 
            // gminBar
            // 
            this.gminBar.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.gminBar.Location = new System.Drawing.Point(-3, 659);
            this.gminBar.Maximum = 255;
            this.gminBar.Name = "gminBar";
            this.gminBar.Size = new System.Drawing.Size(147, 24);
            this.gminBar.TabIndex = 3;
            this.gminBar.Scroll += new System.Windows.Forms.ScrollEventHandler(this.gminBar_Scroll);
            // 
            // gmaxBar
            // 
            this.gmaxBar.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.gmaxBar.Location = new System.Drawing.Point(-3, 608);
            this.gmaxBar.Maximum = 255;
            this.gmaxBar.Name = "gmaxBar";
            this.gmaxBar.Size = new System.Drawing.Size(147, 24);
            this.gmaxBar.TabIndex = 2;
            this.gmaxBar.Value = 255;
            this.gmaxBar.Scroll += new System.Windows.Forms.ScrollEventHandler(this.gmaxBar_Scroll);
            // 
            // maskButton
            // 
            this.maskButton.Location = new System.Drawing.Point(9, 76);
            this.maskButton.Name = "maskButton";
            this.maskButton.Size = new System.Drawing.Size(132, 57);
            this.maskButton.TabIndex = 1;
            this.maskButton.Text = "Load Mask";
            this.maskButton.UseVisualStyleBackColor = true;
            this.maskButton.Click += new System.EventHandler(this.maskButton_Click);
            // 
            // fileButton
            // 
            this.fileButton.Location = new System.Drawing.Point(9, 13);
            this.fileButton.Name = "fileButton";
            this.fileButton.Size = new System.Drawing.Size(132, 57);
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
            this.panel2.Location = new System.Drawing.Point(3, 3);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(1137, 64);
            this.panel2.TabIndex = 1;
            // 
            // sliceLabel
            // 
            this.sliceLabel.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.sliceLabel.AutoSize = true;
            this.sliceLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.sliceLabel.Location = new System.Drawing.Point(599, 37);
            this.sliceLabel.Name = "sliceLabel";
            this.sliceLabel.Size = new System.Drawing.Size(80, 25);
            this.sliceLabel.TabIndex = 2;
            this.sliceLabel.Text = "No data";
            // 
            // maskLabel
            // 
            this.maskLabel.AutoSize = true;
            this.maskLabel.Location = new System.Drawing.Point(10, 37);
            this.maskLabel.Name = "maskLabel";
            this.maskLabel.Size = new System.Drawing.Size(115, 17);
            this.maskLabel.TabIndex = 1;
            this.maskLabel.Text = "No Mask Loaded";
            // 
            // fileLabel
            // 
            this.fileLabel.AutoSize = true;
            this.fileLabel.Location = new System.Drawing.Point(10, 10);
            this.fileLabel.Name = "fileLabel";
            this.fileLabel.Size = new System.Drawing.Size(112, 17);
            this.fileLabel.TabIndex = 0;
            this.fileLabel.Text = "No Data Loaded";
            // 
            // sliceBar
            // 
            this.sliceBar.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.sliceBar.Location = new System.Drawing.Point(1083, 70);
            this.sliceBar.Name = "sliceBar";
            this.sliceBar.Size = new System.Drawing.Size(60, 689);
            this.sliceBar.TabIndex = 2;
            this.sliceBar.Value = 50;
            this.sliceBar.Scroll += new System.Windows.Forms.ScrollEventHandler(this.sliceBar_Scroll);
            // 
            // renderWindowControl
            // 
            this.renderWindowControl.AddTestActors = false;
            this.renderWindowControl.Dock = System.Windows.Forms.DockStyle.Fill;
            this.renderWindowControl.Location = new System.Drawing.Point(154, 74);
            this.renderWindowControl.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.renderWindowControl.Name = "renderWindowControl";
            this.renderWindowControl.Size = new System.Drawing.Size(925, 681);
            this.renderWindowControl.TabIndex = 3;
            this.renderWindowControl.TestText = null;
            this.renderWindowControl.Load += new System.EventHandler(this.renderWindowControl_Load);
            // 
            // fileDialog
            // 
            this.fileDialog.FileName = "openFileDialog1";
            // 
            // voiButton
            // 
            this.voiButton.Location = new System.Drawing.Point(6, 424);
            this.voiButton.Name = "voiButton";
            this.voiButton.Size = new System.Drawing.Size(132, 41);
            this.voiButton.TabIndex = 3;
            this.voiButton.Text = "Select VOI";
            this.voiButton.UseVisualStyleBackColor = true;
            this.voiButton.Click += new System.EventHandler(this.voiButton_Click);
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1143, 759);
            this.Controls.Add(this.tableLayoutPanel1);
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
    private System.Windows.Forms.Button transverse2Button;
    private System.Windows.Forms.Button transverse1Button;
    private System.Windows.Forms.Button coronalButton;
    private System.Windows.Forms.Button volumeButton;
    private System.Windows.Forms.Label viewLabel;
    private System.Windows.Forms.OpenFileDialog fileDialog;
    private System.Windows.Forms.Label sliceLabel;
        private System.Windows.Forms.Button voiButton;
    }
}

