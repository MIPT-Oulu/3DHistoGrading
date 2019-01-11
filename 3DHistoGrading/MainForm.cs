using System;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using Kitware.VTK;

using HistoGrading.Components;
using HistoGrading.Models;

namespace HistoGrading
{
    /// <summary>
    /// Windows forms object that includes the main graphical user interface of the software. Includes callbacks for each interactor on the form.
    /// </summary>
    public partial class MainForm : Form
    {
        //Settings and Initialization

        //Rendering flags
        int is_rendering = 0;
        int is_mask = 0;
        int is_coronal = 0;
        int is_sagittal = 0;
       
        //Saving flags
        int save_mask = 0;
        int save_vois = 0;

        //Volume dimensions
        int[] dims = new int[] { 0, 0, 0, 0, 0, 0 };
        //Current slice
        int[] sliceN = new int[] { 0, 0, 0 };
        //Current orientation, -1 = 3D rendering
        int ori = -1;

        //Gray values
        int[] gray = new int[2] { 0, 255 };
       

        //Rendering object
        Rendering.renderPipeLine volume = new Rendering.renderPipeLine();
        
        //Render window
        private vtkRenderWindow renWin;

        //Mouse interactor
        //bool mouseDown1 = false;
        //bool mouseDown2 = false;

        // Sample name
        string fname = null;

        // Tooltip
        string tip = "Start by loading a PTA sample";

        //Save directory
        string savedir = "C:\\Users\\Tuomas Frondelius\\Desktop\\PTAResults";

        // Grading variables
        Model model = new Model();
        int[,] features = new int[0,0];

        /// <summary>
        /// Form that includes all major components in the software.
        /// </summary>
        public MainForm()
        {
            InitializeComponent();
            gradeLabel.Text = tip;
        }

        /// <summary>
        /// Updates GUI text about current rendering method in use.
        /// </summary>
        public void TellSlice()
        {
            //Tell volume rendering
            if(ori == -1)
            {
                sliceLabel.Text = "Rendering volume";
                gradeLabel.Text = tip;
            }
            //Tell transverse rendering
            if (ori == 2)
            {
                sliceLabel.Text = string.Format("Transverse, XY | {0} / {1}", sliceN[ori]-dims[4], dims[5]-dims[4]);
                gradeLabel.Text = tip;
            }
            //Tell coronal rendering
            if (ori == 0)
            {
                sliceLabel.Text = string.Format("Coronal, XZ | {0} / {1}", sliceN[ori]-dims[0], dims[1]-dims[0]);
                gradeLabel.Text = "Drag right mouse button to crop artefacts";
            }
            //Tell sagittal rendering
            if (ori == 1)
            {
                sliceLabel.Text = string.Format("Sagittal, YZ | {0} / {1}", sliceN[ori]-dims[2], dims[3]-dims[2]);
                gradeLabel.Text = "Drag right mouse button to crop artefacts";
            }
        }

        //Scroll bar updates
        private void updateEvents()
        {
            gmaxBar.ValueChanged += new EventHandler(gmaxBar_ValueChanged);
            gminBar.ValueChanged += new EventHandler(gminBar_ValueChanged);
        }

        private void gmaxBar_ValueChanged(object sender, EventArgs e)
        {
            //Check if rendering
            if (is_rendering == 1)
            {
                //Update gray value range and render volume
                gray[1] = gmaxBar.Value;
                volume.updateCurrent(sliceN,ori,gray);                
                //Update slice if rendering
                if (ori > -1)
                {
                    volume.renderImage();
                    //Render mask
                    if (is_mask == 1)
                    {
                        volume.renderImageMask();
                    }
                }
                else
                {
                    volume.setVolumeColor();
                }
            }
        }

        private void gminBar_ValueChanged(object sender, EventArgs e)
        {
            //Check if rendering
            if (is_rendering == 1)
            {
                //Update gary value range and render volume
                gray[0] = gminBar.Value;
                volume.updateCurrent(sliceN, ori, gray);
                //Update slice if rendering
                if(ori>-1)
                {
                    volume.renderImage();
                    //Render mask
                    if(is_mask==1)
                    {
                        volume.renderImageMask();
                    }
                }
                else
                {
                    volume.setVolumeColor();
                }
            }
        }

        private void sliceBar_ValueChanged(object sender, EventArgs e)
        {
            //Check if rendering slice
            if (ori > -1)
            {
                //Set slice
                sliceN[ori] = sliceBar.Value;
                volume.updateCurrent(sliceN, ori, gray);
                volume.renderImage();
                if (is_mask==1)
                {
                    volume.renderImageMask();
                }
                TellSlice();                
            }
        }

        //Render window updates
        private void renderWindowControl_Load(object sender, EventArgs e)
        {            
            //Set renderwindow
            renWin = renderWindowControl.RenderWindow;                   

        }

        //Buttons

        //Load CT data
        private void fileButton_Click(object sender, EventArgs e)
        {
            label4.Text = "Loading: ";
            mainProgress.Value = 10;
            //Select a file and render volume
            if (fileDialog.ShowDialog() == DialogResult.OK)
            {                
                if(is_rendering == 1)
                {
                    //Remove renderer
                    renWin.RemoveRenderer(renWin.GetRenderers().GetFirstRenderer());
                    volume.Dispose();                    
                    volume = null;
                    GC.Collect();
                }
                
                //Initialize new volume
                volume = new Rendering.renderPipeLine();
                //Update renderwindow
                renderWindowControl_Load(this, null);

                fname = "";

                //Get path and files
                string impath = fileDialog.FileName;
                string folpath = Path.GetDirectoryName(@impath);

                // Get sample name
                var file = Path.GetFileName(impath).Split('_');
                for (int k = 0; k < file.Length - 1; k++)
                {
                    fname += file[k];
                    if (k < file.Length - 2)
                    {
                        fname += "_";
                    }
                }
                mainProgress.Value = 50;

                //Load data
                volume.connectData(impath);
                mainProgress.Value = 70;

                //Get dimensions and set slices. Middle slice is set to current slice
                dims = volume.getDims();
                sliceN[0] = (dims[1] + dims[0]) / 2;
                sliceN[1] = (dims[3] + dims[2]) / 2;
                sliceN[2] = (dims[5] + dims[4]) / 2;

                Console.WriteLine("Loaded fine");
                //Connect slice to renderer
                volume.connectWindow(renWin);

                Console.WriteLine("Connected window fine");
                //Render
                volume.renderVolume();

                Console.WriteLine("rendering..");

                //Flags for GUI components
                is_rendering = 1;
                is_mask = 0;
                is_coronal = 0;
                is_sagittal = 0;
                //Saving flags
                save_mask = 0;
                save_vois = 0;

                //Orientation
                ori = -1;
                //Update pipeline parameters
                volume.updateCurrent(sliceN, ori, gray);
                volume.setVolumeColor();

                //Update GUI
                maskButton.Text = "Load Mask";
                maskLabel.Text = "No Mask Loaded";
                coronalButton.Text = "Coronal, XZ";
                sagittalButton.Text = "Sagittal, YZ";
                TellSlice();
                                

                // Enable buttons
                sagittalButton.Enabled = true;
                coronalButton.Enabled = true;
                transverseButton.Enabled = true;
                volumeButton.Enabled = true;
                resetButton.Enabled = true;
                gminBar.Enabled = true;
                gmaxBar.Enabled = true;
                maskButton.Enabled = true;
                panel2.Enabled = true;
                sliceBar.Enabled = true;
                renderWindowControl.Enabled = true;
                cropButton.Enabled = true;
                rotate_button.Enabled = true;
                getVoiButton.Enabled = false;
                segmentButton.Enabled = false;
                predict.Enabled = false;
                saveButton.Enabled = false;

                GC.Collect();

                //Update GUI text to tell path to data folder
                fileLabel.Text = fname;
                tip = "Sample loaded and tools enabled";
                gradeLabel.Text = tip;
                mainProgress.Value = 100;
                label4.Text = "Done";
            }
        }

        //Load bone mask
        private void maskButton_Click(object sender, EventArgs e)
        {
            label4.Text = "Loading: ";
            mainProgress.Value = 10;
            //Dwitch between loading and removing the bone mask
            switch (is_mask)
            {
                case 0:
                    //Check if volume is rendered
                    if (is_rendering == 1)
                    {
                        //Select a file
                        if (fileDialog.ShowDialog() == DialogResult.OK)
                        {
                            //Get path and files
                            string impath = fileDialog.FileName;
                            string extension = Path.GetExtension(@impath);
                            string folpath = Path.GetDirectoryName(@impath);

                            maskLabel.Text = folpath;
                            //Load image data
                            volume.connectMask(impath);
                            //Update pipeline
                            volume.updateCurrent(sliceN, ori, gray);
                            // Set cartilage grids based on mask
                            //volume.SampleGrids();
                            mainProgress.Value = 50;

                            //Render
                            if (ori == -1)
                            {
                                volume.renderVolumeMask();
                                volume.setVolumeColor();
                            }
                            if (ori > -1)
                            {
                                volume.renderImageMask();
                            }

                            //Update flags
                            is_mask = 1;
                            maskButton.Text = "Remove Mask";                            
                        }
                    }
                    break;

                case 1:
                    volume.removeMask();
                    volume.updateCurrent(sliceN, ori, gray);
                    if (ori == -1)
                    {
                        volume.renderVolume();
                        TellSlice();
                    }
                    if (ori > -1)
                    {
                        volume.renderImage();
                        TellSlice();
                    }
                    is_mask = 0;
                    maskButton.Text = "Load Mask";
                    maskLabel.Text = "No Mask Loaded";
            break;
            }
            GC.Collect();

            // Update tooltip
            tip = "Loaded mask. If calcified zone mask was used, deep and surface zones can be extracted.";
            gradeLabel.Text = tip;
            mainProgress.Value = 100;
            label4.Text = "Done";
        }

        //Reset camera
        private void resetButton_Click(object sender, EventArgs e)
        {
            if (is_rendering == 1)
            {
                volume.resetCamera();

                //Reset slices
                sliceN[0] = (dims[1] + dims[0]) / 2;
                sliceN[1] = (dims[3] + dims[2]) / 2;
                sliceN[2] = (dims[5] + dims[4]) / 2;

                if (ori > -1)
                {
                    volume.updateCurrent(sliceN, ori, gray);
                    volume.renderImage();
                    if(is_mask == 1)
                    {
                        volume.renderImageMask();
                    }                    
                }
                TellSlice();
                GC.Collect();
            }
        }

        //Render volume
        private void volumeButton_Click(object sender, EventArgs e)
        {
            //Update parameters
            ori = -1;
            volume.updateCurrent(sliceN, ori, gray);
            //Render volume
            volume.renderVolume();

            if (is_mask==1)
            {
                volume.renderVolumeMask();
            }

            TellSlice();

            //Update GUI
            coronalButton.Text = "Coronal, XZ";
            sagittalButton.Text = "Sagittal, YZ";

            is_sagittal = 0;
            is_coronal = 0;

            GC.Collect();
        }

        //Render transverse slice, XY plane
        private void transverseButton_Click(object sender, EventArgs e)
        {
            if (is_rendering == 1)
            {
                //Set orientation
                ori = 2;
                //Update scroll bar
                sliceBar.Minimum = dims[4];
                sliceBar.Maximum = dims[5];
                sliceBar.Value = sliceN[2];
                sliceBar.Update();
                //Update rendering pipeline and render
                volume.updateCurrent(sliceN, ori, gray);
                volume.renderImage();
                //Check mask
                if (is_mask == 1)
                {
                    volume.renderImageMask();
                }
                TellSlice();                

                //Update GUI
                coronalButton.Text = "Coronal, XZ";
                sagittalButton.Text = "Sagittal, YZ";

                is_sagittal = 0;
                is_coronal = 0;
            }
            GC.Collect();
        }

        //Render coronal slice, XZ plane
        private void coronalButton_Click(object sender, EventArgs e)
        {
            if (is_rendering == 1 && is_coronal == 0)
            {                
                //Set orientation
                ori = 0;
                //Update scroll bar
                sliceBar.Minimum = dims[0];
                sliceBar.Maximum = dims[1];
                sliceBar.Value = sliceN[0];
                sliceBar.Update();

                //Update rendering pipeline and render
                volume.updateCurrent(sliceN, ori, gray);
                volume.renderImage();
                //Check mask
                if (is_mask == 1)
                {
                    volume.renderImageMask();
                }
                TellSlice();                

                //Update GUI
                coronalButton.Text = "Crop artefact";
                sagittalButton.Text = "Sagittal, YZ";

                is_sagittal = 0;
                is_coronal = 1;
            }
            else if (is_rendering == 1 && is_coronal == 1) // Artefact cropping
            {
                //Get Line ends and crop above the line
                volume.remove_artefact();
                volume.renderImage();
                //Check mask
                if (is_mask == 1)
                {
                    volume.renderImageMask();
                }

                // Update tooltip
                tip = "Cropped surface artefacts";
                gradeLabel.Text = tip;
            }
            GC.Collect();
        }

        //Render sagittal slice, YZ plane
        private void sagittalButton_Click(object sender, EventArgs e)
        {
            //Check if rendering
            if(is_rendering==1 && is_sagittal == 0)
            {
                //Set orientation
                ori = 1;
                //Update scroll bar
                sliceBar.Minimum = dims[2];
                sliceBar.Maximum = dims[3];
                sliceBar.Value = sliceN[1];
                sliceBar.Update();
                //Update rendering pipeline and render
                volume.updateCurrent(sliceN, ori, gray);
                volume.renderImage();
                //Check mask
                if(is_mask==1)
                {
                    volume.renderImageMask();
                }
                TellSlice();

                //Update GUI
                sagittalButton.Text = "Crop artefact";
                coronalButton.Text = "Coronal, YZ";


                is_sagittal = 1;
                is_coronal = 0;
            }
            else if (is_rendering == 1 && is_sagittal == 1) // Artefact cropping
            {
                //Get Line ends and crop above the line                
                volume.remove_artefact();
                volume.renderImage();
                //Check mask
                if (is_mask == 1)
                {
                    volume.renderImageMask();
                }

                // Update tooltip
                tip = "Cropped surface artefacts";
                gradeLabel.Text = tip;
            }
            GC.Collect();
        }

        //Automatically reorient the sample
        private void rotate_button_Click(object sender, EventArgs e)
        {
            label4.Text = "Calculating: ";
            mainProgress.Value = 10;
            string angles = volume.auto_rotate();
            mainProgress.Value = 80;
            dims = volume.getDims();
            sliceN[0] = (dims[1] + dims[0]) / 2;
            sliceN[1] = (dims[3] + dims[2]) / 2;
            sliceN[2] = (dims[5] + dims[4]) / 2;

            volume.updateCurrent(sliceN, ori, gray);

            //Render
            if (ori == -1)
            {
                volume.renderVolume();
                volume.setVolumeColor();
                if (is_mask == 1)
                {
                    volume.renderVolumeMask();
                }                
                
            }
            if (ori > -1)
            {                
                volume.renderImage();
                TellSlice();
                if (is_mask == 1)
                {
                    volume.renderImageMask();
                }                
            }
            GC.Collect();

            // Update tip
            tip = angles;
            gradeLabel.Text = tip;
            mainProgress.Value = 100;
            label4.Text = "Done";
        }

        //Automatically segment the BC interface
        private void segmentButton_Click(object sender, EventArgs e)
        {
            label4.Text = "Calculating: ";
            mainProgress.Value = 10;
            //VOI for segmentation

            volume.segmentation();
            mainProgress.Value = 80;

            //Update rendering pipeline
            is_mask = 1;
            maskLabel.Text = "Mask: automatic";

            
            //Render
            if (ori == -1)
            {
                volume.renderVolumeMask();
                volume.setVolumeColor();
            }
            if (ori > -1)
            {
                volume.renderImageMask();
            }
            
            getVoiButton.Enabled = true;
            saveButton.Enabled = true;

            //Saving flags
            save_mask = 1;
            save_vois = 0;

            GC.Collect();

            // Update tooltip
            tip = "Calcified zone interface segmented. Deep and surface zones can be extracted.";
            gradeLabel.Text = tip;
            mainProgress.Value = 100;
            label4.Text = "Done";
        }

        //Automatically crop the center of the sample
        private void cropButton_Click(object sender, EventArgs e)
        {
            label4.Text = "Calculating: ";
            mainProgress.Value = 10;
            //Connect mask to segmentation pipeline
            int size = 448;
            volume.center_crop(size);
            mainProgress.Value = 80;
            //Update sample dimensions
            dims = volume.getDims();
            sliceN[0] = (dims[1] + dims[0]) / 2;
            sliceN[1] = (dims[3] + dims[2]) / 2;
            sliceN[2] = (dims[5] + dims[4]) / 2;
            //Update pipeline
            volume.updateCurrent(sliceN, ori, gray);
            
            
            //Render
            if (ori == -1)
            {
                volume.renderVolume();
            }
            if (ori > -1)
            {
                volume.renderImage();                
            }
                        
            segmentButton.Enabled = true;
            predict.Enabled = true;
            tip = "Cropped to " + size.ToString() + " | " + size.ToString() + " size";
            gradeLabel.Text = tip;
            GC.Collect();
            mainProgress.Value = 100;
            label4.Text = "Done";
        }

        //Remove preparation artefacts from the surface
        private void getVoiButton_Click(object sender, EventArgs e)
        {
            label4.Text = "Calculating: ";
            mainProgress.Value = 10;
            volume.analysis_vois();
            mainProgress.Value = 80;

            is_mask = 1;

            //Render
            if (ori == -1)
            {
                volume.renderVolume();
                volume.renderVolumeMask();
                volume.setVolumeColor();
            }
            if (ori > -1)
            {
                volume.renderImage();
                volume.renderImageMask();
            }

            GC.Collect();
            saveButton.Enabled = true;
            //Saving flags
            save_mask = 0;
            save_vois = 1;

            /*
            cropBar.Enabled = true;

            byte[] surface = new byte[(dims[1]-dims[0]) * (dims[3] - dims[2])];
            Parallel.For(0, surface.Length, (int k) => 
            {
                surface[k] = 1;
            });

            */

            // Update tooltip
            tip = "Surface, calcified and deep zones extracted. Automatic grading can be conducted.";
            gradeLabel.Text = tip;
            mainProgress.Value = 100;
            label4.Text = "Done";
        }

        // Predict OA grade
        private void predict_Click(object sender, EventArgs e)
        {
            // Initialize
            label4.Text = "Calculating: ";
            mainProgress.Value = 10;
            string[] models = new string[] { ".\\Default\\calc_weights.dat", ".\\Default\\deep_weights.dat", ".\\Default\\surf_weights.dat" };
            string[] parameters = new string[] { ".\\Default\\calc_parameters.csv", ".\\Default\\deep_parameters.csv", ".\\Default\\surf_parameters.csv" };

            // Pipeline for grading
            string grade = volume.grade_vois(models, parameters, fname);

            // Output result
            gradeLabel.Text = grade;
            mainProgress.Value = 100;
            label4.Text = "Done";
        }

        private void saveButton_Click(object sender, EventArgs e)
        {
            label4.Text = "Saving: ";
            mainProgress.Value = 10;
            if (save_mask == 1)
            {
                //Save sample
                volume.save_data(fname, savedir);
                //Save segmentation mask
                volume.save_masks(new string[] { fname + "_UNET" }, savedir);
            }
            if(save_vois == 1)
            {
                //Save analysis VOIs
                string[] names = new string[] { fname + "_calcified", fname + "_deep", fname + "_surface" };
                volume.save_masks(names, savedir);
            }

            // Update tooltip
            tip = "Saved images in path " + savedir;
            gradeLabel.Text = tip;
            mainProgress.Value = 100;
            label4.Text = "Done";
        }

        //Scroll bars

        //Scroll slices
        private void sliceBar_Scroll(object sender, ScrollEventArgs e)
        {
            //Call at the end of scroll event
            if (e.Type == ScrollEventType.EndScroll)
            {
                sliceBar_ValueChanged(this, null);
                TellSlice();
            }
        }

        //Scroll gray max
        private void gmaxBar_Scroll(object sender, ScrollEventArgs e)
        {
            //Call at the end of scroll event
            if (e.Type == ScrollEventType.EndScroll)
            {
                gmaxBar_ValueChanged(this, null);
            }
        }

        //Scroll gray min
        private void gminBar_Scroll(object sender, ScrollEventArgs e)
        {
            //Call at the end of scroll event
            if (e.Type == ScrollEventType.EndScroll)
            {
                gminBar_ValueChanged(this, null);
            }
        }

        private void cropBar_Scroll(object sender, ScrollEventArgs e)
        {

        }

    }
}
