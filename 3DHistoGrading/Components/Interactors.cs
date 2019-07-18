using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Kitware.VTK;

namespace HistoGrading.Components
{
    class Interactors : IDisposable
    {
        //Declarations

        //Render window
        private static vtkRenderWindow renWin;
        //Styles
        private static vtkInteractorStyle default_style;
        private static vtkInteractorStyle drag_style;
        private static vtkInteractorStyle pick_style;
        private static vtkInteractorStyle style_2d;
        private static vtkInteractorStyle style_2d_nodraw;
        //Line object        
        private static Rendering.vtkLine line;

        //Mouse status
        private static bool mouseDownLeft = false;
        private static bool mouseDownRight = false;

        //Coordinates
        private static double[] position = new double[4];
        private static int idx = 0;
        private static int slice = 0;

        //Camera properties
        double[] cam_pos = new double[3];
        double[] cam_foc = new double[3];
        double[] def_pos = new double[3];
        double[] def_foc = new double[3];

        double[] prev_pos = new double[3];

        double cam_zoom = 1.0;

        //Flags
        private static bool has_actor = false;        

        public Interactors(vtkRenderWindow inputWin)
        {
            //Get default interactor and interactor style
            renWin = inputWin;

            //Get default style
            default_style = (vtkInteractorStyle)vtkInteractorStyleTrackballCamera.New();

            //Create drag style
            drag_style = vtkInteractorStyle.New();
            drag_style.LeftButtonPressEvt += drag_LeftDown;
            drag_style.LeftButtonReleaseEvt += drag_LeftUp;
            drag_style.MouseMoveEvt += drag_MouseMove;

            //Create pick style
            pick_style = vtkInteractorStyle.New();
            pick_style.LeftButtonPressEvt += pick_LeftDown;
            pick_style.LeftButtonReleaseEvt += pick_LeftUp;

            //Create 2D style
            style_2d = vtkInteractorStyle.New();
            style_2d.LeftButtonPressEvt += LeftDown2D;
            style_2d.LeftButtonReleaseEvt += LeftUp2D;
            style_2d.MouseMoveEvt += Move2D;
            style_2d.RightButtonPressEvt += drag_RightDown;
            style_2d.RightButtonReleaseEvt += drag_RightUp;            
            style_2d.MouseWheelForwardEvt += scrollUp2D;
            style_2d.MouseWheelBackwardEvt += scrollDown2D;

            //Create 2D style without drawing tool
            style_2d_nodraw = vtkInteractorStyle.New();
            style_2d_nodraw.LeftButtonPressEvt += LeftDown2D;
            style_2d_nodraw.LeftButtonReleaseEvt += LeftUp2D;            
            style_2d_nodraw.MouseMoveEvt += Move2D;
            style_2d_nodraw.MouseWheelForwardEvt += scrollUp2D;
            style_2d_nodraw.MouseWheelBackwardEvt += scrollDown2D;

            //Get camera position and focal point
            get_camera();
        }

        //Get camera position and focal point
        public void get_camera()
        {
            cam_pos = renWin.GetRenderers().GetFirstRenderer().GetActiveCamera().GetPosition();
            cam_foc = renWin.GetRenderers().GetFirstRenderer().GetActiveCamera().GetFocalPoint();
            def_pos = renWin.GetRenderers().GetFirstRenderer().GetActiveCamera().GetPosition();
            def_foc = renWin.GetRenderers().GetFirstRenderer().GetActiveCamera().GetFocalPoint();

            prev_pos = renWin.GetRenderers().GetFirstRenderer().GetActiveCamera().GetPosition();
        }

        //Set extent of the current image
        public void set_slice(int N)
        {
            slice = N;
            has_actor = false;
        }


        //Set default
        public void set_default()
        {
            renWin.GetInteractor().SetInteractorStyle(default_style);
        }

        //Set 2D interactor
        public void set_2d()
        {
            //Set interactor
            renWin.GetInteractor().SetInteractorStyle(style_2d);
            //Get camera position
            get_camera();            
        }

        //Set 2D interactor
        public void set_2d_nodraw()
        {
            //Set interactor
            renWin.GetInteractor().SetInteractorStyle(style_2d_nodraw);
            //Get camera position
            get_camera();            
        }

        //Set drag
        public void set_drag()
        {
            renWin.GetInteractor().SetInteractorStyle(drag_style);
        }

        //Set picker
        public void set_pick()
        {
            renWin.GetInteractor().SetInteractorStyle(pick_style);
        }
        //Return coordinates
        public double[] get_position()
        {
            return position;
        }

        //Button press events

        //Drag events
        private void drag_LeftDown(vtkObject sender, vtkObjectEventArgs e)
        {
            if (mouseDownLeft == false)
            {
                //Get position
                int[] pos = renWin.GetInteractor().GetEventPosition();
                vtkCoordinate coord = vtkCoordinate.New();
                coord.SetCoordinateSystemToDisplay();
                coord.SetValue(pos[0], pos[1], 0);
                double[] tmp = coord.GetComputedWorldValue(renWin.GetRenderers().GetFirstRenderer());

                position[0] = (int)tmp[0];
                position[1] = (int)tmp[1];
                position[2] = (int)tmp[0];
                position[3] = (int)tmp[1];

                //Create new line object
                if (has_actor == true)
                {
                    line.update(position);
                }
                else
                {
                    line = new Rendering.vtkLine(renWin, slice);
                    line.draw(position);
                }


                renWin.Render();

                has_actor = true;

                mouseDownLeft = true;
            }
        }

        private void drag_LeftUp(vtkObject sender, vtkObjectEventArgs e)
        {
            if (mouseDownLeft == true)
            {
                //Get position
                int[] pos = renWin.GetInteractor().GetEventPosition();
                vtkCoordinate coord = vtkCoordinate.New();
                coord.SetCoordinateSystemToDisplay();
                coord.SetValue(pos[0], pos[1], 0);
                double[] tmp = coord.GetComputedWorldValue(renWin.GetRenderers().GetFirstRenderer());

                position[2] = (int)tmp[0];
                position[3] = (int)tmp[1];

                line.update(position);

                has_actor = true;

                renWin.Render();

                mouseDownLeft = false;
            }
        }

        private void drag_MouseMove(vtkObject sender, vtkObjectEventArgs e)
        {
            if (mouseDownLeft == true)
            {
                int[] pos = renWin.GetInteractor().GetEventPosition();
                vtkCoordinate coord = vtkCoordinate.New();
                coord.SetCoordinateSystemToDisplay();
                coord.SetValue(pos[0], pos[1], 0);
                double[] tmp = coord.GetComputedWorldValue(renWin.GetRenderers().GetFirstRenderer());
                position[2] = tmp[0];
                position[3] = tmp[1];

                line.update(position);

                renWin.Render();
            }
        }

        private void drag_RightDown(vtkObject sender, vtkObjectEventArgs e)
        {
            if (mouseDownRight == false)
            {
                //Get position
                int[] pos = renWin.GetInteractor().GetEventPosition();
                vtkCoordinate coord = vtkCoordinate.New();
                coord.SetCoordinateSystemToDisplay();
                coord.SetValue(pos[0], pos[1], 0);
                double[] tmp = coord.GetComputedWorldValue(renWin.GetRenderers().GetFirstRenderer());

                position[0] = (int)tmp[0];
                position[1] = (int)tmp[1];
                position[2] = (int)tmp[0];
                position[3] = (int)tmp[1];

                //Create new line object
                if (has_actor == true)
                {
                    line.update(position);
                }
                else
                {
                    line = new Rendering.vtkLine(renWin, slice);
                    line.draw(position);
                }


                renWin.Render();

                has_actor = true;

                mouseDownRight = true;
            }
        }

        private void drag_RightUp(vtkObject sender, vtkObjectEventArgs e)
        {
            if (mouseDownRight == true)
            {
                //Get position
                int[] pos = renWin.GetInteractor().GetEventPosition();                
                vtkCoordinate coord = vtkCoordinate.New();
                coord.SetCoordinateSystemToDisplay();
                coord.SetValue(pos[0], pos[1], 0);
                double[] tmp = coord.GetComputedWorldValue(renWin.GetRenderers().GetFirstRenderer());

                position[2] = (int)tmp[0];
                position[3] = (int)tmp[1];

                line.update(position);

                has_actor = true;

                renWin.Render();

                mouseDownRight = false;
            }
        }

        //Pick events
        private void pick_LeftDown(vtkObject sender, vtkObjectEventArgs e)
        {
            if (mouseDownLeft == false)
            {
                if (idx == 0)
                {
                    int[] tmp = renWin.GetInteractor().GetEventPosition();
                    position[0] = tmp[0];
                    position[1] = tmp[1];
                    mouseDownLeft = true;
                    idx = 1;
                }
                if (idx == 1)
                {
                    int[] tmp = renWin.GetInteractor().GetEventPosition();
                    position[0] = tmp[0];
                    position[1] = tmp[1];
                    mouseDownLeft = true;
                    idx = 0;
                }
            }
        }

        private void pick_LeftUp(vtkObject sender, vtkObjectEventArgs e)
        {
            if (mouseDownLeft == true)
            {
                mouseDownLeft = false;
            }
        }

        private void LeftDown2D(vtkObject sender, vtkObjectEventArgs e)
        {
            if (mouseDownLeft == false)
            {
                //Get position
                int[] pos = renWin.GetInteractor().GetEventPosition();
                vtkCoordinate coord = vtkCoordinate.New();
                coord.SetCoordinateSystemToDisplay();
                coord.SetValue(pos[0], pos[1], 0);
                double[] tmp = coord.GetComputedWorldValue(renWin.GetRenderers().GetFirstRenderer());
                prev_pos[0] = tmp[0]; prev_pos[1] = tmp[1];
                mouseDownLeft = true;
            }
        }

        private void LeftUp2D(vtkObject sender, vtkObjectEventArgs e)
        {
            if (mouseDownLeft == true)
            {
                mouseDownLeft = false
;
            }
        }

        private void Move2D(vtkObject sender, vtkObjectEventArgs e)
        {
            if (mouseDownLeft == true)
            {
                //Get position
                int[] pos = renWin.GetInteractor().GetEventPosition();
                vtkCoordinate coord = vtkCoordinate.New();
                coord.SetCoordinateSystemToDisplay();
                coord.SetValue(pos[0], pos[1]);
                double[] tmp = coord.GetComputedWorldValue(renWin.GetRenderers().GetFirstRenderer());

                cam_pos[0] -= (tmp[0] - prev_pos[0]);
                cam_pos[1] -= (tmp[1] - prev_pos[1]);
                cam_foc[0] -= (tmp[0] - prev_pos[0]);
                cam_foc[1] -= (tmp[1] - prev_pos[1]);
                renWin.GetRenderers().GetFirstRenderer().GetActiveCamera().SetPosition(cam_pos[0], cam_pos[1], cam_pos[2]);
                renWin.GetRenderers().GetFirstRenderer().GetActiveCamera().SetFocalPoint(cam_foc[0], cam_foc[1], cam_foc[2]);
                renWin.Render();                
            }
            if (mouseDownRight == true)
            {
                int[] pos = renWin.GetInteractor().GetEventPosition();
                vtkCoordinate coord = vtkCoordinate.New();
                coord.SetCoordinateSystemToDisplay();
                coord.SetValue(pos[0], pos[1], 0);
                double[] tmp = coord.GetComputedWorldValue(renWin.GetRenderers().GetFirstRenderer());
                position[2] = tmp[0];
                position[3] = tmp[1];

                line.update(position);

                renWin.Render();
            }
        }

        private void scrollUp2D(vtkObject sender, vtkObjectEventArgs e)
        {
            cam_zoom = 1.1;
            renWin.GetRenderers().GetFirstRenderer().GetActiveCamera().Zoom(cam_zoom);
            renWin.Render();
        }

        private void scrollDown2D(vtkObject sender, vtkObjectEventArgs e)
        {
            cam_zoom = 0.9;
            renWin.GetRenderers().GetFirstRenderer().GetActiveCamera().Zoom(cam_zoom);
            renWin.Render();
        }

        //Disposing methods
        public void Dispose()
        {            
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        protected virtual void Dispose(bool disposing)
        {
            Disposed = true;
        }
        protected bool Disposed { get; private set; }

}
}

