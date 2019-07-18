using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Kitware.VTK;

namespace HistoGrading.Components
{
    class Interaction
    {
        //Declarations
        static vtkRenderWindow renWin;

        //Public methods
        public void set_renderer(vtkRenderWindow input_renwin)
        {
            renWin = input_renwin;
        }
        public vtkRenderWindowInteractor coordinate_interactor()
        {
            //Declare new interactor style
            vtkInteractorStyle interactorSyle = vtkInteractorStyle.New();

            //Set new mouse events
            interactorSyle.LeftButtonPressEvt += null;
            interactorSyle.LeftButtonReleaseEvt += new vtkObject.vtkObjectEventHandler(get_coordinates);

            //Create new interactor
            vtkRenderWindowInteractor interactor = vtkRenderWindowInteractor.New();
            interactor.SetInteractorStyle(interactorSyle);

            return interactor;
        }

        //Interactors
        private static void get_coordinates(vtkObject sender, vtkObjectEventArgs e)
        {
            int[] cur = renWin.GetPosition();
            string txt = "";
            foreach(int num in cur)
            {
                txt += System.String.Format("{0}",num);
                txt += "|";
            }
            StreamWriter file = new StreamWriter(@"C:\\users\\jfrondel\\desktop\\GITS\\VTKOUTPUT.txt");
            file.WriteLine(txt);
        }
    }
}
