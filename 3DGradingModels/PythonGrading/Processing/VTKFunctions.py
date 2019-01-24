import vtk
import numpy as np
from vtk.util import numpy_support


def render_volume(data, savepath=None, white=True):
    # Input data as uint8
    data_matrix = np.uint8(data)
    dims = np.shape(data)

    # Store data as VTK images
    data_importer = vtk.vtkImageImport()
    # Array is converted to a string of chars and imported.
    data_string = data_matrix.tostring()
    data_importer.CopyImportVoidPointer(data_string, len(data_string))
    # Set type to unsigned char (uint8)
    data_importer.SetDataScalarTypeToUnsignedChar()
    # Data contains only gray values (not RGB)
    data_importer.SetNumberOfScalarComponents(1)
    # Data storing dimensions
    data_importer.SetDataExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)
    data_importer.SetWholeExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)

    # Gray value transparency
    alpha_channel = vtk.vtkPiecewiseFunction()
    alpha_channel.AddPoint(0, 0.0)
    alpha_channel.AddPoint(70, 0.0)
    alpha_channel.AddPoint(80, 0.1)
    alpha_channel.AddPoint(100, 0.2)
    alpha_channel.AddPoint(150, 0.9)

    # Possibility to use RGB colors.
    color_transfer = vtk.vtkColorTransferFunction()
    color_transfer.AddRGBPoint(50, 0.0, 0.0, 0.0)
    color_transfer.AddRGBPoint(100, 0.5, 0.5, 0.5)
    color_transfer.AddRGBPoint(255, 1.0, 1.0, 1.0)

    # Apply transparency and colors to volume
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_transfer)
    volume_property.SetScalarOpacity(alpha_channel)

    # Create volume mapper and specify data to volume
    # mapper = vtk.vtkFixedPointVolumeRayCastMapper()
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputConnection(data_importer.GetOutputPort())

    # Create actual volume. Set mapper and properties
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volume_property)

    # Create renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # # Set outline
    # outline = vtk.vtkOutlineFilter()
    # outline.SetInputConnection(data_importer.GetOutputPort())
    # mapper2 = vtk.vtkPolyDataMapper()
    # mapper2.SetInputConnection(outline.GetOutputPort())
    # actor2 = vtk.vtkActor()
    # actor2.SetMapper(mapper2)
    # renderer.AddActor(actor2)

    # Set background color
    if white:
        renderer.SetBackground(1, 1, 1)
    else:
        renderer.SetBackground(0, 0, 0)
    
    # Add volume to renderer
    renderer.AddVolume(volume)

    # Camera options
    renderer.GetActiveCamera().SetPosition(0.5, 0.5, 0.5)
    renderer.GetActiveCamera().Azimuth(-110)
    renderer.GetActiveCamera().Elevation(-170)
    renderer.ResetCameraClippingRange()
    renderer.ResetCamera()

    # # Camera options
    # camera = vtk.vtkCamera()
    # camera.SetPosition(0.5, 1, 0)
    # camera.SetFocalPoint(0, 0.5, 0.5)
    # camera.SetViewUp(1, 0, 1)
    # renderer.SetActiveCamera(camera)
    # renderer.ResetCamera()

    # 2nd set of options
    # camera = vtk.vtkCamera()
    # camera.SetPosition(0.5, 1, 0)
    # camera.SetFocalPoint(0, 0.5, 0.5)
    # camera.SetViewUp(1, 0, 1)
    # renderer.SetActiveCamera(camera)
    # renderer.ResetCamera()

    # Window size
    render_window.SetSize(600, 600)

    # Application quit
    def exit_check(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)

    # Tell the application to use the function as an exit check.
    render_window.AddObserver("AbortCheckEvent", exit_check)

    interactor.Initialize()
    render_window.Render()

    if savepath:  # Take a screenshot
        img = vtk.vtkWindowToImageFilter()
        img.SetInput(render_window)
        img.Update()

        writer = vtk.vtkPNGWriter()
        print('Saved to: ' + savepath)
        writer.SetFileName(str(savepath))
        writer.SetInputData(img.GetOutput())
        writer.Write()
        return
    else:
        interactor.Start()


def array_to_vtk(A):
    imagedata = vtk.vtkImageData()
    depth_array = numpy_support.numpy_to_vtk(A.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    imagedata.SetDimensions(A.shape)
    imagedata.SetOrigin(0, 0, 0)
    imagedata.GetPointData().SetScalars(depth_array)
    return imagedata


def vtk_to_array(vtkdata, shape):
    array = numpy_support.vtk_to_numpy(vtkdata)
    array = array.reshape(shape)
    return array


def rotate_vtk(vtkdata, angles):
    # Initialize
    mapper = vtk.vtkFixedPointVolumeRayCastMapper()
    mapper.SetInputData(vtkdata)
    actor = vtk.vtkActor()

    cx, cy, cz = actor.GetCenter()

    transf = vtk.vtkTransform()
    transf.Translate(cx, cy, cz)
    transf.RotateX(angles[0])
    transf.RotateY(angles[1])
    transf.RotateZ(angles[2])
    transf.Translate(-cx, -cy, -cz)
    
    slicer = vtk.vtkImageReslice()
    slicer.SetInputData(vtkdata)
    slicer.Set

    return vtkdata