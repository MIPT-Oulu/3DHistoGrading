import vtk
import numpy as np
import os


def render_volume(data, savepath=None, white=True, use_outline=False):
    """Renders three-dimensional visualization of the given array.

    Transparency options should be customized to cut background and show only relevant parts of the sample.

    Sample can also be visualized with different RGB colors based on gray values.

    Data mapper should be selected accordingly. vtkSmartVolumeMapper uses GPU for accelerated rendering if possible,
    vtkFixedPointVolumeRayCastMapper is maybe more reliable at some cases but slower.

    Viewing angle should also be customized to visualize sample well during the screenshot.

    Parameters
    ----------
    data : 3d numpy array
        Input array that is going to be visualized
    savepath : str
        Full file name for the saved image. If not given, data can be interactively visualized.
        Example: C:/path/rendering.png
    white : bool
        Choose whether to have white or black background. Defaults to white.
    use_outline : bool
        Choose whether to use an outline to show data extent.
    """

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

    # Gray value transparency (this can be modified for different samples)
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
    
    # Set outline
    if use_outline:
        outline = vtk.vtkOutlineFilter()
        outline.SetInputConnection(data_importer.GetOutputPort())
        mapper2 = vtk.vtkPolyDataMapper()
        mapper2.SetInputConnection(outline.GetOutputPort())
        actor2 = vtk.vtkActor()
        actor2.SetMapper(mapper2)
        renderer.AddActor(actor2)

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
        os.makedirs(savepath.rsplit('/', 1)[0], exist_ok=True)

        img = vtk.vtkWindowToImageFilter()
        img.SetInput(render_window)
        img.Update()

        writer = vtk.vtkPNGWriter()
        print('Saved to: ' + savepath)
        writer.SetFileName(str(savepath))
        writer.SetInputData(img.GetOutput())
        writer.Write()
        return
    else:  # Run the window with user interactions
        interactor.Start()
