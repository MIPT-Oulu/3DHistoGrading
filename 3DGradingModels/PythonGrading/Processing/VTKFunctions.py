import vtk
import numpy as np

def render_volume(data, savepath=None, white=True):
    # We begin by creating the data we want to render.
    # For this tutorial, we create a 3D-image containing three overlaping cubes.
    # This data can of course easily be replaced by data from a medical CT-scan or anything else three dimensional.
    # The only limit is that the data must be reduced to unsigned 8 bit or 16 bit integers.
    data_matrix = np.uint8(data)
    dims = np.shape(data)

    # For VTK to be able to use the data, it must be stored as a VTK-image. This can be done by the vtkImageImport-class
    # which imports raw data and stores it.
    dataImporter = vtk.vtkImageImport()
    # The previously created array is converted to a string of chars and imported.
    data_string = data_matrix.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    # The type of the newly imported data is set to unsigned char (uint8)
    dataImporter.SetDataScalarTypeToUnsignedChar()
    # Because the data that is imported only contains an intensity value (it isnt RGB-coded or someting similar), the importer
    # must be told this is the case.
    dataImporter.SetNumberOfScalarComponents(1)
    # The following two functions describe how the data is stored and the dimensions of the array it is stored in. For this
    # simple case, all axes are of length 75 and begins with the first element. For other data, this is probably not the case.
    # I have to admit however, that I honestly dont know the difference between SetDataExtent() and SetWholeExtent() although
    # VTK complains if not both are used.
    dataImporter.SetDataExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)
    dataImporter.SetWholeExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)

    # The following class is used to store transparencyv-values for later retrival. In our case, we want the value 0 to be
    # completely opaque whereas the three different cubes are given different transperancy-values to show how it works.
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, 0.0)
    alphaChannelFunc.AddPoint(70, 0.0)
    alphaChannelFunc.AddPoint(80, 0.1)
    alphaChannelFunc.AddPoint(100, 0.2)
    alphaChannelFunc.AddPoint(150, 0.9)

    # This class stores color data and can create color tables from a few color points. For this demo, we want the three cubes
    # to be of the colors red green and blue.
    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(50, 0.0, 0.0, 0.0)
    colorFunc.AddRGBPoint(100, 0.5, 0.5, 0.5)
    colorFunc.AddRGBPoint(255, 1.0, 1.0, 1.0)

    # The preavius two classes stored properties. Because we want to apply these properties to the volume we want to render,
    # we have to store them in a class that stores volume prpoperties.
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)

    # We can finally create our volume. We also have to specify the data for it, as well as how the data will be rendered.
    volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    # The class vtkVolume is used to pair the preaviusly declared volume as well as the properties to be used when rendering that volume.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    # With almost everything else ready, its time to initialize the renderer and window, as well as creating a method for exiting the application
    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWin)
    
    ## Set outline
    #outline = vtk.vtkOutlineFilter()
    #outline.SetInputConnection(dataImporter.GetOutputPort())
    #mapper2 = vtk.vtkPolyDataMapper()
    #mapper2.SetInputConnection(outline.GetOutputPort())
    #actor2 = vtk.vtkActor()
    #actor2.SetMapper(mapper2)
    #renderer.AddActor(actor2)

    # Camera options
    #camera = vtk.vtkCamera()
    #camera.SetPosition(0.5, 1, 0)
    #camera.SetFocalPoint(0, 0.5, 0.5)
    #camera.SetViewUp(1, 0, 1)

    #renderer.SetActiveCamera(camera)
    #renderer.ResetCamera()
    
    #camera = vtk.vtkCamera()
    #camera.SetPosition(0.5, 1, 0)
    #camera.SetFocalPoint(0, 0.5, 0.5)
    #camera.SetViewUp(1, 0, 1)
    #renderer.SetActiveCamera(camera)
    #renderer.ResetCamera()

    # Set background color
    if white:
        renderer.SetBackground(1,1,1)
    else:
        renderer.SetBackground(0,0,0)
    
    # We add the volume to the renderer ...
    renderer.AddVolume(volume)
    renderer.GetActiveCamera().SetPosition(0.5, 0.5, 0.5)
    renderer.GetActiveCamera().Azimuth(-110)
    renderer.GetActiveCamera().Elevation(-170)
    renderer.ResetCameraClippingRange()
    renderer.ResetCamera()
    
    # Window size
    renderWin.SetSize(600, 600)

    # Application quit
    def exitCheck(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)

    # Tell the application to use the function as an exit check.
    renderWin.AddObserver("AbortCheckEvent", exitCheck)

    renderInteractor.Initialize()
    renderWin.Render()

    if savepath:  # Take a screenshot
        img = vtk.vtkWindowToImageFilter()
        img.SetInput(renderWin)
        img.Update()

        writer = vtk.vtkPNGWriter()
        print('Saved to: ' + savepath)
        writer.SetFileName(str(savepath))
        writer.SetInputData(img.GetOutput())
        writer.Write()
        return
    else:
        renderInteractor.Start()
    
def ArrayToVTK(A):
    imagedata = vtk.vtkImageData()
    depthArray = numpy_support.numpy_to_vtk(A.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    imagedata.SetDimensions(A.shape)
    imagedata.SetOrigin(0,0,0)
    imagedata.GetPointData().SetScalars(depthArray)
    return imagedata

def VTKToArray(vtkdata, shape):
    array = numpy_support.vtk_to_numpy(vtkdata)
    array = array.reshape(shape)
    return array

def RotateVTK(vtkdata, angles):
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