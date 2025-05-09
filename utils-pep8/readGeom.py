
import numpy as np
import glob
import os
from natsort import natsorted, ns  # natural-sort
import pyvista
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def save_array2vti(file_name, array):
    """
    Credit to Leidong Xu (UConn) with some generalizations and correction
    (1) .ravel() -> .T.flatten()
    (2) SetDimension(array.shape)
    Save a 3D numpy array to a VTI file.
    Args:
    - file_name (str): Path where the VTI file should be saved.
    - array (np.ndarray): 3D numpy array to be saved.

    Returns:
    - None
    """
    # Convert the numpy array to a VTK array
    vtk_data_array = numpy_to_vtk(
        array.T.flatten(), deep=True, array_type=vtk.VTK_INT)
    # Create an image data object and set its dimensions and scalars
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(array.shape)
    image_data.GetPointData().SetScalars(vtk_data_array)
    # Initialize the VTI writer, set the filename and input data
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(image_data)
    writer.Write()
    return None


def delete(lst, to_delete):
    '''
    Recursively delete an element with content described by  'to_delete' variable
    https://stackoverflow.com/questions/53265275/deleting-a-value-from-a-list-using-recursion/
    Parameter
    ---------
    to_delete: content needs removing
    lst: list
    Return
    ------
    a list without to_delete element
    '''
    return [element for element in lst if element != to_delete]


def geom2npy(filename):
    '''
    Read a .geom file and return a numpy array with meta-data
    '''
    outFileName = filename.split('.')[0]
    fileHandler = open(filename)
    txt = fileHandler.readlines()
    fileHandler.close()
    numSkippingLines = int(txt[0].split(' ')[0])+1
    # Search for 'size' within header:
    for j in range(numSkippingLines):
        if 'size' in txt[j]:
            cleanString = delete(txt[j].replace('\n', '').split(' '), '')
            Nx = int(cleanString[2])
            Ny = int(cleanString[4])
            Nz = int(cleanString[6])

    geomBlock = txt[numSkippingLines:]
    geom = ''
    for i in range(len(geomBlock)):
        geom += geomBlock[i]

    geom = geom.split(' ')
    geom = list(filter(('').__ne__, geom))
    geom = np.array(geom, dtype=int).reshape(Nz, Ny, Nx).T
    headers = txt[:numSkippingLines]  # also return headers
    return Nx, Ny, Nz, geom, headers
