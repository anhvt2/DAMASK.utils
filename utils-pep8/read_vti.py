#!/usr/bin/env python3
# Copyright by Leidong Xu and Zihan Wang

import vtk
from vtk.util.numpy_support import vtk_to_numpy


def load_vti_to_array(file_name):
    """
    Load a VTI file into a 3D numpy array.

    Args:
    - file_name (str): Path of the VTI file to be loaded.

    Returns:
    - np.ndarray: 3D numpy array loaded from the VTI file.
    """

    # Initialize the VTI reader and set the filename
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(file_name)
    reader.Update()

    # Retrieve the image data from the reader
    image_data = reader.GetOutput()

    # Convert VTK array to numpy array
    vtk_data_array = image_data.GetPointData().GetScalars()
    np_array = vtk_to_numpy(vtk_data_array)

    # Reshape the numpy array to the dimensions of the image data
    dimensions = image_data.GetDimensions()
    # 'F' to match VTK's Fortran-style order
    np_array = np_array.reshape(dimensions, order='F')

    return np_array
