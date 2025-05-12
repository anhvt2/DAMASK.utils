#!/usr/bin/env python3

import argparse

  # natural-sort
import numpy as np

PARSER = argparse.ArgumentParser()
PARSER.add_argument("-g", "--geom", type=str, required=True)
ARGS = PARSER.parse_args()

FILE_NAME = ARGS.geom


def _save_array2_vti(file_name, array):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    vtk_data_array = numpy_to_vtk(array.T.flatten(), deep=True, array_type=vtk.VTK_INT)
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(array.shape)
    image_data.GetPointData().SetScalars(vtk_data_array)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(image_data)
    writer.Write()
    return None


def _delete(lst, to_delete):
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


# deprecate fileName.split('.')[0] to avoid '.' in outFileName
OUT_FILE_NAME = FILE_NAME[:-5]
with open(FILE_NAME) as fileHandler:
    txt = fileHandler.readlines()


NUM_SKIPPING_LINES = int(txt[0].split(' ')[0]) + 1
# Search for 'size' within header:
for j in range(NUM_SKIPPING_LINES):
    if 'grid' in txt[j]:
        cleanString = _delete(txt[j].replace('\n', '').split(' '), '')
        Nx = int(cleanString[2])
        Ny = int(cleanString[4])
        Nz = int(cleanString[6])

GEOM_BLOCK = txt[NUM_SKIPPING_LINES:]
GEOM = sum(GEOM_BLOCK)

GEOM = GEOM.split(' ')
GEOM = list(filter(('').__ne__, GEOM))

# Convert from 1 line format to 3d format
GEOM = np.array(GEOM, dtype=int).reshape(Nz, Ny, Nx).T
  # to reverse: geom = geom.T.flatten()

# Save output
np.save(OUT_FILE_NAME + '.npy', GEOM)
_save_array2_vti(OUT_FILE_NAME + '.vti', GEOM)
