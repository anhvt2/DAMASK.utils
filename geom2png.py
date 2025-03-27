
import numpy as np
import glob, os
from natsort import natsorted, ns # natural-sort
import pyvista
import argparse
import matplotlib.pyplot as plt
import gc

'''
Example
-------

python3 geom2npy.py --geom spk_dump_12_out.geom

Parameters
----------
--geom: geometry file

Return
------
microstructure in 3d numpy array: spk_dump_12_out.npy
vti file: spk_dump_12_out.vti

'''

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--geom", type=str, required=True)
parser.add_argument("-threshold", "--threshold", help='threshold', type=int, default=-1, required=False)
parser.add_argument("-nameTag", "--nameTag", help='', type=str, default='', required=False)
parser.add_argument("-show_edges", "--show_edges", help='pyvista show_edges', type=lambda x:bool(strtobool(x)), default=True, required=False, nargs='?', const=True)
args = parser.parse_args()

fileName = args.geom
threshold = args.threshold
show_edges = bool(args.show_edges)
nameTag = args.nameTag

def save_array2vti(file_name, array):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    import numpy as np
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
    vtk_data_array = numpy_to_vtk(array.T.flatten(), deep=True, array_type=vtk.VTK_INT)
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

outFileName = fileName[:-5] # deprecate fileName.split('.')[0] to avoid '.' in outFileName
fileHandler = open(fileName)
txt = fileHandler.readlines()
fileHandler.close()
numSkippingLines = int(txt[0].split(' ')[0])+1 
# Search for 'size' within header:
for j in range(numSkippingLines):
    if 'grid' in txt[j]:
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

# Convert from 1 line format to 3d format
geom = np.array(geom, dtype=int).reshape(Nz, Ny, Nx).T # to reverse: geom = geom.T.flatten()

grid = pyvista.UniformGrid() # old pyvista
# grid = pyvista.ImageData() # new pyvista
# grid = pyvista.RectilinearGrid()
grid.dimensions = np.array(geom.shape) + 1
grid.origin = (0, 0, 0)     # The bottom left corner of the data set
grid.spacing = (1, 1, 1)    # These are the cell sizes along each axis
grid.cell_data["microstructure"] = geom.flatten(order="F") # ImageData()

pl = pyvista.Plotter(off_screen=True)
cmap = plt.cm.get_cmap('coolwarm')
pl.add_mesh(grid.threshold(value=threshold+1e-6), scalars='microstructure', show_edges=show_edges, line_width=1, cmap=cmap)
pl.background_color = "white"
pl.remove_scalar_bar()
# pl.show(screenshot='%s.png' % fileName[:-4])
# pl.show()
# pl.add_axes(color='k')
# pl.show_axes() # https://docs.pyvista.org/api/plotting/_autosummary/pyvista.renderer.add_axes

if geom.shape[2] == 1:
    pl.camera_position = 'xy'

if nameTag == '':
    pl.screenshot(fileName[:-5] + '.png', window_size=[1860*6,968*6])
else:
    pl.screenshot(fileName[:-5] + '_' + nameTag + '.png', window_size=[1860*6,968*6])

# pl.close()
gc.collect()






