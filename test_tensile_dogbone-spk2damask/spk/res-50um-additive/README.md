
# Purposes

The main utility of this folder is to create visualization for 

1. Example how to use:
    ```shell
    python3 vizAM.py --npyFolderName='npy' --phaseFileName='void+phase_dump_12_out.npy'
    ```
1. 
    then
    ```shell
    bash highlight.sh # python3 ../../../npy2png.py --threshold=1
    ```
Note: in `../../../npy2png.py`
* Newer pyvista (0.44.1) will ask for `grid = pyvista.ImageData() # new pyvista`
* Old pyvista (0.37.0) will ask for `grid = pyvista.UniformGrid() # old pyvista`

This folder 
1. converts a series of microstructures formatted in `*.npy` (must be)
1. along with a chosen phase (for example, dogbone)
1. to a series of images (that could be converted to video) for illustration purpose

Work for any voxelized-STL file.

# How?

Step-1: Mask the resulting microstructures according to phase (and hide voids)
Step-2: Compare the current (masked) microstructure with initial (masked microstructure) and calculate the difference
Step-3: Show the difference microstructure

Other random thoughts:
* Convert `*.vti` to `*.npy`
* Dump masked `*.npy` based on (1) phase and (2) difference with the initial condition `*.npy`
* Convert masked `*.npy` to `*.geom`
* Convert `*.geom` to `*.vtr`

OR

* ~~Convert masked `*.npy` directly to `*.vtr` # https://tutorial.pyvista.org/tutorial/02_mesh/solutions/c_create-uniform-grid.html~~ (done, see `npy2png.py`)

Show us with threshold, hide the rest.
* Always show the final us (with opacity=0 for consistent grain ID colormap)
* Only show the diff b/w the current us and the initial us, but NOT include masked phase

