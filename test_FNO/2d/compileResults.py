
"""
This script
    (1) collects
            microstructures in certain format,
            solutions ('Mises(Cauchy)' and 'Mises(ln(V))'),
    located in each folder
    (2) dump 3 bigger .npy files

Parameters
----------
    Nx, Ny, Nz: 3d computational domain
    Np: number of simulations
    e: number of microstructure parameterization (Euler angles)
"""

import numpy as np
Np = 5000
Nx = 32
Ny = 32
Nz = 1
e = 3

# Initialize
ms = np.zeros([Np, Nx, Ny, Nz, e])
sols_MisesCauchy = np.zeros([Np, Nx, Ny, Nz])
sols_MisesLnV = np.zeros([Np, Nx, Ny, Nz])


def grainIdTo4dTensor(g, texture):
    """
    Parameters
    ----------
    g: microstructure in grainID representation
    texture: grainID -> Euler angles map

    Return
    ------
    m: 4d microstructure (no grain ID)
    """
    Nx, Ny, Nz = g.shape
    m = np.zeros((Nx, Ny, Nz, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Look up grain ID (grain ID starts from 1 so -1 to go back to row index, ignore first column of grain ID)
                m[i, j, k, :] = texture[g[i, j, k]-1, 1:]
    return m


for i in range(Np):
    texture = np.load(f'{str(i+1)}/texture.npy')
    g = np.load(f'{str(i+1)}/simple2d.npy')
    m = grainIdTo4dTensor(g, texture)
    sol_MisesCauchy = np.load(f'{str(i+1)}/postProc/MisesCauchy.npy')
    sol_MisesLnV = np.load(f'{str(i+1)}/postProc/MisesLnV.npy')
    # Stack solutions
    ms[i, :, :, :, :] = m
    sols_MisesCauchy[i, :, :, :] = sol_MisesCauchy
    sols_MisesLnV[i, :, :, :] = sol_MisesLnV
    print(f"Done {i}.")

np.save('ms.npy', ms)
np.save('sols_MisesCauchy.npy', sols_MisesCauchy)
np.save('sols_MisesLnV.npy', sols_MisesLnV)
