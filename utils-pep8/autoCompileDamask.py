#!/usr/bin/env python3

"""
The objective of this script is to randomly search for a suitable combination of modules.
    (1) Gather a complete list of modules
    (2) Randomly shuffles candidates within a module
    (3) Randomly shuffles the order of loading
"""

import os
import random

import numpy as np

AUE_INTEL_LIST = [
    'intel/19.1',
    'intel/21.3.0',
    'intel/23.0.0',
    'intel/23.1.0',
    'intel/23.2.0',
]

AUE_INTEL_ONEAPI_COMPILERS_LIST = [
    'aue/intel-oneapi-compilers/2021.4.0',
    'aue/intel-oneapi-compilers/2021.5.0',
    'aue/intel-oneapi-compilers/2023.2.0',
    'aue/intel-oneapi-compilers/2024.1.0',
    'aue/intel-oneapi-compilers/2024.2.1',
]

AUE_INTEL_ONEAPI_MKL_LIST = [
    'aue/intel-oneapi-mkl/2021.5.0-oneapi-2021.5.0',
    'aue/intel-oneapi-mkl/2023.2.0-oneapi-2023.2.0',
    'aue/intel-oneapi-mkl/2024.1.0',
    'aue/intel-oneapi-mkl/2024.2.2',
]

AUE_INTEL_ONEAPI_MPI_LIST = [
    'aue/intel-oneapi-mpi/2021.5.0-intel-2021.5.0',
    'aue/intel-oneapi-mpi/2021.12.0',
    'aue/intel-oneapi-mpi/2021.13.1',
]

AUE_OPENMPI_LIST = [
    'aue/openmpi/4.1.6-clang-16.0.6',
    'aue/openmpi/4.1.6-gcc-10.3.0',
    'aue/openmpi/4.1.6-gcc-11.4.0-cuda-11.8.0',
    'aue/openmpi/4.1.6-gcc-11.4.0',
    'aue/openmpi/4.1.6-gcc-12.1.0',
    'aue/openmpi/4.1.6-gcc-12.3.0-cuda-12.4.0',
    'aue/openmpi/4.1.6-gcc-12.3.0',
    'aue/openmpi/4.1.6-intel-2021.4.0',
    'aue/openmpi/4.1.6-intel-2021.5.0',
    'aue/openmpi/4.1.6-intel-2023.2.0',
    'aue/openmpi/4.1.6-oneapi-2023.2.0',
    'aue/openmpi/4.1.6-oneapi-2024.1.0-cuda-11.8.0',
    'aue/openmpi/4.1.6-oneapi-2024.1.0-cuda-12.4.0',
    'aue/openmpi/4.1.6-oneapi-2024.1.0',
    'aue/openmpi/5.0.6-gcc-14.2.0',
]

OPENMPI_GNU_LIST = [
    'openmpi-gnu/4.0',
    'openmpi-gnu/4.1',
]

GNU_LIST = [
    'gnu/10.2',
    'gnu/10.3.1',
    'gnu/11.2.1',
    'gnu/12.2.1',
    'gnu/13.3.1',
]


def _load_random_module(module_list):
    seed = np.random.randint(low=0, high=len(module_list) + 1)
    if seed < len(module_list):
        os.system(f'module load {module_list[seed]}')
    # else:
# to be compiled in main 'damask' folder, not 'src'

MODULE_GROUPS = [
    AUE_INTEL_LIST,
    AUE_INTEL_ONEAPI_COMPILERS_LIST,
    AUE_INTEL_ONEAPI_MKL_LIST,
    AUE_INTEL_ONEAPI_MPI_LIST,
    AUE_OPENMPI_LIST,
    OPENMPI_GNU_LIST,
    GNU_LIST,
]

while not os.path.isfile('./bin/DAMASK_spectral'):
    os.system('clear')
    os.system(f"module purge")

    random.shuffle(MODULE_GROUPS)  # shuffle order of modules
    for module_list in MODULE_GROUPS:
        _load_random_module(module_list)  # shuffle module version
    os.system('module list ')
    os.system(f"make clean")
    os.system(f"make")
    os.system('sleep 2')
