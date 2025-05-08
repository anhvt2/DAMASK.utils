
"""
The objective of this script is to randomly search for a suitable combination of modules.
    (1) Gather a complete list of modules
    (2) Randomly shuffles candidates within a module
    (3) Randomly shuffles the order of loading
"""
import os, sys, time
import numpy as np
import random

aue_intel_list = [
    'intel/19.1',
    'intel/21.3.0',
    'intel/23.0.0',
    'intel/23.1.0',
    'intel/23.2.0',
    ]

aue_intel_oneapi_compilers_list = [
    'aue/intel-oneapi-compilers/2021.4.0', 
    'aue/intel-oneapi-compilers/2021.5.0', 
    'aue/intel-oneapi-compilers/2023.2.0', 
    'aue/intel-oneapi-compilers/2024.1.0', 
    'aue/intel-oneapi-compilers/2024.2.1', 
    ]

aue_intel_oneapi_mkl_list = [
    'aue/intel-oneapi-mkl/2021.5.0-oneapi-2021.5.0',
    'aue/intel-oneapi-mkl/2023.2.0-oneapi-2023.2.0',
    'aue/intel-oneapi-mkl/2024.1.0',
    'aue/intel-oneapi-mkl/2024.2.2',
    ]

aue_intel_oneapi_mpi_list = [
    'aue/intel-oneapi-mpi/2021.5.0-intel-2021.5.0',
    'aue/intel-oneapi-mpi/2021.12.0',
    'aue/intel-oneapi-mpi/2021.13.1',
    ]

aue_openmpi_list = [
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

openmpi_gnu_list = [
    'openmpi-gnu/4.0',
    'openmpi-gnu/4.1',
    ]

gnu_list = [
    'gnu/10.2',
    'gnu/10.3.1',
    'gnu/11.2.1',
    'gnu/12.2.1',
    'gnu/13.3.1',
    ]

def load_random_module(module_list):
    seed = np.random.randint(low=0,high=len(module_list)+1)
    if seed < len(module_list): # if seed = len(module_list) then don't load this module
        # print(f'module load {module_list[seed]}')
        os.system(f'module load {module_list[seed]}')
    # else:
    #     print(f"Did not load module in {module_list}")


# to be compiled in main 'damask' folder, not 'src'
while not os.path.isfile('./bin/DAMASK_spectral'):
    os.system('clear')
    os.system(f"module purge")
    module_groups = [
        aue_intel_list,
        aue_intel_oneapi_compilers_list, 
        aue_intel_oneapi_mkl_list,
        aue_intel_oneapi_mpi_list,
        aue_openmpi_list,
        openmpi_gnu_list,
        gnu_list,
    ]
    random.shuffle(module_groups) # shuffle order of modules
    for module_list in module_groups:
        load_random_module(module_list) # shuffle module version
    os.system('module list ')
    os.system(f"make clean")
    os.system(f"make")
    os.system('sleep 2')
