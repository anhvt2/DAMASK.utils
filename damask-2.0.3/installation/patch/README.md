# DAMASK patching

This folder contains patches that modify the functionality of the current development version of DAMASK ahead of the corresponding adoption in the official release.

## Usage

```bash
cd DAMASK_ROOT
patch -p1 < installation/patch/nameOfPatch
```

## Available patches

  * **disable_HDF5** disables all HDF5 output.
    HDF5 output is an experimental feature. Also, some routines not present in HDF5 1.8.x are remove to allow compilation of DAMASK with HDF5 < 1.10.x

## Create patch
commit your changes

```bash
git format-patch PATH_TO_COMPARE --stdout >
```
