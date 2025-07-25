
# POD ROM for oligocrystal with different strain rates/temperature

This repository contains a test case for building projection-based reduced-order model (POD) for the **same** microstructure going under different strain rates and temperatures. 

To build the FOM, we fix the microstructure and vary other testing conditions. The microstructure is generated from SPPARKS. 

#### SPPARKS

The fixed microstructure is generated using 3D AM, where the SPK script is `DAMASK.utils/test_tensile_dogbone-spk2damask/spk/res-50um-additive/in.potts_additive_dogbone`. 

```
# SPPARKS potts/additive test file

seed         56789

variable     res    equal   50    # resolution: 1 pixel = ${res} um
variable     L      equal   10000 # dogbone specimen length (um)
variable     W      equal   6000  # dogbone specimen width (um)
variable     T      equal   1000  # dogbone specimen thickness (um)

variable     Nx     equal ${W}/${res}
variable     Ny     equal ${T}/${res}
variable     Nz     equal ${L}/${res}
variable     Nspin  equal ${Nx}*${Ny}*${Nz}/10
variable     NxNyNz equal ${Nx}*${Ny}*${Nz}

variable     imgRes equal 5
variable     NxRes  equal ${Nx}*${imgRes}
variable     NyRes  equal ${Ny}*${imgRes}
variable     NzRes  equal ${Nz}*${imgRes}

#———————————————————————————————————————————

#Define melt pool parameters
#——————————————————————————————————————————— 
         
# app_style     potts/additive     1000 30 70 30 7     50 90 45 12 0.1 # default
# app_style     potts/additive     1000 10 23 10 4     16 30 15  5 0.1 # user-defined: kind of working
#
variable    spotWidth       equal   8
variable    meltTailLength  equal   20
variable    meltDepth       equal   8
variable    capHeight       equal   4

variable    haz             equal   16
variable    tailHaz         equal   25
variable    depthHaz        equal   12
variable    capHaz          equal   5

variable    thickness       equal   ${meltDepth}
variable    numLayers       equal   floor(${Nz}/${meltDepth})+1

variable    speed           equal   10
variable    hatchSpacing    equal   10

#———————————————————————————————————————————

#Setup melt pool and scanning patterns
#——————————————————————————————————————————— 

app_style   potts/additive     1000 ${spotWidth} ${meltTailLength} ${meltDepth} ${capHeight}     ${haz} ${tailHaz} ${depthHaz}  ${capHaz} 0.1 # user-defined

#  |————————————————————————————————————————
#  | nspins     = atoi(arg[1])
#  |————————————————————————————————————————
#  | nspins = atoi(arg[1]); # Number of spins
#  | spot_width = atoi(arg[2]); # Width of the melt pool
#  | melt_tail_length = atoi(arg[3]); # Length of tail from meltpool midpoint
#  | melt_depth = atoi(arg[4]); # How many lattice sites deep the melt pool is
#  | cap_height = atoi(arg[5]); # Height of the cap leading the meltpool
#  |————————————————————————————————————————
#  | HAZ = atoi(arg[6]); #Size of the HAZ surrounding the melt pool (must be larger than spot_width and melt_depth)
#  | tail_HAZ = atoi(arg[7]); #Length of hot zone behind meltpool (must be larger than melt_tail_length)
#  | depth_HAZ = atof(arg[8]); //Depth of the hot zone underneath the meltpool (must be larger than melt_depth)
#  | cap_HAZ = atoi(arg[8]); #Size of HAZ infront of the melt pool (must be larger than cap_height)
#  | exp_factor = atof(arg[9]); #Exponential parameter for mobility decay in haz M(d) = exp(-exp_factor * d)
#  |————————————————————————————————————————


#Define simulation domain and initialize site variables
#———————————————————————————————————————————
dimension    3
lattice      sc/26n 1.0
# region         box block 0 100 0 100 0 100
region       box block 0 ${Nx} 0 ${Ny} 0 ${Nz}

boundary     n n n # non-periodic boundary condition: https://spparks.github.io/doc/boundary.html

create_box      box
create_sites    box
set             i1 range 1 1000 
set             d1 value 0.0
#———————————————————————————————————————————


# Define an additive scan pattern on rectangular domain 
# using am pass and cartesian layer commands.   
#———————————————————————————————————————————

am pass 1 dir X speed ${speed} hatch ${hatchSpacing}
# am pass 2 dir Y speed 10 hatch 10



am cartesian_layer 1 start LL pass_id 1 thickness ${thickness} offset -${Nx} 0.0
# am cartesian_layer 2 start UL pass_id 2 thickness 5 # offset 0.0 400.0
# am cartesian_layer 3 start UR pass_id 1 thickness 5 # offset 400.0 0.0
# am cartesian_layer 4 start LR pass_id 2 thickness 5 # offset 0.0 -400.0

am build start ${thickness} num_layers ${numLayers}

# example:
# am pass 1 dir X speed 10 hatch 25
# am pass 2 dir Y speed 10 hatch 25

# am cartesian_layer 1 start LL pass_id 1 thickness 25 offset -80.0 0.0
# am cartesian_layer 2 start UL pass_id 2 thickness 25 offset 0.0 80.0
# am cartesian_layer 3 start UR pass_id 1 thickness 25 offset 80.0 0.0
# am cartesian_layer 4 start LR pass_id 2 thickness 25 offset 0.0 -80.0

# am build start 25.0 num_layers 4
#———————————————————————————————————————————

#Setup the solver type and parameters. Must use a "sweep" style solver
#——————————————————————————————————————————— 
sector          yes
sweep           random mask no
temperature     0.25
#———————————————————————————————————————————

#Specify output commands and styles.
#——————————————————————————————————————————— 
diag_style      energy
stats           1.0


# dump 1 stitch 2 additiveDogbone.23Jul2020.linux site # RE-ENABLE for production run
dump 2 text 1.0 dump.additive_dogbone.*    id i1 d1 x y z # RE-ENABLE for production run
dump vtkFile vtk 1.0 additive_dogbone.*.vti i1 # dump vtkFile vtk 1.0 additive_dogbone.*.vti i1
dump_modify vtkFile vtk ${Nx} ${Ny} ${Nz} 0 ${NxNyNz} sort id # dump_modify vtkFile vtk 100 100 100 0 10000 sort id

#If SPPARKS was not compiled with libjpeg, comment out the lines below.
# https://spparks.github.io/doc/dump_image.html
# top
dump top image 2 top.additive_dogbone.*.jpg site site crange 1 1000 drange 1 1 view 0.0 0.0 shape cube box no 1 zoom 2 size ${NyRes} ${NxRes} sdiam 1.05

# tranverse
dump transverse image 2 transverse.additive_dogbone.*.jpg site site crange 1 1000 drange 1 1 view 90.0 -90.0 shape cube box no 1 zoom 1.0 size ${NxRes} ${NzRes}

# longitudinal
dump longitudinal image 2 longitudinal.additive_dogbone.*.jpg site site crange 1 1000  drange 1 1 view 90.0 0.0 shape cube box no 1 zoom 1.0 size ${NyRes} ${NzRes}

#——————————————————————————————————————————— 

run             50
# run              5000000
```

After that, we seed the void sampled from a void dictionary, which is built from DREAM.3D and obeys a log normal distribution in void sizes. After seeding voids, we **fix** the microstructure to move forward. 

#### Deprecated SPPARKS

This SPK script runs the normal GG in 3D and generate a microstructure ensemble. However, as we wish to fix the microstructure, this script is deprecated and used for another test module. 

```shell
# Run SPPARKS
cd spk/
mpirun -np 8 spk < in.potts_3d
cd ..

# Run DREAM.3D for generating a dictionary of void morphology
cd dream3d-void-libs/

cd ../

# Run seedVoid.py
cd seedVoid/
python3 seedVoid.py \
    --origGeomFileName spk_dump_12_out.geom \
    --percentage 1 \
    --voidDictionary voidEquiaxed.geom \
    --phaseFileName phase_dump_12_out.npy
cd ..
```

#### SPPARKS

To reproduce the microstructure, run the input script on `$roglocal` computer with exactly the same seed. 

~~Run `in.potts_3d` with hybrid phase-field/kMC with sufficiently long trajectory.~~

Run `in.potts_additive_dogbone` and visualize the whole AM process. Note: use an appropriate `.mp4` viewer (or risk having your computer shutdown for choking memory).

#### DREAM.3D

Sample orientation, and voids morphology. See `dream3d-void-libs/voidDAMASK.json`. This is a DREAM3D file and can be opened by DREAM3D.

#### seedVoid.py

Sample void throughout the computational domain where voids are sampled from a dictionary with log normal distribution. See `seedVoid/seedVoid.py`. 

#### DAMASK

Make sure to use dislocation-density-based constitutive model for high-/low-strain rate with different temperature. 

For loading conditions, see [https://damask2.mpie.de/bin/view/Documentation/LoadDefinition.html](https://damask2.mpie.de/bin/view/Documentation/LoadDefinition.html)

# ROM

ROM is constructed from FOM following these steps.

1. Extract numerical values from FOM: set fields of interest (FoI) - see `export2npy.py`
 ```
 ...
 FieldsOI = ['Mises(Cauchy)','Mises(ln(V))']
 ...
 ```
1. Calculate mean.
1. Center the train/test datasets.
1. Compute POD basis and POD coefficients
1. Train ML
1. Predict POD coefficients
1. Reconstruct ROM
1. Parse ROM to FOM for visualization. See `parseRom.py`

# Error analysis

Measure and visualize in $L_2$ and $L_1$ error. 

# 3D Visualization

```shell
export fileName='main_tension_inc19' # change this fileName
vtk_rectilinearGrid ${fileName}.txt
vtk_addRectilinearGridData \
 --inplace \
 --data '1_fp','2_fp','3_fp','4_fp','5_fp','6_fp','7_fp','8_fp','9_fp','1_f','2_f','3_f','4_f','5_f','6_f','7_f','8_f','9_f','1_p','2_p','3_p','4_p','5_p','6_p','7_p','8_p','9_p','1_grainrotation','2_grainrotation','3_grainrotation','4_grainrotation','texture','1_ln(V)','2_ln(V)','3_ln(V)','4_ln(V)','5_ln(V)','6_ln(V)','7_ln(V)','8_ln(V)','9_ln(V)','1_Cauchy','2_Cauchy','3_Cauchy','4_Cauchy','5_Cauchy','6_Cauchy','7_Cauchy','8_Cauchy','9_Cauchy','Mises(Cauchy)','Mises(ln(V))' \
 --vtk "${fileName}_pos(cell).vtr" \
 ${fileName}.txt

vtk_addRectilinearGridData \
 --data 'fluct(f).pos','avg(f).pos' \
 --vtk "${fileName}_pos(cell).vtr" \
 ${fileName}_nodal.txt

# use PyVista to pick up .vtr and hide air+voids
python3 plotStress3dDeformedGeom.py --vtr "${fileName}_pos(cell)_added.vtr"
python3 plotStrain3dDeformedGeom.py --vtr "${fileName}_pos(cell)_added.vtr"
```
