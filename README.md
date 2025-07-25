# PETSc

Compiling PETsc can be a challenging task. Version control, dependencies, for examples, are a few challenges.

To download external packages for *any* PETSc packages, go to [https://ftp.mcs.anl.gov/pub/petsc/externalpackages/](https://ftp.mcs.anl.gov/pub/petsc/externalpackages/). Many Sandia HPCs sit behind a great firewall and will not be able to download a simple package (even from a `.gov` domain), so it may be wise to supply all external packages internally. 

### PETSc-3.9.4

##### Ghost HPC
```shell

# Ghost - 25Sep24 (WORKED - in production)
module purge
module load gnu/13.1.1
# Currently Loaded Modules:
#   1) gnu/13.1.1
rm -rfv petsc-3.9.4/
tar -xvzf petsc-3.9.4.tar.gz
cd petsc-3.9.4


./configure \
    --prefix=$HOME/local/petsc-3.9.4 \
    --download-openmpi=$HOME/data/petsc/externalpackages-3.9.4/openmpi/openmpi-3.0.1.tar.gz \
    --download-cmake=$HOME/data/petsc/externalpackages-3.9.4/cmake/cmake-3.9.6.tar.gz \
    --download-fblaslapack=$HOME/data/petsc/externalpackages-3.9.4/fblaslapack/fblaslapack-3.4.2.tar.gz \
    --download-fftw=$HOME/data/petsc/externalpackages-3.9.4/fftw/fftw-3.3.3.tar.gz \
    --download-hdf5=$HOME/data/petsc/externalpackages-3.9.4/hdf5/hdf5-1.8.18.tar.gz \
    --download-chaco=$HOME/data/petsc/externalpackages-3.9.4/Chaco-2.2-p2.tar.gz \
    --download-hypre=$HOME/data/petsc/externalpackages-3.9.4/git.hypre \
    --download-ml=$HOME/data/petsc/externalpackages-3.9.4/git.ml \
    --download-metis=$HOME/data/petsc/externalpackages-3.9.4/git.metis \
    --download-netcdf=$HOME/data/petsc/externalpackages-3.9.4/netcdf-4.5.0 \
    --download-parmetis=$HOME/data/petsc/externalpackages-3.9.4/git.parmetis \
    --download-superlu=$HOME/data/petsc/externalpackages-3.9.4/git.superlu \
    --download-superlu_dist=$HOME/data/petsc/externalpackages-3.9.4/git.superlu_dist \
    --download-triangle=$HOME/data/petsc/externalpackages-3.9.4/Triangle/Triangle.tar.gz \
    --download-zlib=$HOME/data/petsc/externalpackages-3.9.4/zlib/zlib-1.2.11.tar.gz \
    --with-x=0 \
    --with-cxx-dialect=C++11 \
    --with-c2html=0 \
    --with-fc=gfortran \
    --with-cc=gcc \
    --with-f77=gfortran \
    --with-f90=gfortran \
    --with-debugging=0 \
    --with-ssl=0 \
    --FCFLAGS='-w -fallow-argument-mismatch -O2' \
    --FFLAGS='-w -fallow-argument-mismatch -O2' \
    PETSC_ARCH="arch-linux2-c-opt" PETSC_DIR=`pwd`

make PETSC_DIR=/ascldap/users/anhtran/data/petsc/petsc-3.9.4 PETSC_ARCH=arch-linux2-c-opt all
make PETSC_DIR=/ascldap/users/anhtran/data/petsc/petsc-3.9.4 PETSC_ARCH=arch-linux2-c-opt install
make PETSC_DIR=/ascldap/users/anhtran/local/petsc-3.9.4 PETSC_ARCH="" test
```

##### Skybridge HPC

```shell
# Skybridge - 14Jun23 (WORKED - in production - see modules load for running)
module purge
module load openmpi-gnu/2.1
module load gnu/8.2.1 
# Currently Loaded Modules:
#   1) openmpi-gnu/2.1   2) gnu/8.2.1
rm -rfv petsc-3.9.4/
tar -xvzf petsc-3.9.4.tar.gz
cd petsc-3.9.4

./configure \
    --prefix=$HOME/local/petsc-3.9.4 \
    --download-openmpi \
    --download-cmake \
    --download-fblaslapack \
    --download-fftw \
    --download-hdf5 \
    --download-chaco \
    --download-hypre=$HOME/data/petsc/externalpackages-3.9.4/git.hypre \
    --download-ml=$HOME/data/petsc/externalpackages-3.9.4/git.ml \
    --download-metis=$HOME/data/petsc/externalpackages-3.9.4/git.metis \
    --download-netcdf=$HOME/data/petsc/externalpackages-3.9.4/netcdf-4.5.0 \
    --download-parmetis=$HOME/data/petsc/externalpackages-3.9.4/git.parmetis \
    --download-superlu=$HOME/data/petsc/externalpackages-3.9.4/git.superlu \
    --download-superlu_dist=$HOME/data/petsc/externalpackages-3.9.4/git.superlu_dist \
    --download-triangle \
    --download-zlib \
    --with-x=0 \
    --with-cxx-dialect=C++11 \
    --with-c2html=0 \
    --with-debugging=0 \
    --with-ssl=0 \
    --with-fc=mpif90 \
    --with-cc=mpicc \
    --with-cxx=mpicxx \
    --with-f77=mpif77 \
    --with-f90=mpif90 \
    PETSC_ARCH="arch-linux2-c-opt" PETSC_DIR=`pwd`


    # COPTFLAGS="-frecursive" \
    # COPTFLAGS="-O3 -xHost -no-prec-div -frecursive" \
    # CXXOPTFLAGS="-O3 -xHost -no-prec-div" \
    # FOPTFLAGS="-O3 -xHost -no-prec-div" \
    # PETSC_ARCH="arch-linux2-c-opt" PETSC_DIR=`pwd`

make PETSC_DIR=/ascldap/users/anhtran/data/petsc/petsc-3.9.4 PETSC_ARCH=arch-linux2-c-opt all
make PETSC_DIR=/ascldap/users/anhtran/data/petsc/petsc-3.9.4 PETSC_ARCH=arch-linux2-c-opt install

# when running -- use these modules instead
module load gnu/10.2.1
module load openmpi-gnu/4.1
module load tce
module load python/3.6.0

```

# DAMASK.utils

DAMAKS utilities scripts

```shell
postResults single_phase_equiaxed_tension.spectralOut --cr f,p
filterTable < single_phase_equiaxed_tension.txt --white inc,1_f,1_p > stress_strain.log
python3 plotStressStrain.py --file "stress_strain.log"
```

Updates from `DAMASK-3.0.0`

* Density plot with pandas: https://damask3.mpie.de/documentation/tutorials/python/density-plot-with-pandas/
* Plot stress-strain curve of selected grains with scatter: https://damask3.mpie.de/documentation/tutorials/python/plot-stress-strain-curve-of-selected-grains-with-scatter/
* Create histogram: https://damask3.mpie.de/documentation/tutorials/python/create-histogram/
* Create heatmap: https://damask3.mpie.de/documentation/tutorials/python/create-heatmap/
* Plot a stress-strain curve with yield point: https://damask3.mpie.de/documentation/tutorials/python/plot-a-stress-strain-curve-with-yield-point/

# DAMASK

## DAMASK-2.0.2

## DAMASK-2.0.3

* Change of keywords is not well documented but has been reported [https://www.researchgate.net/post/Did-anyone-used-DAMASK-code-for-crystal-plasticity-simulations-of-HCP-materials](https://www.researchgate.net/post/Did-anyone-used-DAMASK-code-for-crystal-plasticity-simulations-of-HCP-materials). 
* `covera_ratio` -> `c/a`
* require `mech            isostrain`
* require `nconstituents   1`
* disable a few fields such as
    * `# (output) e              # total strain as Green-Lagrange tensor`
    * `# (output) ee             # elastic strain as Green-Lagrange tensor`



# DREAM.3D

Following the advices of [Mohammadreza Yaghoobi](https://scholar.google.com/citations?user=EOO01WsAAAAJ&hl=en&oi=sra) in TMS2021, the following is adopted from PRISMS-Fatigue and PRISMS-Plasticity for Euler angles:

* Rolled texture: Primary: `Fatigue/src/Al7075_rolled_texture_elongated_grains.dream3d`:
	* `ODF`:
		* `Euler 1`, `Euler 2`, `Euler 3`, `Weight`, `Sigma (ODF)`
		* 145, 45, 5, 7500, 5
		* 35, 45, 5, 7500, 5
		* 35, -45, 5, 7500, 5
		* 145, -45, 5, 7500, 5
		* -52.5, 45, 0, 5000, 5
		* 52.5, 45, 0, 5000, 5
		* 52.5, -45, 0, 5000, 5
		* -52.5, -45, 0, 5000, 5
	* `Axis ODF`:
		* `Euler 1`, `Euler 2`, `Euler 3`, `Weight`, `Sigma (ODF)`
		* 0, 0, 0, 5000, 7
* Cubic texture: Primary (not Precipitate): `Fatigue/src/Al7075_cubic_texture_equiaxed_grains.dream3d`:
	* `Euler 1`, `Euler 2`, `Euler 3`, `Weight`, `Sigma (ODF)`
	* 0, 0, 0, 25, 5

# MultilevelEstimators.jl

## How to install

(also available at `./MultilevelEstimators.jl/README.md`, but this version is more comprehensive)

Mostly comes from private communication with Pieterjan Robbe

```julia
julia> ENV["JULIA_SSL_NO_VERIFY_HOSTS"] = "**" # useful behind Sandia firewall -- no certificate for downloading
pkg> dev https://github.com/PieterjanRobbe/MultilevelEstimators.jl
pkg> add Random
pkg> add Statistics

julia> import Pkg; Pkg.resolve()
julia> Pkg.add("PrettyTables")
julia> Pkg.add("ProgressMeter")

include("Example.jl")
check_variances()
run_multilevel_checkpoint()
run_multilevel()
```

Sometimes will need to switch the branch to `develop` in `~/.julia/dev/MultilevelEstimators/` and
```shell
cd ~/.julia/dev/MultilevelEstimators
git checkout develop
git pull
```

## How to setup

The following list of files needed to be checked.

1. `wrapperMIMC-DREAM3D-DAMASK.py`:
* this is a driver file for calling DREAM.3D (which generates microstructure at different resolutions)
* run DAMASK (adaptively with number of processors)
* call post-processing python script
* print the **correct output** (whether it is Young modulus or yield stress) to screen

2. `check_variance.jl`: pick up from the output
```julia
function get_qoi(out, index)
    for line in split(out, "\n")
        if occursin("Estimated Young modulus at $(index)", line)
            return parse(Float64, split(line)[end - 1])
        end
    end
    return NaN # return NaN if no value could be found
end
```
3. `run_check_variances.jl`:
* set up the overall computational budget
* weighted index set
* number of fidelities in each direction

## How to run

From the email communication around April 22, 2021

Suppose that your wrapper can be called with a script, called "my script.sh". This script would take as input the level parameter \ell. The script would then call the wrapper to compute a solution at mesh level m and mesh level m-1, extract a quantity of interest for each mesh level, and print these two numbers to screen. Let's also assume that for level \ell = 0, the script outputs the same number twice (there is no m-1 in this case). (To clarify: the "level \ell" here would be integers 0, 1, 2 … used in the MLMC method, and the "mesh level m" would correspond to the number of grid points, i.e., (m = 0) => 8x8x8, (m = 1) => 16x16x16, (m = 2) => 32x32x32, …). For example:


terminal:
```shell
$ ./myscript.sh --level 2
<QoI at mesh level 2> <QoI at mesh level 1>
$ ./myscript.sh --level 1
<QoI at mesh level 1> <QoI at mesh level 0>
$ ./myscript.sh --level 0
<QoI at mesh level 0> <QoI at mesh level 0, same number>
```

As a first step, we will look if there is variance decay. Therefore, you would need to do the following in Julia:

1) Write a "sample" function that calls your wrapper. This function will look like this:

julia function:
```julia
function my_sample_function(level, x)
cmd = `./myscript.sh --level $(level)` # command to be executed in the terminal
out = read(cmd, String) # execute command and read output
Qf, Qc = parse.(Float64, split(out)) # split output into Qf and Qc (approximations at mesh level m and mesh level m-1)
return level == 0 ? (Qf, Qf) : (Qf-Qc, Qf) # return multilevel difference and approximation at mesh level m
end
```


2) Use the (simplified) function "check_variances" to check for variance decay. I had included this function in Example.jl a couple of emails back, but here it is again:


julia function:
```julia
function check_variances(; max_level=3, budget=60)
    budget_per_level = budget/(max_level + 1)

    for level in 0:max_level
        samps_dQ = []
        samps_Qf = []
        timer = 0
        while timer < budget_per_level
            timer += @elapsed dQ, Qf = my_sample_function(level, 0) # last argument is 0 because it is not used
            push!(samps_dQ, dQ) 
            push!(samps_Qf, Qf) 
        end 
        println("Level ", level, ", V = ", var(samps_Qf), ", dV = ",
                var(samps_dQ), " (", length(samps_dQ), " samples)")
    end 
end
```


3) Write all of this in a file, and don't forget to include Statistics.jl:

file `my_julia_script.jl`:
```julia
using Statistics

function my_sample_function(level, x)
cmd = `./myscript.sh --level $(level)` # command to be executed in the terminal
out = read(cmd, String) # execute command and read output
Qf, Qc = parse.(Float64, split(out)) # split output into Qf and Qc (approximations at mesh level m and mesh level m-1)
return level == 0 ? (Qf, Qf) : (Qf-Qc, Qf) # return multilevel difference and approximation at mesh level m
end

function check_variances(; max_level=3, budget=60)
    budget_per_level = budget/(max_level + 1)

    for level in 0:max_level
        samps_dQ = []
        samps_Qf = []
        timer = 0
        while timer < budget_per_level
            timer += @elapsed dQ, Qf = my_sample_function(level, [])
            push!(samps_dQ, dQ) 
            push!(samps_Qf, Qf) 
        end 
        println("Level ", level, ", V = ", var(samps_Qf), ", dV = ",
                var(samps_dQ), " (", length(samps_dQ), " samples)")
    end 
end

# parameters:
# max_level = number of levels -1 (max_level = 2 will use level 0, 1 and 2)
# budget = total computational budget in seconds (8*3600 = 8 hours) - function will finish in this amount of time
check_variances(max_level=2, budget=8*3600)
```

4) Call this script from the terminal

terminal:
```shell
$ julia my_julia_script.jl
...
```

5) Wait for the result (will be printed to the screen)

## How to plot results

See the `Reporter` package from Pieterjan Robbe: [https://github.com/PieterjanRobbe/Reporter.jl](https://github.com/PieterjanRobbe/Reporter.jl)

## Discussion

See more at [https://groups.google.com/g/dream3d-users/c/SPCdr-BWROs?pli=1](https://groups.google.com/g/dream3d-users/c/SPCdr-BWROs?pli=1)

# MATLAB/Octave platforms

* `damask-2.0.2` runs on Solo, along with Octave. If optimization is concerned, try using Octave
* `damask-2.0.3` may be compiled on Skybridge, along with MATLAB and Octave. If successful, can run both. However, Skybridge may limit in 48 hours or SNL may disconnect.

# Major Branches

1. consistent Bayesian
    * `testCBayes-Damask-Phase_Dislotwin_TWIP-Steel-FeMnC-64x64x64/`: TWIP grain size inferrence
    * `testCBayes-Damask-Phase_Phenopowerlaw_Aluminum/`
    * `test_Dbone-spk2damask`: learning void distribution in dogbone specimen
2. stochastic collocation for yield
    * `test_stochastic-collocation_runs_s1057681/`: stochastic collocation for constitutive model UQ
    * `test_stochastic-collocation_multiRVE_Solo/`: under construction
3. model calibration using Bayesian optimization
    * `test_model-calibration_Solo/`: asynchronous parallel Bayesian optimization for model calibration
4. multi-{level,index} Monte Carlo
    * `test_MLMC_runs_s1057681_alphaTitanium/`: MLMC for titanium
    * `test_MIMC_runs_s1057681_Aluminum/`: MIMC for aluminum
    * `test_MLMC_multiQoIs_Mg`: multi-output MLMC for stress/strain
5. graph CPFEM:
    * `test_rve_graph`: discrete combinatorics / graph theory for RVE