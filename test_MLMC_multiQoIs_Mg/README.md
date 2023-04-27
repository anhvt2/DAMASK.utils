
# Folder structures

* `template`: vanilla MLMC + CPFEM -- run alternatively, hand-in-hand, MLMC and CPFEM
* `fake`: build a generated dataset from CPFEM and run MLMC on a fake wrapper, which simply looks up values from the dataset
	* `fakerun_multilevel_multiple_qoi.jl` and `fakewrapper_multilevel_multiple_qoi.py`: 5 levels
	* `fakerun_2levels_multiple_qoi.jl` and `fakewrapper_2levels_multiple_qoi.py`: 2 levels
* `hybrid`: combines both `fake` and `template` -- lookup first, if failed, then run CPFEM


# Documentation

copy 
`DAMASK.utils/test_stochastic-collocation_runs_s1057681/phenomenological-slipping+twinning-Mg/test-default` 
to
`test-phenomenological-slipping+twinning-Mg/`

### How to install Julia packages

(also available at `./MultilevelEstimators.jl/README.md`, but this version is more comprehensive)

Mostly comes from private communication with Pieterjan Robbe

```julia
julia> ENV["JULIA_SSL_NO_VERIFY_HOSTS"] = "**" # useful behind Sandia firewall -- no certificate for downloading
pkg> dev https://github.com/PieterjanRobbe/MultilevelEstimators.jl
pkg> dev https://github.com/PieterjanRobbe/Reporter.jl
pkg> add Random
pkg> add Statistics

julia> import Pkg; Pkg.resolve()
julia> Pkg.add("PrettyTables")
julia> Pkg.add("FileIO")
julia> Pkg.add("ProgressMeter")

include("Example.jl")
check_variances()
run_multilevel_checkpoint()
run_multilevel()
```

### How to set up

The file `utils.jl` implements the interface between the MLMC with multiple QoIs and the `wrapper-DREAM3D-DAMASK.py`. At the end of the evaluation, print the results on the screen according to the format. 

1. specify settings in `run_multilevel_multiple_qoi.jl`
1. build an interface in `utils.jl`
1. build a wrapper in `wrapperMLMC-multiQoIs.py` that returns conformal responses according to `utils.jl`
1. run `run_multilevel_multiple_qoi.jl`


### How to run

1. With **real** wrapper `wrapper_multilevel_multiple_qoi.py`

Idea: couple DAMASK as sampler 

```shell
rm -rfv test/
cp -rfv template/ test/
cd test/
ln -sf ../*jl .
rm -f nohup.out; nohup julia run_multilevel_multiple_qoi.jl &
```

2. With **fake** wrapper `fakewrapper_multilevel_multiple_qoi.py`
including:
* `fakeutils.jl`
* `fakerun_multilevel_multiple_qoi.jl`
* `fakewrapper_multilevel_multiple_qoi.py`

Idea: use lookup dataset generated from DAMASK as sampler to save computational cost

```shell
cd fakeWrappers/
python3 cleanseDataset.py # convert log to clean data file -- initial
julia fakerun_multilevel_multiple_qoi.jl
# rm -f nohup.out; nohup julia fakerun_multilevel_multiple_qoi.jl & # run Julia again with fake wrapper
```

3. With **hybrid** wrapper `hybridwrapper_multilevel_multiple_qoi.py`
including
* `hybridutils.jl`
* `hybridrun_multilevel_multiple_qoi.jl`
* `hybridwrapper_multilevel_multiple_qoi.py`

Idea: 	
	(1) first use lookup dataset generated from DAMASK as sampler to save computational cost
	(2) if information not available then run DAMASK

```shell
cd fakeWrappers/
python3 cleanseDataset.py # convert log to clean data file -- initial
# julia hybridrun_multilevel_multiple_qoi.jl
rm -f nohup.out; nohup julia hybridrun_multilevel_multiple_qoi.jl & # run Julia again with fake wrapper
```

### How to post-process

```julia
julia> using MultilevelEstimators, JLD2, Reporter

julia> history = load("DREAM3D-multilevel.jld2", "history")
MultilevelEstimators.jl history file

julia> report(history, include_preamble=true, png=true)
```

```shell
julia history2costfrac.jl
python3 plot_fractional_cost.py
```

### DREAM.3D

1. debug `generateMsDream3d.sh`
1. distribute `material.config.preamble` to each folder
1. construct a valid `material.config`

adopt `material.config` from `DAMASK.utils/test_stochastic-collocation_runs_s1057681/phenomenological-slipping+twinning-Mg/material.config`

### DAMASK

1. check pre-processing
1. check `run_damask.sh`
1. check post-processing

### Computational cost
1. `32x32x32`: 3 hours 27 minutes 32 seconds = 12452 seconds
```
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  100.0    logincs 10    freq 1
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  100.0    incs 20    freq 1
```
1. `16x16x16`: 29 mins 20 seconds
```
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  100.0    logincs 10    freq 1
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  100.0    incs 20    freq 1
```
1. `16x16x16`: 54 minutes 30 seconds = 3270 seconds
```
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  100.0    logincs 10    freq 1
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  100.0    incs 20    freq 1
```
1. `8x8x8`: 17 minutes = 1020 seconds
```
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  100.0    incs 10    freq 1
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  200.0    logincs 10    freq 1
```
1. `8x8x8`: 32 minutes = 1920 seconds
```
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  100.0    logincs 10    freq 1
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  100.0    incs 20    freq 1
```
1. `4x4x4`: 5 minutes 30 seconds or 330 seconds
```
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  100.0    logincs 10    freq 1
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  100.0    incs 20    freq 1
```
1. `2x2x2`: 4 seconds
```
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  100.0    logincs 10    freq 1
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  100.0    incs 20    freq 1
```
