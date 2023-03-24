
# Documentation

copy 
`DAMASK.utils/test_stochastic-collocation_runs_s1057681/phenomenological-slipping+twinning-Mg/test-default` 
to
`test-phenomenological-slipping+twinning-Mg/`

### How to set up

The file `utils.jl` implements the interface between the MLMC with multiple QoIs and the `wrapper-DREAM3D-DAMASK.py`. At the end of the evaluation, print the results on the screen according to the format. 

1. specify settings in `run_multilevel_multiple_qoi.jl`
1. build an interface in `utils.jl`
1. build a wrapper in `wrapperMLMC-multiQoIs.py` that returns conformal responses according to `utils.jl`
1. run `run_multilevel_multiple_qoi.jl`

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
1. `16x16x16`: Mg -- 300 * 1e-3: 29 mins 20 seconds
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
