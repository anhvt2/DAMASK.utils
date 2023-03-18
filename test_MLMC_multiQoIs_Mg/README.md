
# Documentation

copy 
`DAMASK.utils/test_stochastic-collocation_runs_s1057681/phenomenological-slipping+twinning-Mg/test-default` 
to
`test-phenomenological-slipping+twinning-Mg/`

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

1. `16x16x16`: Mg -- 300 * 1e-3: 29 mins 20 secs
```
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  100.0    logincs 10    freq 1
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  100.0    incs 20    freq 1
```
1. `8x8x8`: 17 minutes
```
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  100.0    incs 10    freq 1
fdot    1.0e-3 0 0    0 * 0    0 0 *    stress  * * *   * 0 *   * * 0 time  200.0    logincs 10    freq 1
```
