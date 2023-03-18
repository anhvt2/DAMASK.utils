
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
