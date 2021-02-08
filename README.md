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

