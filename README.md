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


