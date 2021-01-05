# DAMASK.utils
DAMAKS utilities scripts

```shell
postResults single_phase_equiaxed_tension.spectralOut --cr f,p
filterTable < single_phase_equiaxed_tension.txt --white inc,1_f,1_p > stress_strain.log
python3 plotStressStrain.py --file "stress_strain.log"
```
