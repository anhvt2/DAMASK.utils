
# Springback - from SPK to DAMASK

The idea of this project to model microstructure of sprinback through SPPARKS, and run the CPFEM through DAMASK. This repository is mainly adopted from its sibling, `test_tensile_dogbone-spk2damask/`.

# SPPARKS

Geometry of a microspring

[![A microspring.](./microspring-1 "A microspring")](https://doi.org/10.1016/j.matdes.2020.109198)

[![A microspring.](./microspring-2 "A microspring")](https://doi.org/10.1016/j.matdes.2020.109198)

# DAMASK

1. `geom` file is obtained from `geom_spk2dmsk.py`, might require `geom_cad2phase.py` run.
2. combine `material.config` and `material.config.preamble` for material and void
```shell
cat material.config.preamble  | cat - material.config | sponge material.config
```
3. Void config is adopted from `Phase_Isotropic_FreeSurface.config` based on a conversation with Philip Eisenlohr.
```
[Void]

## Isotropic Material model to simulate free surfaces ##
## For more information see paper Maiti+Eisenlohr2018, Scripta Materialia, 
## "Fourier-based spectral method solution to finite strain crystal plasticity with free surfaces"

elasticity              hooke
plasticity              isotropic

/dilatation/

(output)                flowstress
(output)                strainrate

lattice_structure       isotropic
c11                     0.24e9
c12                     0.0
c44                     0.12e9
taylorfactor            3
tau0                    0.3e6
gdot0                   0.001
n                       5
h0                      1e6
tausat                  0.6e6
w0                      2.25
atol_resistance         1
```

