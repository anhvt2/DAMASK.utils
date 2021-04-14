
# DREAM.3D files

These DREAM.3D files are adopted from various places: 

* DREAM.3D examples
* DREAM.3D tutorials
* PRISMS-Fatigue/-Plasticity packages

but the final forms very much depend on the materials, the experimentalists, where the information is not always available (due to various reasons). 

# How to change from `ideapad` version to `Solo` version

Solo path: `/ascldap/users/anhtran/scratch/DAMASK/DAMASK-2.0.2/examples/SpectralMethod/Polycrystal/testMLMC_3Mar21/mu-3.00-sigma-0.50-sve-1/18x18x18`

Ideapad: `/home/anhvt89/Documents/DAMASK/DAMASK.utils/DREAM.3D/18x18x18`

So the only things that need changing are the path.

# Visualization

Using `geom_check` from DAMASK and view the output `.vtr` in ParaView. Unfortunately, DAMASK v2.0.1 uses `python2.7` for post-processing purposes, so one would have to install `vtk` packages for `python2.7`

```shell
geom_check *.geom
```

# Lessons learned

NonExact version works better than the exact version, because there are no guarantee that the DREAM.3D downsizes exactly as float. For example, 0.5 * 64 is not 32 (probably because of the internal packing algorithm for microstructure reconstruction). 


