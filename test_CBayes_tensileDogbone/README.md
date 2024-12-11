
# Purposes

The objective of this case study is to infer/learn a distribution for void (e.g. volume fraction), given a distribution in observables, using **data-consistent stochastic inverse** approach

# Workflow

Step-1: Run SPPARKS with *no periodic boundary condition*
Step-2: Convert SPPARKS geometry to DAMASK-readable, helped by `orientations.dat` generated from DREAM.3D
Step-3: Pad air and seed voids into the newly generated microstructure geometry
Step-4: Run and post-process DAMASK
Step-5: Build a physics-constrained ML model (monotonic GPR)
Step-6: Create mock-up test cases (see `makeTestExamples.sh`)
Step-6: Perform data-consistent inversion

# Roadmap

* 8x8x8: 512 voxels -- very fast
* 10x10x10: 1000 voxels -- reasonable
* 12x12x12: 1728 voxels -- ok

# Sizes

* (120, 20, 200): non-padded air
* (120, 24, 200): padded-air

# Number of voxels

### Pre-inserting void

Shape: (120, 20, 200)

* air (pre-padded): 239740 voxels out of (120 x 20 x 200) = 0.49945833333333334
* solid voxels: 240260 voxels

### After-inserting voids

Index:
* air: grain id = 1
* void: 
	* every void = 1 pixel
	* 2 to `numVoidVoxels+1`
* everything else: solid

Shape: (120, 20, 200)

* 1% = 2402.6 voxels (approx. 2403)

```bash
python3 seedVoid.py \
    --origGeomFileName spk_dump_12_out.geom \
    --percentage 1.5 \
    --phaseFileName phase_dump_12_out.npy
```

### Pad air

(120, 24, 200): depending on the number of padded air

```bash
python3 padAirPolycrystals.py \
    --origGeomFileName spk_dump_12_out.geom
```
# Mesh convergence study

Based on a discussion with Philip Eisenlohr on Nov 30, 2023.

mesh convergence study for void spatial discretization
coarse/fine: 1x1x1 and 2x2x2 and 4x4x4

4x4x4: instead of cube, try to model sphere with voxels
is the answer the same? yes: converged; no: not converged

