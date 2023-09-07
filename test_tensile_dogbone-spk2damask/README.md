
# AM Dogbone - from SPK to DAMASK

The idea of this project to model microstructure of high-throughput dogbone specimen through SPPARKS, and run the CPFEM through DAMASK.

Here is a few versions of the dogbone specimens and their references:

1. The born-qualified version: **17-4PH** materials

* [https://doi.org/10.1016/j.jmatprotec.2016.10.023](https://dx.doi.org/10.1016/j.jmatprotec.2016.10.023): High-throughput stochastic tensile performance of additively manufactured stainless steel
* [Combining Measure Theory and Bayes Rule to Solve a Stochastic Inverse Problem](https://www.osti.gov/servlets/purl/1877851): slides from Tim Wildey
* [https://doi.org/10.1002/adem.201700102](https://doi.org/10.1002/adem.201700102): Extreme-Value Statistics Reveal Rare Failure-Critical Defects in Additive Manufacturing

![Geometry of the dogbone specimen](./17-4PH_dogbone_GrandChallenge.png)

![Geometry of the dogbone specimen](./17-4PH_dogbone_GrandChallenge-2.png)

2. The A-/B- (large-/small-) size version: **AlSi10Mg**

![Geometry of the dogbone specimen](./AlSi10Mg_dogbone.png)

![Geometry of the dogbone specimen](./AlSi10Mg_dogbone_intentional_void.png)

* [https://doi.org/10.1007/s11837-021-04888-4](https://doi.org/10.1007/s11837-021-04888-4): High-Throughput Statistical Interrogation of Mechanical Properties with Build Plate Location and Powder Reuse in AlSi10Mg
* [https://doi.org/10.1016/j.msea.2020.139922](https://doi.org/10.1016/j.msea.2020.139922): Relationship between ductility and the porosity of additively manufactured AlSi10Mg

* [https://doi.org/10.1007/s11340-021-00696-8](https://doi.org/10.1007/s11340-021-00696-8): (*intentionally precipitated voids*) The Interplay of Geometric Defects and Porosity on the Mechanical Behavior of Additively Manufactured Components

Microstructure info are in

* [https://doi.org/10.1557/jmr.2018.405](https://doi.org/10.1557/jmr.2018.405): P. Yang, L.A. Deibler, D.R. Bradley, D.K. Stefan, and J.D. Carroll, J. Mater. Res. 33, 4040-4052. (2018).
* [https://doi.org/10.1557/jmr.2018.82](https://doi.org/10.1557/jmr.2018.82): P. Yang, M.A. Rodriguez, L.A. Deibler, B.H. Jared, J. Griego, A. Kilgo, A. Allen, and D.K. Stefan, J. Mater. Res. 33, 1701–1712. (2018) (*contain microstructure information - Section III.B.2*).

3. **316L** dogbone

![Geometry of the dogbone specimen](./316L_dogbone.png)

![Geometry of the dogbone specimen](./316L_dogbone_2.png)

![Geometry of the dogbone specimen](./316L_dogbone_3.png)

![Geometry of the dogbone specimen](./316L_dogbone_4.png)

* [https://doi.org/10.1016/j.msea.2019.138632](https://doi.org/10.1016/j.msea.2019.138632): Automated high-throughput tensile testing reveals stochastic process parameter sensitivity
* [https://doi.org/10.1016/j.addma.2020.101090](https://doi.org/10.1016/j.addma.2020.101090): Size-dependent stochastic tensile properties in additively manufactured 316L stainless steel
* [https://doi.org/10.1016/j.addma.2022.102943](https://doi.org/10.1016/j.addma.2022.102943): Optimization of stochastic feature properties in laser powder bed fusion



# SPPARKS (+ DREAM.3D)

~~need a parser from SPPARKS to DAMASK `.geom` file~~: `geom_spk2dmsk.py`

Dimension: 10 mm (4 mm middle) x 6 mm x 1 mm

Attempted resolution:
1. 10 um: 60M pixels
    ```
    variable     Nx     equal  600
    variable     Ny     equal  100
    variable     Nz     equal 1000
    ```

2. 20 um: 7.5M pixels
    ```
    variable     Nx     equal  300
    variable     Ny     equal   50
    variable     Nz     equal  500
    ```

3. 50 um: 0.48M pixels
    ```
    variable     Nx     equal  120
    variable     Ny     equal   20
    variable     Nz     equal  200
    ```

Input decks:

1. try with `t = 16.681021` corresponding to `spk/res-50um/dump.12.out`
2. DREAM.3D file for orientation generation: 
    1. run `test-Magnesium.json` to generate `material.config`
    2. rename `material.config` to `dream3d.material.config`
    3. `grep -ir 'phi1' dream3d.material.config > orientations.dat`
    4. (manually) remove all text in `orientations.dat`, only keep number
    5. save `orientations.dat`
3. sketch (and visualize) dogbone specimen using `draw_dogbone.py`.
For example: 
- medium fillet radius
![Sketch of the dogbone specimen](./dogbone-sketch.png)
- large fillet radius
![Sketch of the dogbone specimen](./dogbone-sketch-big-fillet.png)
- small fillet radius
![Sketch of the dogbone specimen](./dogbone-sketch-small-fillet.png)

4. `geom_cad2phase.py`: dump a phase matrix from dogbone geometry
    * produce a `.npy` file: `phase_' + dumpFileName.replace('.','_') + '.npy`
5. `geom_spk2dmsk.py`: 
    1. read `dump.12.out` and `orientations.dat`
    2. write `material.config`
    3. write `spk_dump_12_out.geom`


Results from exporting SPPARKS: 
* 50um
![Geometry of the dogbone specimen](./vtk-visualization-res-50um.png)
* 20um
![Geometry of the dogbone specimen](./vtk-visualization-res-20um.png)
* 10um
![Geometry of the dogbone specimen](./vtk-visualization-res-10um.png)

# DAMASK

Note: 0.49945833333333334 volume of fraction is void.

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

# To-Do

1. Test out `restart` capability option: https://damask2.mpie.de/bin/view/Usage/SpectralSolver#Restart

##### `restart` examples
```
DAMASK_spectral --geom  PathToGeomFile/NameOfGeomFile.geom --load PathToLoadFile/NameOfLoadFile.load --workingdir PathToWorkingDir --restart XX
```

`--restart / -r / --rs XX`: Reads in total increment No. XX and continues to calculate total increment No. XX+1. Appends to existing results file 

2. Test out `postResults` capability with `--filter` option: https://damask2.mpie.de/bin/view/Documentation/PostResults

##### `postResults` Examples
* volume-averaged results of deformation gradient and first Piola-Kirchhoff stress for all increments
```
--cr f,p
```
* spatially resolved slip resistance (of phenopowerlaw) in separate files for increments 10, 11, and 12
```
--range 10 12 1 --increments --split --separation x,y,z --co resistance_slip
```
* get averaged results in slices perpendicular to x for all negative y coordinates split per increment
```
--filter 'y < 0.0'  --split --separation x --map 'avg'
```
* global sum of squared data falling into first quadrant arc between R1 and R2
```
--filter 'x >= 0.0 and y >= 0.0 and x*x + y*y >= R1*R1 and x*x + y*y <=R2*R2' --map 'lambda n,b,a: n*b+a*a'
```

See private communication with Philip Eisenlohr:
```
I believe restarting was already possible with DAMASK2. One needed to specify a restart frequency in the load file. Probably “r 10” or something to write out a restartable file every 10 increments. Restarting itself then required to add a —restart (maybe) argument to the DAMASK_spectral call and a load file that “runs” longer than the current restarting increment. I believe that the (binary) output file will just be extended with new data until the end of the load file is reached (regular termination).

The addDisplacment nodal should write a new file that contains the nodal displacements as a dataset. That data can be included when creating a VTK file (there is either two calls, one for all cell data, another for all nodal data, or both can be specified in one go...) Once the displacement is included in the VTK, you can "warp by vector" and use the displacements as source.
I believe what you are currently doing is to create cell displacements, which have the same data count as all other (cell) quantities. For (nicer) visualization, it might be advisable to add nodal displacements, which have more data points than cells (because there are one extra layer of nodes compared to cells). Then the above mentioned two-step/two-file solution is needed. Fortunately, VTK can contain both nodal and cell data in one container!

When running postResults, there is also an option to filter the data. You would probably use something along the lines of "z<upper and z>lower" with upper and lower the z-coordinates of the end of the gage section. The damask2 website (or help in postResutls) should explain this...
``

# Future directions

1. Precipitate pores into dogbone
2. Physics-informed machine learning
3. Stochastic inverse UQ with pores
4. Constitutive model calibration under uncertainty and pores
5. Microstructure-sensitive fracture?
6. Multi-fidelity UQ/SciML? Fusing computational + experimental
7. Reduced-order model for polycrystalline materials?
8. Fracture modeling with CPFEM/PF
9. Fatigue prediction


