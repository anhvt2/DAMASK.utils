
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

4. **17-4PH** dogbone

* [https://doi.org/10.1016/j.jmatprotec.2016.10.023](https://doi.org/10.1016/j.jmatprotec.2016.10.023): High-throughput stochastic tensile performance of additively manufactured stainless steel


# SPPARKS (+ DREAM.3D)

~~need a parser from SPPARKS to DAMASK `.geom` file~~: `geom_spk2dmsk.py`

Attempts:
    1. 3D grain growth as conventional microstructure to benchmark AM: `spk/in.potts_3d`
    2. Simulating AM with parameters: `spk/in.potts_additive_dogbone`

SPPARKS commands used:
* [https://spparks.github.io/doc/variable.html](https://spparks.github.io/doc/variable.html)
* [https://spparks.github.io/doc/am_cartesian_layer.html](https://spparks.github.io/doc/am_cartesian_layer.html)
* [https://spparks.github.io/doc/am_pass.html](https://spparks.github.io/doc/am_pass.html)
* [https://spparks.github.io/doc/am_build.html](https://spparks.github.io/doc/am_build.html)
* [https://spparks.github.io/doc/app_potts_am_path_gen.html](~~https://spparks.github.io/doc/app_potts_am_path_gen.html~~)


Dimension: 10 mm (4 mm middle) x 6 mm x 1 mm

Attempted resolution:

```
dump.20.out -> spk/res-20um/dump.20.out
dump.12.out -> spk/res-50um/dump.12.out
dump.10.out -> spk/res-10um/dump.10.out
```

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
6. `../writeGeom.py`: write `.geom` file from a complete header and flatten 1d `geom` array
7. `../readGeom.py`: read `geom` file into 3d numpy array and other header variables (grid, size, etc.)



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
3. void config is adopted from `Phase_Isotropic_FreeSurface.config` based on a conversation with Philip Eisenlohr.
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
4. create `vtr` file by `geom_check` command in DAMASK pre-/post-processing script. 

# To-Do

1. Implement a visualization pipeline for visualizing microstructure growth due to AM. Some ideas:

* Only visualize the different part when comparing with initial microstructure
* Only visualize the different between current and last time-step

2. Test out `restart` capability option: https://damask2.mpie.de/bin/view/Usage/SpectralSolver#Restart

##### `restart` examples
```
DAMASK_spectral --geom  PathToGeomFile/NameOfGeomFile.geom --load PathToLoadFile/NameOfLoadFile.load --workingdir PathToWorkingDir --restart XX
```

`--restart / -r / --rs XX`: Reads in total increment No. XX and continues to calculate total increment No. XX+1. Appends to existing results file 

2. Test out `postResults` capability with `--filter` option: https://damask2.mpie.de/bin/view/Documentation/PostResults

3. Implement a AM SPPARKS apps to generate microstructure from SPPARKS
* **with possible visualization**

4. ~~Implement a `pyvista` script that shows stresses warped by a deformed geometry.~~

A DAMASK example is given by: https://damask2.mpie.de/bin/view/Usage/SpectralSolver
```
postResults --cr f,p --split --separation x,y,z 20grains16x16x16_tensionX.spectralOut

cd postProc
viewTable -a 20grains16x16x16_tensionX_inc100.txt

addCauchy 20grains16x16x16_tensionX_inc100.txt
addMises -s Cauchy 20grains16x16x16_tensionX_inc100.txt
viewTable -a 20grains16x16x16_tensionX_inc100.txt

addStrainTensors --left --logarithmic 20grains16x16x16_tensionX_inc100.txt
addMises -e 'ln(V)' 20grains16x16x16_tensionX_inc100.txt
viewTable -a 20grains16x16x16_tensionX_inc100.txt

vtk_rectilinearGrid 20grains16x16x16_tensionX_inc100.txt
vtk_addRectilinearGridData \
 --data 'Mises(Cauchy)',1_p,'1_ln(V)',1_Cauchy \
 --vtk '20grains16x16x16_tensionX_inc100_pos(cell).vtr' \
 20grains16x16x16_tensionX_inc100.txt

addDisplacement --nodal 20grains16x16x16_tensionX_inc100.txt

vtk_addRectilinearGridData \
 --data 'fluct(f).pos','avg(f).pos' \
 --vtk '20grains16x16x16_tensionX_inc100_pos(cell).vtr' \
 20grains16x16x16_tensionX_inc100_nodal.txt
```
followed by `Filters` > `Common` > `Warp By Vector` in ParaView from the menu and select first `avg(f).pos`. Select the new entry in the pipeline to visualize the uniformly deformed geometry. Similarly, choose `Filters` > `Common` > `Warp By Vector` from the menu and select first `fluct(f).pos` to also see the fluctuations resulting from the solution of static mechanical equilibrium. 

An example is given in the PyVista documentation: https://docs.pyvista.org/version/stable/examples/01-filter/warp-by-vector.html

Some possible hints include

* https://github.com/pyvista/pyvista/issues/650
* https://discourse.paraview.org/t/visualizing-stress-on-the-deformed-geometry/654

3. Seed void from a void geometry dictionary. Maybe see some works from experimentalists, e.g. Andrew Polonsky, Philip Noell

* Polonsky, A. T., Madison, J. D., Arnhart, M., Jin, H., Karlson, K. N., Skulborstad, A. J., ... & Murawski, S. G. (2023). Toward accurate prediction of partial-penetration laser weld performance informed by three-dimensional characterization–Part I: High fidelity interrogation. Tomography of Materials and Structures, 2, 100006.
* Karlson, K. N., Skulborstad, A. J., Madison, J. D., Polonsky, A. T., Jin, H., Jones, A., ... & Lu, W. Y. (2023). Toward accurate prediction of partial-penetration laser weld performance informed by three-dimensional characterization–part II: μCT based finite element simulations. Tomography of Materials and Structures, 2, 100007.
* Madison, J. D., & Aagesen, L. K. (2012). Quantitative characterization of porosity in laser welds of stainless steel. Scripta Materialia, 67(9), 783-786.

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
I believe restarting was already possible with DAMASK2. One needed to specify a restart frequency in the load file. Probably "r 10" or something to write out a restartable file every 10 increments. Restarting itself then required to add a —restart (maybe) argument to the DAMASK_spectral call and a load file that “runs” longer than the current restarting increment. I believe that the (binary) output file will just be extended with new data until the end of the load file is reached (regular termination).

The addDisplacment nodal should write a new file that contains the nodal displacements as a dataset. That data can be included when creating a VTK file (there is either two calls, one for all cell data, another for all nodal data, or both can be specified in one go...) Once the displacement is included in the VTK, you can "warp by vector" and use the displacements as source.
I believe what you are currently doing is to create cell displacements, which have the same data count as all other (cell) quantities. For (nicer) visualization, it might be advisable to add nodal displacements, which have more data points than cells (because there are one extra layer of nodes compared to cells). Then the above mentioned two-step/two-file solution is needed. Fortunately, VTK can contain both nodal and cell data in one container!

When running postResults, there is also an option to filter the data. You would probably use something along the lines of "z<upper and z>lower" with upper and lower the z-coordinates of the end of the gage section. The damask2 website (or help in postResutls) should explain this...
```

```
I am glad the simulations did work out. Looking at the structure now in more detail, I realize that the geometry is actually not a dogbone, but a dogsheet (fully periodic in Y). You might need to add some pixels of air to break material continuity along Y...

That your workstation is faster than an HPC might be attributable to differences in their hardware? An undergrad student of mine recently compared the efficiency of using cores for MPI or thread (openMP) parallelization and found that for large (enough) problems, the first about 8 cores can be indiscriminately distributed as either. With more resources at one’s disposal, throwing those towards MPI makes better sense since openMP appears to peter out in the high single digits (of threads).

What to do with such simulations is a good question. We all know that averaged results (for “simple” materials) are usually quite accurately reproduced. I am fairly certain that the volume averaged sigma_zz in the gage section would be equal to the volume average of the farthest (or probably any) Z layer of the widest head section. Together with either the displacement data at the ends of the gage section or the volume averaged gage strain, this stress–strain curve should reflect macroscopic behavior. Now, of course, you are actually not having a large number of grains in the gage section... Hence, evaluating the intrinsic variability (maybe as function of texture sharpness) for micro samples (with only "few" grains) might be a nice exercise (and example to be put on the DAMASK website).
```

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


