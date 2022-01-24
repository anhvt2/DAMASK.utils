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

# DAMASK

## DAMASK-2.0.2

## DAMASK-2.0.3

* Change of keywords is not well documented but has been reported [https://www.researchgate.net/post/Did-anyone-used-DAMASK-code-for-crystal-plasticity-simulations-of-HCP-materials](https://www.researchgate.net/post/Did-anyone-used-DAMASK-code-for-crystal-plasticity-simulations-of-HCP-materials). 
* `covera_ratio` -> `c/a`
* require `mech            isostrain`
* require `nconstituents   1`
* disable a few fields such as
    * `# (output) e              # total strain as Green-Lagrange tensor`
    * `# (output) ee             # elastic strain as Green-Lagrange tensor`



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

# MultilevelEstimators.jl

## How to install

(also available at `./MultilevelEstimators.jl/README.md`, but this version is more comprehensive)

Mostly comes from private communication with Pieterjan Robbe

```julia
julia> ENV["JULIA_SSL_NO_VERIFY_HOSTS"] = "**" # useful behind Sandia firewall -- no certificate for downloading
pkg> dev https://github.com/PieterjanRobbe/MultilevelEstimators.jl
pkg> add Random
pkg> add Statistics

julia> import Pkg; Pkg.resolve()
julia> Pkg.add("PrettyTables")
julia> Pkg.add("ProgressMeter")

include("Example.jl")
check_variances()
run_multilevel_checkpoint()
run_multilevel()
```

Sometimes will need to switch the branch to `develop` in `~/.julia/dev/MultilevelEstimators/` and
```shell
cd ~/.julia/dev/MultilevelEstimators
git checkout develop
git pull
```

## How to setup

The following list of files needed to be checked.

1. `wrapperMIMC-DREAM3D-DAMASK.py`:
* this is a driver file for calling DREAM.3D (which generates microstructure at different resolutions)
* run DAMASK (adaptively with number of processors)
* call post-processing python script
* print the **correct output** (whether it is Young modulus or yield stress) to screen

2. `check_variance.jl`: pick up from the output
```julia
function get_qoi(out, index)
    for line in split(out, "\n")
        if occursin("Estimated Young modulus at $(index)", line)
            return parse(Float64, split(line)[end - 1])
        end
    end
    return NaN # return NaN if no value could be found
end
```
3. `run_check_variances.jl`:
* set up the overall computational budget
* weighted index set
* number of fidelities in each direction

## How to run

From the email communication around April 22, 2021

Suppose that your wrapper can be called with a script, called "my script.sh". This script would take as input the level parameter \ell. The script would then call the wrapper to compute a solution at mesh level m and mesh level m-1, extract a quantity of interest for each mesh level, and print these two numbers to screen. Let's also assume that for level \ell = 0, the script outputs the same number twice (there is no m-1 in this case). (To clarify: the "level \ell" here would be integers 0, 1, 2 … used in the MLMC method, and the "mesh level m" would correspond to the number of grid points, i.e., (m = 0) => 8x8x8, (m = 1) => 16x16x16, (m = 2) => 32x32x32, …). For example:


terminal:
```shell
$ ./myscript.sh --level 2
<QoI at mesh level 2> <QoI at mesh level 1>
$ ./myscript.sh --level 1
<QoI at mesh level 1> <QoI at mesh level 0>
$ ./myscript.sh --level 0
<QoI at mesh level 0> <QoI at mesh level 0, same number>
```

As a first step, we will look if there is variance decay. Therefore, you would need to do the following in Julia:

1) Write a "sample" function that calls your wrapper. This function will look like this:

julia function:
```julia
function my_sample_function(level, x)
cmd = `./myscript.sh --level $(level)` # command to be executed in the terminal
out = read(cmd, String) # execute command and read output
Qf, Qc = parse.(Float64, split(out)) # split output into Qf and Qc (approximations at mesh level m and mesh level m-1)
return level == 0 ? (Qf, Qf) : (Qf-Qc, Qf) # return multilevel difference and approximation at mesh level m
end
```


2) Use the (simplified) function "check_variances" to check for variance decay. I had included this function in Example.jl a couple of emails back, but here it is again:


julia function:
```julia
function check_variances(; max_level=3, budget=60)
    budget_per_level = budget/(max_level + 1)

    for level in 0:max_level
        samps_dQ = []
        samps_Qf = []
        timer = 0
        while timer < budget_per_level
            timer += @elapsed dQ, Qf = my_sample_function(level, 0) # last argument is 0 because it is not used
            push!(samps_dQ, dQ) 
            push!(samps_Qf, Qf) 
        end 
        println("Level ", level, ", V = ", var(samps_Qf), ", dV = ",
                var(samps_dQ), " (", length(samps_dQ), " samples)")
    end 
end
```


3) Write all of this in a file, and don't forget to include Statistics.jl:

file `my_julia_script.jl`:
```julia
using Statistics

function my_sample_function(level, x)
cmd = `./myscript.sh --level $(level)` # command to be executed in the terminal
out = read(cmd, String) # execute command and read output
Qf, Qc = parse.(Float64, split(out)) # split output into Qf and Qc (approximations at mesh level m and mesh level m-1)
return level == 0 ? (Qf, Qf) : (Qf-Qc, Qf) # return multilevel difference and approximation at mesh level m
end

function check_variances(; max_level=3, budget=60)
    budget_per_level = budget/(max_level + 1)

    for level in 0:max_level
        samps_dQ = []
        samps_Qf = []
        timer = 0
        while timer < budget_per_level
            timer += @elapsed dQ, Qf = my_sample_function(level, [])
            push!(samps_dQ, dQ) 
            push!(samps_Qf, Qf) 
        end 
        println("Level ", level, ", V = ", var(samps_Qf), ", dV = ",
                var(samps_dQ), " (", length(samps_dQ), " samples)")
    end 
end

# parameters:
# max_level = number of levels -1 (max_level = 2 will use level 0, 1 and 2)
# budget = total computational budget in seconds (8*3600 = 8 hours) - function will finish in this amount of time
check_variances(max_level=2, budget=8*3600)
```

4) Call this script from the terminal

terminal:
```shell
$ julia my_julia_script.jl
...
```

5) Wait for the result (will be printed to the screen)

## How to plot results

See the `Reporter` package from Pieterjan Robbe: [https://github.com/PieterjanRobbe/Reporter.jl](https://github.com/PieterjanRobbe/Reporter.jl)

## Discussion

See more at [https://groups.google.com/g/dream3d-users/c/SPCdr-BWROs?pli=1](https://groups.google.com/g/dream3d-users/c/SPCdr-BWROs?pli=1)
