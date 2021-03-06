using MultilevelEstimators, Random, Statistics

"""
    Sandwich(he, Ed1, Ed2)

Call the external Python program Sandwich.py with element size `he`
and Young's moduli `Ed1` and `Ed2` for material 1 and 2, and return
the value for the quantity of interest.
"""
function Sandwich(he, Ed1, Ed2)
    cmd = `python3 Sandwich.py --elem $(he) --young $(Ed1) $(Ed2)`
    out = read(cmd, String)
    parse(Float64, last(split(out)))
end

"""
    sample_sandwich(level, x)

Return a sample of the multilevel difference at level `level` with
values for the random parameters `x`. This function returns a `Tuple`
`(dQ, Qf)` where `dQ` is the multilevel difference and `Qf` is the
value for the quantity of interest at the finest level.
"""
function sample_sandwich(level::Int, x)

    # solve on finest grid
    he = 0.05/2^level # he is element size
    Qf = Sandwich(he, x...)

    # compute difference when not on coarsest grid
    dQ = Qf
    if level != 0
        he *= 2
        Qc = Sandwich(he, x...)
        dQ -= Qc
    end

    dQ, Qf
end

# convenience function to call sample_lognormal with Int instead of Level
sample_sandwich(level::Level, x) = sample_sandwich(level[1], x)

# returns distributions for both random parameters (Youngs moduli)
get_distributions() = [Uniform(190e9, 210e9), Uniform(8e9, 12e9)]

"""
    run_multilevel()

Runs an MLMC simulation for the sandwich beam problem.
"""
function run_multilevel()

    # create estimator
    estimator = Estimator(ML(), MC(), sample_sandwich, get_distributions(),
                          nb_of_warm_up_samples=5, continuate=false,
                          name="Sandwich", save=false)

    # run estimator
    run(estimator, 5e-4)
end

"""
    run_multilevel_checkpoint()

Runs a checkpointed MLMC simulation for the sandwich beam problem.
This function should be called multiple times, alternating with an external
code that provides Qf.dat and dQ.dat in the samples directory.
"""
function run_multilevel_checkpoint()

    # create estimator
    estimator = Estimator(ML(), MC(), sample_sandwich, get_distributions(),
                          nb_of_warm_up_samples=5, continuate=false,
                          checkpoint=true, name="Sandwich", save=false)

    # run estimator
    run(estimator, 5e-4)
end

"""
    check_variances([max_level=3], [budget=60])

Take warm-up samples at levels 0 to `max_level` to check for
sufficient variance decay. Continuously take samples for `budget`
seconds.
"""
function check_variances(; max_level=3, budget=60)

    budget_per_level = budget/(max_level + 1)
    distributions = get_distributions()

    for level in 0:max_level
        samps_dQ = []
        samps_Qf = []
        timer = 0
        while timer < budget_per_level
            pts = transform.(distributions, rand(2)) # sample new random params
            timer += @elapsed dQ, Qf = sample_sandwich(level, pts) # compute QoI
            push!(samps_dQ, dQ)
            push!(samps_Qf, Qf)
        end
        println("Level ", level, ", V = ", var(samps_Qf), ", dV = ",
                var(samps_dQ), " (", length(samps_dQ), " samples)")
    end
end

# Call this function to check for variance decay.
# Values will be printed on the screen.
#check_variances()

# Call this function to run a simple MLMC simulation.
#Random.seed!(2021)
#run_multilevel()

# Call this function to run a checkpointed MLMC simulation.
Random.seed!(2021)
run_multilevel_checkpoint()
