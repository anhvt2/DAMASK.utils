using MultilevelEstimators, Random, Statistics

"""
    Sandwich(hex, hey, Ed1, Ed2)

Call the external Python program Sandwich.py with element sizes `hex`
and `hey` and Young's moduli `Ed1` and `Ed2` for material 1 and 2,
and return the value for the quantity of interest.
"""
function Sandwich(hex, hey, Ed1, Ed2)
    cmd = `python3 Sandwich.py --elemx $(hex) --elemy $(hey) --young $(Ed1) $(Ed2)`
    out = read(cmd, String)
    parse(Float64, last(split(out)))
end

"""
    sample_sandwich(index, x)

Return a sample of the multiindex difference at index `lindex` with
values for the random parameters `x`. This function returns a `Tuple`
`(dQ, Qf)` where `dQ` is the multiindex difference and `Qf` is the
value for the quantity of interest at the finest level.
"""
function sample_sandwich(index::Index, x)

    # solve on finest grid
    hex = 0.05/2^index[1] # hex is element size
    hey = 0.05/2^index[2] # hey is element size
    Qf = Sandwich(hex, hey, x...)

    # compute difference when not on coarsest grid
    dQ = Qf
    if index[1] != 0
        hex *= 2
        Qc = Sandwich(hex, hey, x...)
        dQ -= Qc
        hex /= 2
    end
    if index[2] != 0
        hey *= 2
        Qc = Sandwich(hex, hey, x...)
        dQ -= Qc
        hey /= 2
    end
    if index[1] != 0 && index[2] != 0
        hex *= 2
        hey *= 2
        Qc = Sandwich(hex, hey, x...)
        dQ += Qc
    end

    dQ, Qf
end

# returns distributions for both random parameters (Youngs moduli)
get_distributions() = [Uniform(190e9, 210e9), Uniform(8e9, 12e9)]

"""
    run_multiindex()

Runs an MLMC simulation for the sandwich beam problem.
"""
function run_multiindex()

    # create estimator
    estimator = Estimator(TD(2), MC(), sample_sandwich, get_distributions(),
                          nb_of_warm_up_samples=5, continuate=false,
                          name="Sandwich", save=false)

    # run estimator
    run(estimator, 5e-4)
end

"""
    run_multiindex_checkpoint()

Runs a checkpointed MLMC simulation for the sandwich beam problem.
This function should be called multiple times, alternating with an external
code that provides Qf.dat and dQ.dat in the samples directory.
"""
function run_multiindex_checkpoint()

    # create estimator
    estimator = Estimator(TD(2), MC(), sample_sandwich, get_distributions(),
                          nb_of_warm_up_samples=5, continuate=false,
                          checkpoint=true, name="Sandwich", save=false)

    # run estimator
    run(estimator, 5e-4)
end

"""
    check_variances([max_level=3], [budget=60])

Take warm-up samples at indices ∈ TD(`max_level`) to check for
sufficient variance decay. Continuously take samples for `budget`
seconds.
"""
function check_variances(; max_level=3, budget=60)

	indices = collect(get_index_set(TD(2), max_level))

    budget_per_level = budget/(length(indices) + 1)
    distributions = get_distributions()

    for index in indices
        samps_dQ = []
        samps_Qf = []
        timer = 0
        while timer < budget_per_level
            pts = transform.(distributions, rand(2)) # sample new random params
            timer += @elapsed dQ, Qf = sample_sandwich(index, pts) # compute QoI
            push!(samps_dQ, dQ)
            push!(samps_Qf, Qf)
        end
        println("Index ", index, ", V = ", var(samps_Qf), ", dV = ",
                var(samps_dQ), " (", length(samps_dQ), " samples)")
    end
end

# Call this function to check for variance decay.
# Values will be printed on the screen.
#Random.seed!(2021)
#check_variances()

# Call this function to run a simple MLMC simulation.
#Random.seed!(2021)
#run_multiindex()

# Call this function to run a checkpointed MLMC simulation.
Random.seed!(2021)
run_multiindex_checkpoint()
