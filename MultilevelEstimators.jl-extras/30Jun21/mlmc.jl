###############################################################################
#                                                                             #
#                                 mlmc.jl                                     #
#                                                                             #
###############################################################################

using MultilevelEstimators

# Command to run multi-index version of DREAM3D-DAMASK
get_cmd(index::Index) = Cmd(["python3", "wrapper_DREAM3D-DAMASK.py", "--index", string(index)])

# Read DREAM3D-DAMASK wrapper output and return value of estimated yield
# stress for the given level or index
function get_qoi(out, index)
    for line in split(out, "\n")
        if occursin("Estimated Yield Stress at $(index)", line)
            return parse(Float64, split(line)[end - 1])
        end
    end
    return NaN # return NaN if no value could be found
end

# Compute a sample of the multilevel or multi-index difference at the given
# level or index
function sample(index::Index)
    cmd = get_cmd(index)
    out = read(cmd, String)
    Qf = get_qoi(out, index)
    dQ = Qf
    for (key, val) in diff(index)
        Qc = get_qoi(out, key)
        dQ += val * Qc
    end
    return dQ, Qf
end

"""
    run([index_set=ML()], [max_level=3], [budget=20])

Run multilevel or multi-index Monte Carlo for with maximum level parameter 
`max_level` for approximately `buget` seconds.
"""
function run(; index_set=ML(), max_level=3, budget=20, kwargs...)

    estimator = Estimator(index_set,
                          MC(),
                          (ell, x) -> sample(ell),
                          Uniform(),
                          max_index_set_param=max_level;
                          kwargs...)

    # make history
    history = MultilevelEstimators.History()

    # run the sequence of tolerances
    tol = 1
    timer = 0
    while timer < budget # run timer until we run out of buget
        timer += @elapsed MultilevelEstimators._run(estimator, tol) # run new tolerance
        tol /= sqrt(2)
        MultilevelEstimators.update_history!(history, estimator, tol, timer)
    end

    history # return history
end
