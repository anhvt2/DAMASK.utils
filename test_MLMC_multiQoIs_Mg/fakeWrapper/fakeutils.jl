###############################################################################
#                                                                             #
#                            check_variances.jl                               #
#                                                                             #
###############################################################################

# From dummy-wrapper-DREAM3D-DAMASK.py
# =========================================================
# @@@ THIS IS A DUMMY WRAPPER - IT DOES NOT DO ANYTHING @@@
# =========================================================

# example usage
# =============
#
# $ python wrapper-DREAM3D-DAMASK.py --level 0
# Collocated von Mises stresses at 0 is -2.921755914475476
#
# $ python wrapper-DREAM3D-DAMASK.py --level 1
# Collocated von Mises stresses at 1 is 1.4565607955675095
# Collocated von Mises stresses at 0 is -2.4967721807515217
#
# $ python wrapper-DREAM3D-DAMASK.py --level 3 --nb_of_qoi 4
# Collocated von Mises stresses at 3 is 0.2761242362928689, -0.9839518678051921, -0.1033699430980116, 0.13297340048334058
# Collocated von Mises stresses at 2 is 0.8931224201628596, -0.42128388915439724, 0.3768735421579537, 0.24507689336645652
#
# $ python wrapper-DREAM3D-DAMASK.py --index (2, 1) --nb_of_qoi 3
# Collocated von Mises stresses at (2, 1) is -2.5457461714149145, -0.02181018669814895, -2.3552475764029435
# Collocated von Mises stresses at (1, 1) is 0.501231535869913, 0.5191660280513454, -2.1937281246076665
# Collocated von Mises stresses at (2, 0) is -0.10207593026854639, 1.547139317467864, -0.9352563652562738
# Collocated von Mises stresses at (1, 0) is -0.5373920568687983, -0.31172737038566456, 0.24969705569949627

using MultilevelEstimators
using Statistics
using PrettyTables
using ProgressMeter

# Commands to run DREAM3D-DAMASK
get_cmd(index::Index, nb_of_qoi) = Cmd(["python3", "fakewrapper_multilevel_multiple_qoi.py", index isa Level ? "--level" : "--index", string(index), "--nb_of_qoi", string(nb_of_qoi)])

# Read DREAM3D-DAMASK wrapper output and return value of estimated yield stress for the given level or index
function get_qoi(out, index)
    for line in split(out, "\n")
        if occursin("Collocated von Mises stresses at $(index)", line)
            return [parse(Float64, split(words)[end]) for words in split(split(line, ")")[end], ", ")]
        end
    end
    return nothing # return NaN if no value could be found
end

# Compute a sample of the multilevel or multi-index difference at the given level or index
function sample(index::Index, nb_of_qoi::Int)
    out = nothing
    while isnothing(out)
        cmd = get_cmd(index, nb_of_qoi)
        out = read(cmd, String)
    end
    Qf = get_qoi(out, index)
    dQ = Qf
    for (key, val) in diff(index)
        Qc = get_qoi(out, key)
        dQ += val * Qc
    end
    return dQ, Qf
end

# Default color in table header
get_crayon() = crayon"yellow bold"

# Column highlighter
highlight_cols(k) = Highlighter((v, i, j) -> any(j .== k), get_crayon())

# Print summary statistics
function print_table(data)
    ell_name = length(first(keys(data))) > 1 ? "Index" : "Level"
    header = [ell_name, "|Eℓ|", "|ΔEℓ|", "Vℓ", "ΔVℓ", "Nℓ", "Wℓ"]
    E = [abs(mean(val[2])) for (key, val) in data]
    dE = [abs(mean(val[1])) for (key, val) in data]
    V = [var(val[2]) for (key, val) in data]
    dV = [var(val[1]) for (key, val) in data]
    N = [length(val[1]) for (key, val) in data]
    W = [length(val[1])/val[3] for (key, val) in data]
    entries = hcat(sort(collect(keys(data))), E, dE, V, dV, N, W)
    e_fmt = ft_printf("%5.3e", [2, 3, 4, 5, 7])
    pretty_table(entries,
                 header,
                 header_crayon=get_crayon(),
                 header_alignment=:c, 
                 formatters = e_fmt,
                 highlighters=highlight_cols(1),
                 equal_columns_width=true)
end

"""
check_variances([index_set=ML()], [max_level=3], [budget=20])

Check multilevel or multi-index variance decay for level or index set
`index_set` with maximum level parameter `max_level` by taking samples 
on each index for approximately `buget` seconds.
"""
function check_variances(; index_set=ML(), max_level=3, budget=20, nb_of_qoi=1)
    indices = collect(get_index_set(index_set, max_level))
    nb_of_indices = length(indices)
    budget_per_level = round(Int, budget/(nb_of_indices + 1)) # split budget evenly over all levels/indices
    data = Dict{Index, Any}() # empty dictionary to hold samples

    # Loop over all levels/indices
    for index in indices
        samples_Qf = [] # vector to store samples of Qf
        samples_dQ = [] # vector to store samples of dQ
        timer = 0
        p = Progress(budget_per_level, dt=1, barglyphs=BarGlyphs("[=> ]"), color=:none)
        while timer < budget_per_level # run timer until we run out of buget
            timer += @elapsed dQ, Qf = sample(index, nb_of_qoi) # take a new sample
            if isfinite(dQ) && isfinite(Qf) # check if the sample is valid
                push!(samples_dQ, dQ) # update dQ
                push!(samples_Qf, Qf) # update Qf
            end
            update!(p, min(budget_per_level, floor(Int, timer))) # update progress bar
        end
        data[index] = (samples_dQ, samples_Qf, timer) # add new key to the dictionary
        print_table(data) # print the results
    end
end

"""
    run_multilevel([max_level=4], [cost_model=MultilevelEstimators.EmptyFunction()], [ε=1e-2], [nb_of_warm_up_samples=nb_of_warm_up_samples])

Runs an MLMC simulation.
"""
function run_multilevel(; max_level=4, cost_model=MultilevelEstimators.EmptyFunction(), ε=1e1, nb_of_warm_up_samples=nb_of_warm_up_samples, nb_of_qoi=1)

    # create estimator
    estimator = Estimator(ML(),
                          MC(),
                          (level, x) -> sample(level, nb_of_qoi),
                          [Uniform()], # placeholder, not important
                          nb_of_warm_up_samples=nb_of_warm_up_samples,
                          max_index_set_param=max_level,
                          name="DREAM3D-multilevel",
                          save_samples=true,
                          cost_model=cost_model,
                          nb_of_qoi=nb_of_qoi
                         )

    # run estimator
    ## Note: do NOT run forever
    run(estimator, [ε])
    ## Note: run forever
    # while true
    #     run(estimator, [ε])
    #     ε /= 2^(1/8)
    # end
end

"""
    run_multiindex([index_set=AD(2)], [max_search_space=FT(3)], [max_level=max_level], [cost_model=MultilevelEstimators.EmptyFunction()], [ε=1e-2], [nb_of_warm_up_samples=nb_of_warm_up_samples])

Runs an MIMC simulation.
"""
function run_multiindex(; index_set=AD(2), max_search_space=FT(3), max_level=4, cost_model=MultilevelEstimators.EmptyFunction(), ε=1e1, nb_of_warm_up_samples=nb_of_warm_up_samples, nb_of_qoi=1)

    # create estimator
    estimator = Estimator(index_set,
                          MC(),
                          (index, x) -> sample(index, nb_of_qoi),
                          [Uniform()], # placeholder, not important
                          nb_of_warm_up_samples=nb_of_warm_up_samples,
                          max_index_set_param=max_level,
                          name="DREAM3D-multiindex",
                          save_samples=true,
                          cost_model=cost_model,
                          max_search_space=max_search_space,
                          nb_of_qoi=nb_of_qoi
                         )

    # run estimator
    while true
        run(estimator, [ε])
        ε /= 2^(1/8)
    end
end
