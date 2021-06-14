###############################################################################
#                                                                             #
#                            check_variances.jl                               #
#                                                                             #
###############################################################################

using MultilevelEstimators
using Statistics
using PrettyTables
using ProgressMeter

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

# Default color in table header
get_crayon() = crayon"yellow bold"

# Column highlighter
highlight_cols(k) = Highlighter((v, i, j) -> any(j .== k), get_crayon())

# Print summary statistics
function print_table(data)
	indices = sort(collect(keys(data)))
    ell_name = length(first(keys(data))) > 1 ? "Index" : "Level"
    header = [ell_name, "|Eℓ|", "|ΔEℓ|", "Vℓ", "ΔVℓ", "Nℓ", "Wℓ"]
	E = [abs(mean(data[index][2])) for index in indices]
	dE = [abs(mean(data[index][1])) for index in indices]
	V = [var(data[index][2]) for index in indices]
	dV = [var(data[index][1]) for index in indices]
	N = [length(data[index][1]) for index in indices]
	W = [data[index][3]/length(data[index][1]) for index in indices]
    entries = hcat(indices, E, dE, V, dV, N, W)
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
function check_variances(; index_set=ML(), max_level=3, budget=20)
    indices = collect(get_index_set(index_set, max_level))
    nb_of_indices = length(indices)
    budget_per_level = round(Int, budget/(nb_of_indices + 1)) # split budget evenly over all levels/indices
    data = Dict{Index, Any}() # empty dictionary to hold samples

    # Loop over all levels/indices
    for index in indices
		samples = [Vector{NTuple{2, Float64}}(undef, 0) for _ in 1:Threads.nthreads()]
        timer = 0
        p = Progress(budget_per_level, dt=1, barglyphs=BarGlyphs("[=> ]"), color=:none)
        while timer < budget_per_level # run timer until we run out of buget
            timer += @elapsed begin
				Threads.@threads for _ in 1:Threads.nthreads()
					dQ, Qf = sample(index) # take a new sample
					if isfinite(dQ) && isfinite(Qf) # check if the sample is valid
						push!(samples[Threads.threadid()], (dQ, Qf))
					end
				end
			end
            update!(p, min(budget_per_level, floor(Int, timer))) # update progress bar
        end
		data[index] = (first.(vcat(samples...)), last.(vcat(samples...)), timer) # add new key to the dictionary
        print_table(data) # print the results
    end
end
