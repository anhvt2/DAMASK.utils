using FileIO
using JLD2
using MultilevelEstimators
using DelimitedFiles

function write_frac_cost(infile, outfile)
    hist = load(infile)["history"]
    total_cost = [sum(hist[k][:W][idx] * hist[k][:nb_of_samples][idx + one(idx)] for (n, idx) in enumerate(hist[k][:index_set])) for k in 1:length(hist)]
    cost_frac = zeros(length(hist), length(hist[length(hist)][:index_set]))
    for k in 1:length(hist)
        for (n, idx) in enumerate(hist[k][:index_set])
            cost_frac[k, n] = hist[k][:W][idx] * hist[k][:nb_of_samples][idx + one(idx)] / total_cost[k] * 100
        end
    end
    open(outfile, "w") do f
        write(f, join(hist[length(hist)][:index_set], "\t"), "\t", "total", "\n")
        writedlm(f, hcat(cost_frac, total_cost))
    end
end

infile = "test_MLMC_runs_s1057681_alphaTitanium.jld2"
outfile = "mlmc_frac_cost.txt"
write_frac_cost(infile, outfile)

infile = "test_MIMC_runs_s1057681_Aluminum.jld2"
outfile = "mimc_frac_cost.txt"
write_frac_cost(infile, outfile)