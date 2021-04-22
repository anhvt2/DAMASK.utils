using MultilevelEstimators, Random, Statistics

function sample_damask(level, x)
	cmd = `python3 ./wrapper_DREAM3D-DAMASK.py --level=$(level)` # command to be executed in the terminal
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
            timer += @elapsed dQ, Qf = sample_damask(level, [])
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

