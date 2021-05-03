using Statistics

function sample_damask(level, x)
	#cmd = `python3 ./wrapper_DREAM3D-DAMASK.py --level=$(level)` # command to be executed in the terminal
	cmd = `cat out.txt`
	out = read(cmd, String) # execute command and read output
	println("============= actual output =============")
	println(out)
	println("=========================================")
	lines = split(out, "\n") # split output into lines
	Q = [] # empty array
	for line in lines # loop over all lines
		if occursin("Estimated Yield Stress =", line) # if line containes "Estimated Yield Stress"
			words = split(line) # split the line into words
			yield_stress = parse(Float64, words[end-1]) # parse the yield stress value to Float64
			push!(Q, yield_stress) # append the value of the yield stress to the array "Q"
		end
	end
	if level == 0
		Qf = first(Q)
		return Qf, Qf # return twice Qf
	else
		Qf = first(Q)
		Qc = last(Q)
		dQ = Qf - Qc
		return dQ, Qf # return multilevel difference and Qf
	end
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
				var(samps_dQ), " (", length(samps_dQ), 
				" samples, cost per sample = ", timer / length(samps_dQ), ")")
	end 
end

# parameters:
# max_level = number of levels -1 (max_level = 2 will use level 0, 1 and 2)
# budget = total computational budget in seconds (8*3600 = 8 hours) - function will finish in this amount of time
check_variances(max_level=2, budget=20)# 8*3600)
