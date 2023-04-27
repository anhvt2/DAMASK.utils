#!/bash

# for vareps in 

## Objective:
# Mirror effort from MC with MLMC

# rm -rfv log.mlmc
# mkdir -p log.mlmc

python3 get_vareps_mlmc.py

for vareps in $(cat vareps_mlmc.dat); do
	# sed -i "48s|.*|vareps = ${vareps}|" fakerun_multilevel_multiple_qoi.jl
	sed -i "48s|.*|vareps = ${vareps}|" fakerun_2levels_multiple_qoi.jl
	python3 cleanseDataset.py # reset database

	if [ ! -e "log.mlmc.vareps-${vareps}" ]; then
		# rm -f nohup.out; nohup julia fakerun_multilevel_multiple_qoi.jl 2>&1 > log.mlmc.vareps-${vareps} &
		# julia fakerun_multilevel_multiple_qoi.jl 2>&1 > log.mlmc.vareps-${vareps}
		julia fakerun_2levels_multiple_qoi.jl 2>&1 > log.mlmc/log.mlmc.vareps-${vareps}
	fi
done

