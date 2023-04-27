#!/bash

# for vareps in 

## Objective:
# Mirror effort from MC with MLMC

mkdir -p log.mlmc

for i in $(seq $(cat vanilla_mc_cost.dat | wc -l)); do
	if [ "$i" -gt "1" ]; then
		# echo $i
		line=$(sed -n ${i}p vanilla_mc_cost.dat)
		vareps=$(echo $line | cut -d, -f1)
		echo $vareps

		# sed -i "48s|.*|vareps = ${vareps}|" fakerun_multilevel_multiple_qoi.jl
		sed -i "48s|.*|vareps = ${vareps}|" fakerun_2levels_multiple_qoi.jl
		python3 cleanseDataset.py # reset database

		if [ ! -e "log.mlmc.vareps-${vareps}" ]; then
			# rm -f nohup.out; nohup julia fakerun_multilevel_multiple_qoi.jl 2>&1 > log.mlmc.vareps-${vareps} &
			# julia fakerun_multilevel_multiple_qoi.jl 2>&1 > log.mlmc.vareps-${vareps}
			julia fakerun_2levels_multiple_qoi.jl 2>&1 > log.mlmc/log.mlmc.vareps-${vareps}
		fi



	fi
done

