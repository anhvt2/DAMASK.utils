#!/bash

# for vareps in 

## Objective:
# Mirror effort from MC with MLMC

for i in $(seq $(cat vanilla_mc_cost.dat | wc -l)); do
	if [ "$i" -gt "1" ]; then
		# echo $i
		line=$(sed -n ${i}p vanilla_mc_cost.dat)
		vareps=$(echo $line | cut -d, -f1)
		echo $vareps

		sed -i "48s|.*|vareps = ${vareps}|" fakerun_multilevel_multiple_qoi.jl
		julia fakerun_multilevel_multiple_qoi.jl > log.mlmc.vareps-${vareps}
	fi
done

