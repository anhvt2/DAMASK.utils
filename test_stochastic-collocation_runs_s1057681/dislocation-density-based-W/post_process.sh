#!/bin/bash
postResults single_phase_equiaxed_tension.spectralOut --cr f,p

if [ -d "postProc" ]; then
	cd postProc/

	filterTable < single_phase_equiaxed_tension.txt --white inc,1_f,1_p > stress_strain.log
	# addStrainTensors one_phase_equiaxed_tension.txt --left --logarithmic
	# addCauchy one_phase_equiaxed_tension.txt
	# addMises one_phase_equiaxed_tension.txt --strain 'ln(V)' --stress Cauchy
	# filterTable < one_phase_equiaxed_tension.txt --white inc,'Mises(ln(V))','Mises(Cauchy)' > log.stress_strain.txt

	cp ../tension.load . 
	# check $1 argument in running this script, i.e. bash run_damask.sh $1
	if [[ $1 == "YieldStress" ]]; then 
		python3 ../computeYieldStress.py
	elif [[ $1 == "YoungModulus" ]]; then
		python3 ../computeYoungModulus.py
	else
		echo "run_damask_2.0.3.sh: \$1 argument is not detected in run_damask_2.0.3.sh"
	fi

	if [ -f "output.dat" ]; then
		echo 1 > ../log.feasible
		# needed in wrapper_DREAM3D-DAMASK.py
	fi
	cd ..
else
	echo "Simulation does not converge!!!"
	echo 0 > log.feasible
fi
rm -v *.spectralOut

