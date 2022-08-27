for folderName in $(ls -1dv fpga_Iter*/); do
	cd $folderName
	sh ../getInput.sh
	python ../getOutput.py
	echo "done $folderName"
	cd ..
done


