#!/!bin/bash
for i in $(cat TestIdxs.txt); do
    for j in $(seq 120); do
        foldername="test-run-${i}-${j}"
        sourceFolderName="res-50um-additive-run-${i}"
        mkdir -p ${foldername}
        cd ${foldername}
        cp -f ../${sourceFolderName}/{sbatch,in,dump}.* .
        cp -f ../${sourceFolderName}/*.{geom,vtr,py,sh,png,log,dat,npy,txt} .
        cp -f ../${sourceFolderName}/tension.load .
        cp -f ../${sourceFolderName}/*.config .
        cd ..
    done
    echo "Done ${foldername}"
done

