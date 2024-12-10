#!/!bin/bash
for TestIdx in $(cat TestIdxs.txt); do
    for i in $(seq 120); do
        foldername="test-run-${TestIdx}-${i}"
        sourceFolderName="res-50um-additive-run-${TestIdx}"
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

