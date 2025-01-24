#!bin/bash
for folderName in $(find . -mindepth 1 -maxdepth 1 -type d); do
    cd $folderName
    for fileName in $(ls -1v CBayesResults*.png); do
        cp ${fileName} ../${folderName}-${fileName};
    done
    cd ..
done
