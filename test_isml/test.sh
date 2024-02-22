#!/bin/bash

for fileName in 'RobustCovariance' 'OneClassSVM' 'IsolationForest' 'LOF'; do
    python3 ../npy2geom.py --npy ${fileName}.npy
    geom_check ${fileName}.geom
done

