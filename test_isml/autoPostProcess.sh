#!/bin/bash

for fileName in 'origVoid' 'RobustCovariance' 'OneClassSVM' 'IsolationForest' 'LOF'; do
    python3 ../npy2geom.py --npy ${fileName}.npy
    geom_check ${fileName}.geom
    python3 plotVoid3dAnomalyDetection.py --fileName ${fileName}.vtr
done

