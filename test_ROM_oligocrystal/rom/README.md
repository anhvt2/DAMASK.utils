
# Projection-based Reduced-order Model

These are the steps to construct a projection-based ROM. 

1. Sample train/test datasets: `sampleTrainTest.py`. Dump `{Train,Test}Idx.dat`
1. Extract data snapshots: `extractData.py`
1. Construct a global basis: `computeBasis.py`. 
1. Compute POD coefficients: `computeCoefs.py`
1. Train/Dump NN: `trainNN.py`
1. Predict POD coefficients $\widetilde{\boldsymbol{\lambda}}$ for unseen parameters $\widetilde{\mathbf{p}}$.
1. Reconstruct state solution using the global POD basis with predicted POD coefficients. Using the previously computed mean $\overline{\mathbf{w}}$, we can reconstruct the state variable by using the predicted POD coefficients

# Computational cost

1. Build snapshot matrix: `extractData.py: finished in 1610.32 seconds.`
1. Compute POD basis: 
```
Loading time: 221.89 seconds.
Non-zero elements = 5524 elements.
Centering time: 34.98 seconds.
SVD time: 633.08 seconds.
Save time: 215.23 seconds.

Loading time: 1334.92 seconds.
Non-zero elements = 5524 elements.
Centering time: 24.71 seconds.
SVD time: 594.55 seconds.
Save time: 220.47 seconds.

Total time for POD basis: 2177.61 seconds.
```
