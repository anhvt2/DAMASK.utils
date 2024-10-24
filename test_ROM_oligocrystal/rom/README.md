
# Projection-based Reduced-order Model

These are the steps to construct a projection-based ROM. 

1. Sample train/test datasets: `sampleTrainTest.py`. Dump `{Train,Test}Idx.dat`
1. Extract data snapshots: `extractData.py`
1. Construct a global basis: `computeBasis.py`
1. Compute POD coefficients: `computeCoefs.py`
1. Train/Dump NN: `trainNN.py`
1. Predict POD coefficients $\widetilde{\boldsymbol{\lambda}}$ for unseen parameters $\widetilde{\mathbf{p}}$.
1. Reconstruct state solution using the global POD basis with predicted POD coefficients. Using the previously computed mean $\overline{\mathbf{w}}$, we can reconstruct the state variable by using the predicted POD coefficients

