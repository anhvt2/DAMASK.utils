
# Projection-based Reduced-order Model

These are the steps to construct a projection-based ROM. 

1. Sample train/test datasets: `sampleTrainTest.py`. Dump `{Train,Test}Idx.dat`
1. Extract data snapshots: `extractData.py`
1. Construct a global basis: `computeBasis.py`. 
1. Compute POD coefficients: `computeCoefs.py`
1. Build train/test datasets of POD coefs: `extractRomData.py` (only run this file **AFTER** running `computeCoefs.py`)
1. Train/Dump NN: `trainNN.py`
1. Predict POD coefficients $\widetilde{\boldsymbol{\lambda}}$ for unseen parameters $\widetilde{\mathbf{p}}$.
1. Reconstruct state solution using the global POD basis with predicted POD coefficients. Using the previously computed mean $\overline{\mathbf{w}}$, we can reconstruct the state variable by using the predicted POD coefficients

# Computational cost

1. Build snapshot matrix: 
    ```
    extractData.py: extracted data in 1049.07 seconds.
    extractData.py: finished in 1494.23 seconds.
    ```
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
1. Compute POD coefs:
    ```
    computeCoefs.py: Loading POD basis: Elapsed 217.87353467941284 seconds.
    computeCoefs.py: Loading POD basis: Elapsed 436.99397110939026 seconds.
    computeCoefs.py: Elapsed time = 10713.857370376587 seconds.
    ```
1. Extract ROM data:
    ```
    Elapsed time: 749.8915462493896 seconds
    Elapsed time: 976.0721864700317 seconds.
    ```
