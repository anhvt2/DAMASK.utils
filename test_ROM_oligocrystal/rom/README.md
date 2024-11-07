
# Projection-based Reduced-order Model

These are the steps to construct a projection-based ROM. 

1. Export DAMASK output to `.npy` where only QoIs are extracted: `../damask/export2npy.py`
1. Sample train/test datasets: `sampleTrainTest.py`. Dump `{Train,Test}Idx.dat`
1. Extract data snapshots: `extractData.py`
1. Construct a global basis: `computeBasis.py`. 
1. Compute POD coefficients: `computeCoefs.py`
1. Build train/test datasets of POD coefs: `extractRomData.py` (only run this file **AFTER** running `computeCoefs.py`)
1. Train/Dump NN: `nn3d.py`
1. Predict POD coefficients $\widetilde{\boldsymbol{\lambda}}$ for unseen parameters $\widetilde{\mathbf{p}}$: `predictCoefs.py`
1. Reconstruct state solution using the global POD basis with predicted POD coefficients: `reconstructRomSolution.py`
1. Calculate FOM vs. ROM error: `calculateFomRomErrors.py`
1. Plot errors over parameter space: `plotErrorDist.py`

# Neural network architecture

1. How to load a model?
    ```python
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import logging
    class NNRegressor(nn.Module):
        def __init__(self):
            super(NNRegressor, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, numFtrs),
            )
        def forward(self, x):
            return self.network(x)

    model = NNRegressor()
    model.double()
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    ```
1. Best-so-far for `MisesCauchy`:
    ```python
    self.network = nn.Sequential(
        nn.Linear(3, 16),
        nn.Sigmoid(),
        nn.Linear(16, 32),
        nn.Sigmoid(),
        nn.Linear(32, 64),
        nn.Sigmoid(),
        nn.Linear(64, 128),
        nn.Sigmoid(),
        nn.Linear(128, numFtrs),
    )
    ```
1. Best-so-far for `MisesLnV`:
    ```python
    self.network = nn.Sequential(
        nn.Linear(3, 16),
        nn.Sigmoid(),
        nn.Linear(16, 32),
        nn.Sigmoid(),
        nn.Linear(32, 64),
        nn.Sigmoid(),
        nn.Linear(64, 128),
        nn.Sigmoid(),
        nn.Linear(128, numFtrs),
    )
    ```
    ```
    R^2 of POD coefs train for MisesCauchy = 0.969124690335161
    R^2 of POD coefs test for MisesCauchy = 0.9368248529449374
    R^2 of POD coefs train for MisesLnV = 0.9280387877871558
    R^2 of POD coefs test for MisesLnV = 0.9129412497228372
    ```

# Computational cost

1. Build snapshot matrix: `extractData.py`
    ```
    extractData.py: extracted data in 1049.07 seconds.
    extractData.py: finished in 1494.23 seconds.
    ```
1. Compute POD basis: `computeBasis.py`
    ```
    Loading time: 226.97 seconds.
    Non-zero elements = 5540 elements.
    Centering time: 17.85 seconds.
    SVD time: 55.44 seconds.
    Save time: 96.75 seconds.
    MisesCauchy: 90.0% quantile = 84 POD components
    MisesCauchy: 95.0% quantile = 168 POD components
    MisesCauchy: 99.0% quantile = 460 POD components
    MisesCauchy: 99.5% quantile = 586 POD components


    Loading time: 635.07 seconds.
    Non-zero elements = 5540 elements.
    Centering time: 18.24 seconds.
    SVD time: 80.38 seconds.
    Save time: 104.07 seconds.
    MisesLnV  : 90.0% quantile = 71 POD components
    MisesLnV  : 95.0% quantile = 149 POD components
    MisesLnV  : 99.0% quantile = 405 POD components
    MisesLnV  : 99.5% quantile = 514 POD components


    computeBasis.py: Total time for POD basis: 838.64 seconds.
    ```
1. Compute POD coefs: `computeCoefs.py`
    ```
    computeCoefs.py: Loading POD basis: Elapsed 217.87353467941284 seconds.
    computeCoefs.py: Loading POD basis: Elapsed 436.99397110939026 seconds.
    computeCoefs.py: Elapsed time = 10713.857370376587 seconds.
    ```
1. Extract ROM data: `extractRomData.py`
    ```
    Elapsed time: 749.8915462493896 seconds
    Elapsed time: 976.0721864700317 seconds.
    ```
1. Dump predicted local POD coefs: `predictCoefs.py`
    ```
    predictCoefs.py: Finish dumping local POD coefs in 160.8946762084961 seconds.
    ```
1. Reconstruct ROM solution: `reconstructRomSolution.py`
    ```
    reconstructRomSolution.py: Total elapsed time: 8394.798505783081 seconds.
    ```
1. Calculate FOM vs. ROM error: `calculateFomRomError.py`
    ```
    calculateFomRomError.py: Total elapsed time: 5243.82993721962 seconds.
    ```

# To-do

1. ~~Plot FOM vs ROM~~ (see `plotFomvRom.py`)
1. ~~Only calculate error on solids (not voids)~~
1. ~~Implement error metrics.~~ 
1. ~~Plot error vs. parameters~~
1. ~~Remove incomplete dataset in `IncompleteIdx.dat` in `sampleTrainTest.py`: replot sampling parameters~~
