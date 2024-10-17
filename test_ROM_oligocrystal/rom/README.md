
# Projection-based Reduced-order Model

1. Construct a global basis.
1. Compute POD coefficients.
1. Predict POD coefficients $\widetilde{\boldsymbol{\lambda}}$ for unseen parameters $\widetilde{\mathbf{p}}$.
1. Reconstruct state solution using the global POD basis with predicted POD coefficients. Using the previously computed mean $\overline{\mathbf{w}}$, we can reconstruct the state variable by using the predicted POD coefficients

