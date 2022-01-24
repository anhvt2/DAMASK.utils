
# Uncertainty quantification for constitutive models in CPFEM

## Phenomenological model

* See `phenomenological/` 
* Use `Phase_Phenopowerlaw_Copper.config` as an example, vary 5 parameters according to `sedighiani2020efficient`, similar to Table 2.

```
@article{sedighiani2020efficient,
  title={An efficient and robust approach to determine material parameters of crystal plasticity constitutive laws from macro-scale stress--strain curves},
  author={Sedighiani, Karo and Diehl, Martin and Traka, Konstantina and Roters, F and Sietsma, Jilt and Raabe, Dierk},
  journal={International Journal of Plasticity},
  volume={134},
  pages={102779},
  year={2020},
  publisher={Elsevier}
}
```

## Dislocation-density-based model

* See `dislocation-density-based/`
* Use `Phase_Phenopowerlaw_Magnesium.config` as an example, vary 16 parameters according to `sedighiani2020efficient`, similar to Table 8.

```
@article{sedighiani2020efficient,
  title={An efficient and robust approach to determine material parameters of crystal plasticity constitutive laws from macro-scale stress--strain curves},
  author={Sedighiani, Karo and Diehl, Martin and Traka, Konstantina and Roters, F and Sietsma, Jilt and Raabe, Dierk},
  journal={International Journal of Plasticity},
  volume={134},
  pages={102779},
  year={2020},
  publisher={Elsevier}
}
```

## Roadmap for using Dakota (dakota.sandia.gov)

Steps:

1. Create a dummy Dakota input file with the appropriate parameters, domains and distributions for the actual model.
2. Run the dummy input file and save the input samples.
3. Run the CPFEM code using these sample points.
4. Create the actual Dakota input file that just reads in the sample data and does not call anything else.

Example: `dakota-6.15/build/test/examples-users/rosen_uq_sc.in`

```
# Dakota Input File: rosen_uq_sc.in

environment

method
  stoch_collocation
    sparse_grid_level = 3 # subject to change
    # dimension_preference = 2 1 # switch from anisotropic to isotropic SG
    samples_on_emulator = 10000 seed = 12347
    response_levels = .1 1. 50. 100. 500. 1000.
    variance_based_decomp #interaction_order = 1
  # output silent
  output verbose # print input parameters

variables
  uniform_uncertain = 2 # Legendre polynomial
    lower_bounds      = -1.0  +1.0
    upper_bounds      = -1.0  +1.0
    descriptors       = 'x1'  'x2'

interface
  analysis_drivers = 'textbook'
    direct

responses
  response_functions = 1
  no_gradients
  no_hessians
```

```shell
cd /home/anhvt89/Documents/dakota/6.15/build/test/examples-users
../../test/dakota -i rosen_uq_sc.in  > dakota.log
```

For input parameters, see `dakota_sparse_tabular.dat` file generated from revoking the above command.
