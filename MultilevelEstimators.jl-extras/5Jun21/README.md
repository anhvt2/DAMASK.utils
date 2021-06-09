
Hi Anh,

Thank you! Attached are the updated files. 

Please, before running these files, you should do the following:

- Install two additional Julia packages ("PrettyTables" and "ProgressMeter"). You can do this as follows:
(1) From the terminal, type `julia` to enter the Julia REPL:

```julia
==============================================================================
Pieterjan@curie: julia
[…]
==============================================================================
```

(2) Enter the following:

```julia
julia> using Pkg; Pkg.add("PrettyTables"); Pkg.add("ProgressMeter")
```

(3) Exit Julia:

```julia
julia> exit()
   This will install both packages.
```julia

Here's an overview of the content of the files (please, read this carefully because some things have changed with respect to the Python version):

*1. check_variances.jl*

This is the main driver script. You should not change anything inside this file.

*2. run_check_variances.jl*

This is the file that should be run if you want to start the analysis. To run the example, type `julia run_check_variances.jl` in the terminal:

```shell
==============================================================================
Pieterjan@curie: julia run_check_variances.jl
[…]
==============================================================================
```

Inside this file, there are some options you can change. Basically, you have to select the following

(a) the index set: choose between multilevel / multi-index (total degree or full tensor)

(b) the maximum level parameter: this controls the size of the index set (the number of levels/indices)

(c) the total run time: total run time of the script in seconds

Please read "run_check_variances.jl" carefully for details on how to specify the options.

*3. wrapper_DREAM3D-DAMASK.py*

This is an example wrapper that you can use. For now, it returns only dummy values, but you should replace it by your own wrapper (just delete this file and replace it with your own file with the same name.

Please have a look at this file to see how you should implement your own wrapper for multi-index sampling. The most difficult part here is deciding on which coarser levels to run. You can easily generate all possible combinations of coarser levels with the "inter tools.product" command (see line 41).

One very important remark is that the level or index MUST be included in the print statement for the estimated yield stress. If you don't specify the level or index in the exact same way as is done in the example wrapper, the Julia code will NOT work. So, instead of 

```shell
==============================================================================
Pieterjan@curie: python3 wrapper_DREAM3D-DAMASK.py --index="2"
[…]
Estimated Yield Stress is 0.7978621372004265 GPa
[…]
Estimated Yield Stress is 0.9908208613578336 GPa
[…]
==============================================================================
```

your wrapper should output something like

```shell
==============================================================================
Pieterjan@curie: python3 wrapper_DREAM3D-DAMASK.py --index="2"
[…]
Estimated Yield Stress at 2 is 0.7978621372004265 GPa
[…]
Estimated Yield Stress at 1 is 0.9908208613578336 GPa
[…]
==============================================================================
```

Notice the additional "at 2" and "at 1" here!

Here are some multi-index examples:

```shell
==============================================================================
Pieterjan@curie: python3 wrapper_DREAM3D-DAMASK.py --index="(2, 3, 0)"
[…]
Estimated Yield Stress at (2, 3, 0) is 0.35010737298313654 GPa
[…]
Estimated Yield Stress at (2, 2, 0) is 0.30866029960987396 GPa
[…]
Estimated Yield Stress at (1, 3, 0) is 0.4955928185601899 GPa
[…]
Estimated Yield Stress at (1, 2, 0) is 0.8942111690664385 GPa
[…]
==============================================================================
```

```shell
==============================================================================
Pieterjan@curie: python3 wrapper_DREAM3D-DAMASK.py --index="(1, 1, 1)"
[…]
Estimated Yield Stress at (1, 1, 1) is 0.8086862342957798 GPa
[…]
Estimated Yield Stress at (1, 1, 0) is 0.02220030510572535 GPa
[…]
Estimated Yield Stress at (1, 0, 1) is 0.08552417223175368 GPa
[…]
Estimated Yield Stress at (1, 0, 0) is 0.06101256487316187 GPa
[…]
Estimated Yield Stress at (0, 1, 1) is 0.32105797242385414 GPa
[…]
Estimated Yield Stress at (0, 1, 0) is 0.274867472834313 GPa
[…]
Estimated Yield Stress at (0, 0, 1) is 0.9530657170603737 GPa
[…]
Estimated Yield Stress at (0, 0, 0) is 0.32493901521853585 GPa
[…]
==============================================================================
```

Please, let me know if you're having trouble running these files. I'd be happy to assist you in case you're having problems constructing the wrapper for the multi-index case.

Best,
Pieterjan