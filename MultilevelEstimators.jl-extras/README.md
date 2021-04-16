
# Instruction to run MultilevelEstimators.jl

Mostly comes from private communication with Pieterjan Robbe

```julia
pkg> dev https://github.com/PieterjanRobbe/MultilevelEstimators.jl

julia> import Pkg; Pkg.resolve()

include("Example.jl")
check_variances()
run_multilevel_checkpoint()
run_multilevel()
```

Sometimes will need to switch the branch to `develop` in `~/.julia/dev/MultilevelEstimators/` and
```shell
git pull
git checkout develop
```

