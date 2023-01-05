###############################################################################
#                                                                             #
#                           run_multilevel.jl                                 #
#                                                                             #
###############################################################################

include("check_variances.jl") # load check_variances

# ========================
# @@@ SELECT INDEX SET @@@
# ========================
index_set = ML()

# ========================
# @@@ SELECT MAX LEVEL @@@
# ========================
max_level = 4 # See note below

# @@@ NOTE @@@
# !!! In a multilevel setting, `max_level=3` means 4 levels in total (i.e.,
# !!! 0, 1, 2, and 3).

# =========================
# @@@ SELECT COST MODEL @@@
# =========================
# >> Specify cost per level (e.g. in seconds)
# >> For example, if we have:
# +-------+----------------+
# | level | cost (seconds) |
# +-------+----------------+
# | 0     | 1              |
# | 1     | 2              |
# | 2     | 4              |
# | 3     | 8              |
# | 4     | 16             |
# +-------+----------------+
# Then, these values can be specified as:
# cost_per_level = [1 2 4 8 16]
cost_per_level = [1 7.51 8 19 51]

# =============================
# @@@ SELECT RMSE TOLERANCE @@@
# =============================
# >> Specify target RMSE (this is just a guess)
# >> Lower values of the RMSE means a more accurate result, but also more samples
ε = 5e0 # 1e-2: ~5e5 samples

# ========================================
# @@@ SELECT NUMBER OF WARM-UP SAMPLES @@@
# ========================================
# >> Specify number of warm-up samples on levels 0, 1, and 2
# >> This value should not be changed
nb_of_warm_up_samples = 10

# ======================
# @@@ RUN MULTILEVEL @@@
# ======================
history = run_multilevel(max_level=max_level,
               cost_model=level -> sum(cost_per_level[key + one(key)] for key in push!(collect(keys(diff(level))), level)),
               ε=ε,
               nb_of_warm_up_samples=nb_of_warm_up_samples
              )

# ==================
# @@@ PRINT INFO @@@
# ==================
open("estimators.dat", "w") do file
    write(file, "mean of QoI = $(history.data[end][:mean])\n")
    write(file, "var of QoI = $(history.data[end][:var])\n")
    write(file, "var of estimator = $(history.data[end][:varest])\n")
    write(file, "current level or index set =")
    write(file, "$(history.data[end][:current_index_set])\n")
    write(file, "expected value E[Q] = \n")
    write(file, "$(history.data[end][:E])\n")
    write(file, "expected value E[dQ] = \n")
    write(file, "$(history.data[end][:dE])\n")
    write(file, "variance V[Q] = \n")
    write(file, "$(history.data[end][:V])\n")
    write(file, "variance V[dQ] = \n")
    write(file, "$(history.data[end][:dV])\n")
    write(file, "number of samples = \n")
    write(file, "$(history.data[end][:nb_of_samples])\n")