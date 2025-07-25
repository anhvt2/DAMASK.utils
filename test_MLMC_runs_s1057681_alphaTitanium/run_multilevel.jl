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
run_multilevel(max_level=max_level,
               cost_model=level -> sum(cost_per_level[key + one(key)] for key in push!(collect(keys(diff(level))), level)),
               ε=ε,
               nb_of_warm_up_samples=nb_of_warm_up_samples
              )
