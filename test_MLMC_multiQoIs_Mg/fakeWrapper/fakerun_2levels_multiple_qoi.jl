###############################################################################
#                                                                             #
#                           run_multilevel.jl                                 #
#                                                                             #
###############################################################################

include("fakeutils_2levels.jl") # load fakeutils w/ fake Python wrapper -- only 2 levels {3,4}

# ========================
# @@@ SELECT INDEX SET @@@
# ========================
index_set = ML()

# ========================
# @@@ SELECT MAX LEVEL @@@
# ========================
max_level = 1 # See note below

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
# cost_per_level = [4    330    1920    3270    12452]
# Add cost to generate microstructures: +35 seconds
cost_per_level = [3305    12487]

# =============================
# @@@ SELECT RMSE TOLERANCE @@@
# =============================
# >> Specify initial RMSE (this is just a guess)
# >> Lower values of the RMSE means a more accurate result, but also more samples
vareps = 4.0e-1

# ========================================
# @@@ SELECT NUMBER OF WARM-UP SAMPLES @@@
# ========================================
# >> Specify number of warm-up samples on levels 0, 1, and 2
# >> This value should not be changed
nb_of_warm_up_samples = 3

# =============================
# @@@ SPECIFY NUMBER OF QOI @@@
# =============================
nb_of_qoi = 10

# ======================
# @@@ RUN MULTILEVEL @@@
# ======================
run_multilevel(max_level=max_level,
               cost_model=level -> sum(cost_per_level[key + one(key)] for key in push!(collect(keys(diff(level))), level)),
               ε=vareps,
               nb_of_warm_up_samples=nb_of_warm_up_samples,
               nb_of_qoi=nb_of_qoi
              )
