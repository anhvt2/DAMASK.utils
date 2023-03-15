###############################################################################
#                                                                             #
#                           run_multiindex.jl                                 #
#                                                                             #
###############################################################################

include("utils.jl") # load utils

# ========================
# @@@ SELECT INDEX SET @@@
# ========================
index_set = AD(2) # adaptive index set in 2 dimensions

# >> Specify the maximum search space for the adaptive algorithm
#    The settings below generate an index set as follows:
#    julia> print(FT(1, (2-1)/(5-1)), 4)
#       ◼ ◼ ◼ ◼ ◼ 
#       ◼ ◼ ◼ ◼ ◼ 
max_search_space = FT(1, (2-1)/(5-1))

# ========================
# @@@ SELECT MAX LEVEL @@@
# ========================
# >> Specify the maximum level parameter
max_level = 4 # See note below

# @@@ NOTE @@@
# !!! For AD, the `max_level` parameter determines the size of the search space (see above)

# =========================
# @@@ SELECT COST MODEL @@@
# =========================
# >> Specify cost per index (e.g. in seconds)
# >> For example, if we have:
# +--------+----------------+
# | level  | cost (seconds) |
# +--------+----------------+
# | (0, 0) | 1              |
# | (1, 0) | 2              |
# | (2, 0) | 4              |
# | (3, 0) | 8              |
# | (4, 0) | 16             |
# | (0, 1) | 11             |
# | (1, 1) | 12             |
# | (2, 1) | 14             |
# | (3, 1) | 18             |
# | (4, 1) | 26             |
# +--------+----------------+
# Then, these values can be specified as:
# cost_per_level = [1 11; 2 12; 4 14; 8 18; 16 26]

# +--------+----------------+
# | level  | cost (seconds) |
# +--------+----------------+
# | (0, 0) | 0.49           |
# | (1, 0) | 0.67           |
# | (2, 0) | 1.20           |
# | (3, 0) | 2.20           |
# | (4, 0) | 8.46           |
# | (0, 1) | 1.55           |
# | (1, 1) | 3.01           |
# | (2, 1) | 3.62           |
# | (3, 1) | 7.00           |
# | (4, 1) | 26.43          |
# | extra  |                |
# | (5,0)  | 55.04          |
# | (5,1)  | 273.89         |
# +--------+----------------+
cost_per_level = [0.49 1.55; 0.67 3.01; 1.20 3.62; 2.20 7.00; 8.46 26.43]


# =============================
# @@@ SELECT RMSE TOLERANCE @@@
# =============================
# >> Specify initial RMSE (this is just a guess)
# >> Lower values of the RMSE means a more accurate result, but also more samples
ε = 1e1

# ========================================
# @@@ SELECT NUMBER OF WARM-UP SAMPLES @@@
# ========================================
# >> Specify number of warm-up samples on levels 0, 1, and 2
# >> This value should not be changed
nb_of_warm_up_samples = 10

# =============================
# @@@ SPECIFY NUMBER OF QOI @@@
# =============================
nb_of_qoi = 100

# ======================
# @@@ RUN MULTILEVEL @@@
# ======================
run_multiindex(index_set=index_set,
               max_search_space=max_search_space,
               max_level=max_level,
               cost_model=index -> sum(cost_per_level[key + one(key)] for key in push!(collect(keys(diff(index))), index)),
               ε=ε,
               nb_of_warm_up_samples=nb_of_warm_up_samples,
               nb_of_qoi=nb_of_qoi
              )
