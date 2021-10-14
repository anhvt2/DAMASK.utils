###############################################################################
#                                                                             #
#                       run_check_variances.jl                                #
#                                                                             #
###############################################################################

include("check_variances.jl") # load check_variances

# ========================
# @@@ SELECT INDEX SET @@@
# ========================
# >>> Select one of three options below:

# index_set = ML() # OPTION 1: multilevel

# index_set = TD(3) # OPTION 2: multi-index, total degree
index_set = TD(1, (2-1)/(5-1)) # weighted index set
# index_set = TD(1, 2/4) # weighted index set
# index_set = AD(2) # based on discussion with Pieterjan Robbe
# index_set = FT(3) # OPTION 3: multi-index, full tensor

# ========================
# @@@ SELECT MAX LEVEL @@@
# ========================
# >>> Select maximum level parameter:

max_level = 4 # See note below

# @@@ NOTE @@@
# !!! In a multilevel setting, `max_level=3` means 4 levels in total (i.e.,
# !!! 0, 1, 2, and 3). In a multi-index setting, `max_level=3` determines the
# !!! size of the index set. For example, for a 3-dimensional total degree index
# !!! set, there are 20 indices:
# !!! (0, 0, 0)
# !!! (1, 0, 0)
# !!! (2, 0, 0)
# !!! (3, 0, 0)
# !!! (0, 1, 0)
# !!! (1, 1, 0)
# !!! (2, 1, 0)
# !!! (0, 2, 0)
# !!! (1, 2, 0)
# !!! (0, 3, 0)
# !!! (0, 0, 1)
# !!! (1, 0, 1)
# !!! (2, 0, 1)
# !!! (0, 1, 1)
# !!! (1, 1, 1)
# !!! (0, 2, 1)
# !!! (0, 0, 2)
# !!! (1, 0, 2)
# !!! (0, 1, 2)
# !!! (0, 0, 3)

# =============================
# @@@ SELECT TOTAL RUN TIME @@@
# =============================
# >>> Specify total run time (in seconds):

budget = 4*24*3600 # this means 20 seconds, should be much larger in your example

# @@@ NOTE @@@
# !!! You can use for example the following specifications:
# !!! budget = 3600 # 1 hour
# !!! budget = 24*3600 # 1 day
# !!! budget = 7*24*3600 # 1 week

# ===========================
# @@@ RUN CHECK VARIANCES @@@
# ===========================
# >>> Performs the analysis
check_variances(index_set=index_set, max_level=max_level, budget=budget)
