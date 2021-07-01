###############################################################################
#                                                                             #
#                        wrapper_DREAM3D-DAMASK.py                            #
#                                                                             #
###############################################################################

# This is a dummy file to test my implementation of check_variances.jl
# It shows how you should implement multilevel and multi-index sampling for 
# your example. 
#
# @@@ WARNING @@@
# !!! You should replace this file with your own wrapper.
#
# @@@ WARNING @@@
# !!! I've added an additional keyword argument "index".
# !!! You MUST implement this keyword in your own wrapper for multi-index
# !!! sampling, the Julia code will NOT work otherwise
#
# @@@ WARNING @@@
# !!! You MUST include the level/index when printing the yield stress.
# !!! The Julia code will NOT work otherwise
# !!! See the example in this dummy wrapper for instructions on how to do this.

import argparse
import itertools
import numpy as np

# =============================================================================
def main():

    # parse arguments
    options = parse_arguments()
    chars = options.index.replace("(", "").replace(")", "").split(",")
    index = np.array([int(char) for char in chars])

    # @@@ WARNING @@@
    # !!! Do some preprocessing here that is shared for all subsequent solves.
    # !!! No preprocessing is required in this dummy example.

    # generate all possible combinations of levels or indices
    for idx in itertools.product([0, 1], repeat=len(index)):
        new_index = index - np.array(idx)
        if all(new_index > -1): # only generate coarse if index > 0
            solve(new_index)

    # @@@ WARNING @@@
    # !!! You will have to implement something similar on your wrapper
    # !!! I imagine the structure of the wrapper will look like this:
    # !!!     (1) do some preprocessing (see previous warning)
    # !!!     (2) loop over all combinations of levels or indices
    # !!!     (3) for each level/index and do the following:
    # !!!     (4)     generate a jobscript
    # !!!     (5)     submit the batchscript
    # !!!     (6)     wait for the result to finish
    # !!!     (7)     print the estimated yield stress 

# =============================================================================
def solve(index):
    idx = ", ".join(str(i) for i in index)
    if len(idx) > 1:
        idx = "(" + idx + ")"
    # @@@ WARNING @@@
    # !!! Returns dummy values for testing purposes.
    # !!! In your code, this should actually compute the solution at the given
    # !!! level or index.
    print("Estimated Yield Stress at", idx, "is", np.random.rand(), "GPa")

# =============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", type=str, default=None)
    return parser.parse_args()

# =============================================================================
if __name__ == "__main__":
    main()
