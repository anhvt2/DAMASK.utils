
import numpy as np
import argparse

cost_per_level = [39,    365,    1955,    3305,    12487] # taken from run_multilevel_multiple_qoi.py
num_level = len(cost_per_level)


parser = argparse.ArgumentParser()
parser.add_argument("-ms", "--max_sample", type=int) # desired number of samples at the highest level of fidelity



args = parser.parse_args()
max_sample = int(args.max_sample)


for i in range(num_level):
	print(f"Need {int( max_sample * cost_per_level[-1] / cost_per_level[i])} samples at level {i}")

