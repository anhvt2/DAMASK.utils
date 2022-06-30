import numpy as np
import os,math

# Note from Scott:
    # 1. Warp20C - Warpage at 20C
    # 2. Warp250C - Warpage at 250C
    # 3. BGA_SEDperCycle - damage metric for BGA lifetime prediction
    # 4. C4_SEDperCycle - damage metric for C4 interconnection lifetime prediction
    # 5. C4_s1 - first principal stress at C4 corner, where cracking/delamination frequently occurs

if os.path.isfile('Results.txt'):
	outResults = np.loadtxt('Results.txt',dtype=str)
	if not '*' in outResults[0]:
		# application specific
		objectiveFunction = - float(outResults[2]) # minimize damage metric
		warpageAt20C = float(outResults[0])
		warpageAt250C = float(outResults[1])
	else:
		objectiveFunction = 0
else:
	objectiveFunction = 0 

# write outputs

o = open('output.dat', 'w') # can be 'r', 'w', 'a', 'r+'
o.write('%0.8f\n' % objectiveFunction)
o.close()

f = open('feasible.dat', 'w')
if math.isnan(objectiveFunction): # if NaN then not feasible
	f.write('0\n')
elif not isinstance(objectiveFunction, float): # if not a float then not feasible
	f.write('0\n')
elif objectiveFunction == 0:
        f.write('0\n')
# else: # if no error is detected
elif warpageAt20C < 300 and warpageAt250C < 75: # if no error is detected AND constraint satisfied
	# if feasible, then dump rewards.dat
	f.write('1\n')
	r = open('rewards.dat', 'w')
	r.write('%0.8f\n' % objectiveFunction)
	r.close()
else: # any other cases
	f.write('0\n')

f.close()




