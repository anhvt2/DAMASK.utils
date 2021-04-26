import subprocess
import numpy as np

def my_sample_function(level):
	cmd = ["python3", "wrapper_DREAM3D-DAMASK.py", "--level=" + str(level), "--isNewMs='False'"]
	out = subprocess.run(cmd, capture_output=True, text=True).stdout
	# assuming here that "Estimated Yield Stress" appears twice in the output,
	# once for Qf and once for Qc,
	# except for level 0, which generates only 1 output value for the yield stress
	lines = out.splitlines()
	Q = []
	for line in lines:
		if "Estimated Yield Stress" in line:
			Q.append(float(line.split[-2]))
			if level == 0:
				return Q[0], Q[0]
			else:
				return Q[0] - Q[1], Q[0]

def check_variances(max_level=3, budget=3600):
	buget_per_level = budget / (max_level + 1)
	for level in range(max_level):
		samps_dQ = []
		samps_Qf = []
		timer = 0
		while timer < buget_per_level:
			t = time.time()
			dQ, Qf = my_sample_function(level)
			timer += time.time() - t
			samps_dQ.append(dQ)
			samps_Qf.append(Qf)
			print("level =", level,
				", E =", np.mean(samps_Qf),
				", dE =", np.mean(samps_dQ),
				", V =", np.var(samps_Qf),
				", dV =", np.var(samps_dQ),
				",", len(samps_dQ), "samples, time per sample is", timer/len(samps_dQ))

if __name__ == "__main__":
	check_variances()
