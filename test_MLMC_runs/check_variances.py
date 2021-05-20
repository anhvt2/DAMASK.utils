import time
import subprocess
import numpy as np
import os
os.system('rm -fv log.check_variances')

def sample_cpfem(level):
    cmd = ["python3", "wrapper_DREAM3D-DAMASK.py", "--level=" + str(level)]
    out = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8") # execute command and read output
    lines = out.splitlines() # split output into lines
    Q = [] # empty array
    for line in lines: # loop over all lines
        if "Estimated Yield Stress" in line: # if line containes "Estimated Yield Stress"
            try:
                yield_stress = float(line.split()[-2]) # parse the yield stress value to Float64
                Q.append(yield_stress) # append the value of the yield stress to the array "Q"
            except:
                Q.append(None)
    print(Q)
    if len(Q) == 2: # safeguard
        if level == 0:
            return Q[0], Q[0] # return twice Qf
        else:
            return Q[0] - Q[1], Q[0] # return multilevel difference and Qf
    else:
        return None, None # safeguard

def check_variances(max_level=3, budget=3600*24*2):
    buget_per_level = budget/(max_level + 1)

    # for level in range(max_level):
    for level in range(max_level - 1, -1, -1): # reverse order
        samps_dQ = []
        samps_Qf = []
        timer = 0
        while timer < buget_per_level:
            t = time.time()
            dQ, Qf = sample_cpfem(level)
            if dQ != None and Qf != None: # if both dQ and Qf are not both None
                timer += time.time() - t
                samps_dQ.append(dQ)
                samps_Qf.append(Qf)
                print("level =", level,
                        ", E =", np.mean(samps_Qf),
                        ", dE =", np.mean(samps_dQ),
                        ", V =", np.var(samps_Qf),
                        ", dV =", np.var(samps_dQ),
                        ",", len(samps_dQ), "samples, time per sample is", timer/len(samps_dQ))
                logFile = open('log.check_variances', 'a')
                logFile.write("level = %d, E = %.8f, dE = %.8f, V = %.8f, dV = %.8f, len(samps_dQ) = %d samples, time per sample is %.8f\n" % (level, np.mean(samps_Qf), np.mean(samps_dQ), np.var(samps_Qf), np.var(samps_dQ), len(samps_dQ), timer/len(samps_dQ)))


if __name__ == "__main__":
    check_variances()
