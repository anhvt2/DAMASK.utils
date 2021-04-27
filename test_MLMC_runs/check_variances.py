import subprocess
import numpy as np

def my_sample_function(level):
    cmd = ["python3", "wrapper_DREAM3D-DAMASK.py", "--level=" + str(level), "--isNewMs='False'"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout # execute command and read output
    lines = out.splitlines() # split output into lines
    Q = [] # empty array
    for line in lines: # loop over all lines
        if "Estimated Yield Stress" in line: # if line containes "Estimated Yield Stress"
            yield_stress = float(line.split[-2]) # parse the yield stress value to Float64
            Q.append(yield_stress) # append the value of the yield stress to the array "Q"
    if level == 0:
        return Q[0], Q[0] # return twice Qf
    else:
        return Q[0] - Q[1], Q[0] # return multilevel difference and Qf

def check_variances(max_level=3, budget=3600):
    buget_per_level = budget/(max_level + 1)

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
