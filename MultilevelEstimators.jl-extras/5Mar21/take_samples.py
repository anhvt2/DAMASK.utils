import numpy as np
import os
import time

os.chdir("samples")

levels = os.listdir(".")
levels.sort(key=int)
nlevels = len(levels) - 1

# loop over all levels
for level_dir in levels:
    level = int(level_dir)
    os.chdir(level_dir)

    print("checking level", level_dir + "/" + str(nlevels) + "...")

    samples = os.listdir(".")
    samples.sort(key=int)
    nsamples = len(samples)

    # loop over all sample numbers
    for sample in samples:
        os.chdir(sample)

        print("checking sample", sample + "/" + str(nsamples) + "... ", end="")

        # read random parameters
        Ed = np.loadtxt("params.dat")

        # grid sizes
        he_fine = 0.05/2**level
        he_coarse = he_fine / 2
        
        # check if Qf.dat exists
        if not os.path.isfile("Qf.dat") or not os.path.isfile("dQ.dat") or not os.path.isfile("time.dat"):
            # start timer
            t_start = time.time()

            # compute next "fine" sample
            print("computing Qf... ", end="")
            out = os.popen("python3 ../../../Sandwich.py --elem " + str(he_fine) + " --young " + str(Ed[0]) + " " + str(Ed[1])).read()
            Qf = float(out.split()[-1])

            # compute next "coarse" sample
            if level > 0: # only do coarse simulation if level > 0
                print("computing Qc... ", end="")

                out = os.popen("python3 ../../../Sandwich.py --elem " + str(he_coarse) + " --young " + str(Ed[0]) + " " + str(Ed[1])).read()
                Qc = float(out.split()[-1])
            else:
                Qc = 0

            # update timer
            elapsed = time.time() - t_start
            print("finished in {:5.2f} seconds!".format(elapsed), end="")

            # write Qf.dat
            with open("Qf.dat", "w") as f:
                f.write(str(Qf) + "\n")

            # write dQ.dat
            with open("dQ.dat", "w") as f:
                f.write(str(Qf - Qc) + "\n")

            # write time.dat
            with open("time.dat", "w") as f:
                f.write(str(elapsed) + "\n")

        else:

            print("Qf.dat found! ", end="")
            print("dQ.dat found! ", end="")
            print("time.dat found! ", end="")

        print(" => all done!")

        os.chdir("..")
    
    os.chdir("..")
