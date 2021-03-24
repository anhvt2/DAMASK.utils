import numpy as np
import os
import time

os.chdir("samples")

indices = os.listdir(".")
indices.sort()

# loop over all indices
for index_dir in indices:
    index = [int(i) for i in index_dir.split("_")] 
    os.chdir(index_dir)

    print("checking index", index_dir + "...")

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
        hex_fine = 0.05/2**index[0]
        hex_coarse = hex_fine/2
        hey_fine = 0.05/2**index[1]
        hey_coarse = hey_fine/2
        
        # check if Qf.dat exists
        if not os.path.isfile("Qf.dat") or not os.path.isfile("dQ.dat") or not os.path.isfile("time.dat"):
            # start timer
            t_start = time.time()

            # compute next "fine" sample
            print("computing Qff... ", end="")
            out = os.popen("python3 ../../../Sandwich.py --elemx " + str(hex_fine) + " --elemy " + str(hey_fine) + " --young " + str(Ed[0]) + " " + str(Ed[1])).read()
            Qff = float(out.split()[-1])

            # compute "coarse" samples
            if index[0] > 0: # only do coarse x simulation if index[0] > 0
                print("computing Qcf... ", end="")

                out = os.popen("python3 ../../../Sandwich.py --elemx " + str(hex_coarse) + " --elemy " + str(hey_fine) + " --young " + str(Ed[0]) + " " + str(Ed[1])).read()
                Qcf = float(out.split()[-1])
            else:
                Qcf = 0
            if index[1] > 0: # only do coarse y simulation if index[1] > 0
                print("computing Qfc... ", end="")

                out = os.popen("python3 ../../../Sandwich.py --elemx " + str(hex_fine) + " --elemy " + str(hey_coarse) + " --young " + str(Ed[0]) + " " + str(Ed[1])).read()
                Qfc = float(out.split()[-1])
            else:
                Qfc = 0
            if index[0] > 0 and index[1] > 0: # only do coarse x / y simulation if index[0] > 0 and index[1] > 0
                print("computing Qcc... ", end="")

                out = os.popen("python3 ../../../Sandwich.py --elemx " + str(hex_coarse) + " --elemy " + str(hey_coarse) + " --young " + str(Ed[0]) + " " + str(Ed[1])).read()
                Qcc = float(out.split()[-1])
            else:
                Qcc = 0

            # update timer
            elapsed = time.time() - t_start
            print("finished in {:5.2f} seconds!".format(elapsed), end="")

            # write Qf.dat
            with open("Qf.dat", "w") as f:
                f.write(str(Qff) + "\n")

            # write dQ.dat
            with open("dQ.dat", "w") as f:
                f.write(str(Qff - Qcf - Qfc + Qcc) + "\n")

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
