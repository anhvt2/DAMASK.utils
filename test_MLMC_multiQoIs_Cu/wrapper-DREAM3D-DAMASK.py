import argparse
import numpy as np

# =========================================================
# @@@ THIS IS A DUMMY WRAPPER - IT DOES NOT DO ANYTHING @@@
# =========================================================

# example usage
# =============
#
# $ python wrapper-DREAM3D-DAMASK.py --level 0
# Estimated Young modulus at 0 is -2.921755914475476
#
# $ python wrapper-DREAM3D-DAMASK.py --level 1
# Estimated Young modulus at 1 is 1.4565607955675095
# Estimated Young modulus at 0 is -2.4967721807515217
#
# $ python wrapper-DREAM3D-DAMASK.py --level 3 --nb_of_qoi 4
# Estimated Young modulus at 3 is 0.2761242362928689, -0.9839518678051921, -0.1033699430980116, 0.13297340048334058
# Estimated Young modulus at 2 is 0.8931224201628596, -0.42128388915439724, 0.3768735421579537, 0.24507689336645652
#
# $ python wrapper-DREAM3D-DAMASK.py --index (2, 1) --nb_of_qoi 3
# Estimated Young modulus at (2, 1) is -2.5457461714149145, -0.02181018669814895, -2.3552475764029435
# Estimated Young modulus at (1, 1) is 0.501231535869913, 0.5191660280513454, -2.1937281246076665
# Estimated Young modulus at (2, 0) is -0.10207593026854639, 1.547139317467864, -0.9352563652562738
# Estimated Young modulus at (1, 0) is -0.5373920568687983, -0.31172737038566456, 0.24969705569949627

# ===================================================================
def main():

    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default=None, type=str)
    parser.add_argument("--index", default=None, type=str)
    parser.add_argument("--nb_of_qoi", default=1, type=int)
    args = parser.parse_args()

    sample = lambda: ", ".join([str(x) for x in np.random.randn(args.nb_of_qoi)]) if args.nb_of_qoi > 1 else np.random.randn()

    # multilevel
    if not args.level is None:
        level = int(args.level)
        print(f"Estimated Young modulus at {level} is {sample()}")
        if level > 0:
            print(f"Estimated Young modulus at {level - 1} is {sample()}")

    # 2d multi-index
    if not args.index is None: # works for 2d only
        index = list(map(int, args.index[1:-1].split(", ")))
        print(f"Estimated Young modulus at ({index[0]}, {index[1]}) is {sample()}")
        if index[0] > 0:
            print(f"Estimated Young modulus at ({index[0] - 1}, {index[1]}) is {sample()}")
        if index[1] > 0:
            print(f"Estimated Young modulus at ({index[0]}, {index[1] - 1}) is {sample()}")
        if index[0] > 0 and index[1] > 0:
            print(f"Estimated Young modulus at ({index[0] - 1}, {index[1] - 1}) is {sample()}")

# ===================================================================
if __name__ == "__main__":
    main()
