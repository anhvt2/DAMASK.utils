import argparse

import numpy as np
from numpy.matlib import repmat

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection

from mpl_toolkits.axes_grid1 import make_axes_locatable

#=======================================================================================
class SandwichBeam:

    def __init__(self, Lx=1.0, Ly=0.25, he=0.025, E1=200e9, E2=100e9, nu=0.25):
        self.Lx = Lx # beam width
        self.Ly = Ly # beam height
        self.he = he # element size

        self.nelx = int(Lx/he) # number of elements in the x-direction
        self.nely = int(Ly/he) # number of elements in the y-direction
        if not self.nely % 5 == 0:
            raise ValueError("nely = Ly/he must be dividable by 5 for the current \
                    layer compostion (see Ed)")
        nlay = int(self.nely/5) # number of elements in top and bottom layer

        self.E1 = E1 # Young's modulus for material 1
        self.E2 = E2 # Young's modulus for material 2
        
        top = E1 * np.ones((nlay, self.nelx))
        mid = E2 * np.ones((3*nlay, self.nelx))
        bot = E1 * np.ones((nlay, self.nelx))
        self.E = np.reshape(np.vstack((top, mid, bot)), 
                (1, self.nelx*self.nely), order="F") # Young's modulus
        self.nu = nu * np.ones((1, self.nelx*self.nely)) # Poisson ratio

        self.ndof = 2*(self.nely + 1)*(self.nelx + 1) # number of degrees of freedom
        self.alldofs = range(self.ndof) # all degrees of freedom
        self.fixeddofs = range(2*(self.nely + 1)) # fixed degrees of freedom
        self.freedofs = np.setdiff1d(self.alldofs, self.fixeddofs)

        # for distributed load at top
        #Fdof = range(1, 2*(self.nely + 1)*self.nelx + 2, 2*(self.nely + 1))
        #Fval = -1e7/self.nelx * np.hstack((1/2, np.ones(self.nelx - 1), 1/2)) 

        # for point load at right side
        Fdof = np.array([2*(self.nelx*(self.nely+1)+i)-1 for i in range(1,self.nely+2)])
        Fval = -1e7/self.nely * np.hstack((1/2, np.ones(self.nely - 1), 1/2))

        self.F = np.zeros(self.ndof)
        self.F[Fdof] = Fval

    def compute_displacement(self):

        # matrices with degrees of freedom and nodes for each element
        nodenrs = np.reshape(range(1, (self.nelx + 1)*(self.nely + 1) + 1), \
                (self.nely + 1, self.nelx + 1), order="F")
        edofVec = np.reshape(2*nodenrs[:-1,:-1] + 1, (self.nelx*self.nely, 1), \
                order="F")
        self.edofMat = repmat(edofVec, 1, 8) + repmat(np.hstack((0, 1, 2*self.nely + \
                np.array([2, 3, 0, 1]), -2, -1)), self.nelx*self.nely, 1) - 1
        enodeVec = np.reshape(nodenrs[1:, :-1], (self.nelx*self.nely, 1), order="F") 
        self.enodeMat = repmat(enodeVec, 1, 4) + repmat(np.hstack((0, \
                self.nely + np.array([1, 0]), -1)), self.nelx*self.nely, 1) - 1

        # number of (neighbour) elements for each node
        neighb = 4*np.ones((self.nelx + 1, self.nely + 1))
        neighb[0, :] /= 2
        neighb[-1, :] /= 2
        neighb[:, 0] /= 2
        neighb[:, -1] /= 2
        neighb = np.reshape(neighb, (neighb.size, 1))
        self.nele = neighb[self.enodeMat]

        # compose stiffness matrix
        iK = np.reshape(np.kron(self.edofMat, np.ones((8, 1), dtype=int)), \
                (64*self.nelx*self.nely, 1)).flatten()
        jK = np.reshape(np.kron(self.edofMat, np.ones((1, 8), dtype=int)), \
                (64*self.nelx*self.nely, 1)).flatten()

        A11 = np.array([[12,  3, -6, -3], [ 3, 12,  3,  0], [-6,  3, 12, -3], \
                [-3,  0, -3, 12]])
        A12 = np.array([[-6, -3,  0,  3], [-3, -6, -3, -6], [ 0, -3, -6,  3], \
                [ 3, -6,  3, -6]])
        B11 = np.array([[-4,  3, -2,  9], [ 3, -4, -9,  4], [-2, -9, -4, -3], \
                [ 9,  4, -3, -4]])
        B12 = np.array([[ 2, -3,  4, -9], [-3,  2,  9, -2], [ 4,  9,  2,  3], \
                [-9, -2,  3,  2]])
        A = np.atleast_2d(np.vstack((np.hstack((A11, A12)), \
                np.hstack((np.transpose(A12), A11)))).flatten("F")).transpose()
        B = np.atleast_2d(np.vstack((np.hstack((B11, B12)), \
                np.hstack((np.transpose(B12), B11)))).flatten("F")).transpose()

        sKe = np.matmul(A, self.E/(1 - self.nu**2)/24) + np.matmul(B, \
                (self.nu*self.E)/(1 - self.nu**2)/24)
        sK = np.reshape(sKe, (64*self.nelx*self.nely, 1), order="F").flatten()
        K = csr_matrix((sK, (iK, jK)))

        Kc = K[self.freedofs, :][:, self.freedofs]
        Fc = self.F[self.freedofs]

        # solve sparse system
        self.u = np.zeros((self.ndof, 1))
        self.u[self.freedofs, 0] = spsolve(Kc, Fc)

    def compute_stress(self):
        # check if displacements have been computed
        if not hasattr(self, "u"):
            raise ValueError("displacements have not yet been computed!")

        # B-matrices
        dN1x12 = -1/self.he; dN2x12 = 1/self.he; dN3x12 = 0; dN4x12 = 0;
        dN1x34 = 0; dN2x34 = 0; dN3x34 = 1/self.he; dN4x34 = -1/self.he;
        dN1y14 = -1/self.he; dN2y14 = 0; dN3y14 = 0; dN4y14 = 1/self.he;
        dN1y23 = 0; dN2y23 = -1/self.he; dN3y23 = 1/self.he; dN4y23 = 0;

        B1 = np.array([[dN1x12,      0, dN2x12,      0, dN3x12,      0, dN4x12,      0],
                       [     0, dN1y14,      0, dN2y14,      0, dN3y14,      0, dN4y14],
                       [dN1y14, dN1x12, dN2y14, dN2x12, dN3y14, dN3x12, dN4y14, dN4x12]]
                     )
        B2 = np.array([[dN1x12,      0, dN2x12,      0, dN3x12,      0, dN4x12,      0],
                       [     0, dN1y23,      0, dN2y23,      0, dN3y23,      0, dN4y23],
                       [dN1y23, dN1x12, dN2y23, dN2x12, dN3y23, dN3x12, dN4y23, dN4x12]]
                     )
        B3 = np.array([[dN1x34,      0, dN2x34,      0, dN3x34,      0, dN4x34,      0],
                       [     0, dN1y23,      0, dN2y23,      0, dN3y23,      0, dN4y23],
                       [dN1y23, dN1x34, dN2y23, dN2x34, dN3y23, dN3x34, dN4y23, dN4x34]]
                     )
        B4 = np.array([[dN1x34,      0, dN2x34,      0, dN3x34,      0, dN4x34,      0],
                       [     0, dN1y14,      0, dN2y14,      0, dN3y14,      0, dN4y14],
                       [dN1y14, dN1x34, dN2y14, dN2x34, dN3y14, dN3x34, dN4y14, dN4x34]]
                     )

        # stresses
        ue = np.reshape(self.u[self.edofMat.transpose(), 0], (8, self.nelx*self.nely), \
                order="F")
        sMat = np.zeros((3,(self.nelx+1)*(self.nely+1)));
        for inde in range(self.nelx*self.nely):
            D = self.E[0, inde]/(1-self.nu[0, inde]**2)*np.array( \
                    [[               1, self.nu[0, inde],                        0], \
                     [self.nu[0, inde],                1,                        0], \
                     [               0,                0, (1 - self.nu[0, inde])/2]]
                )
            DB1 = np.reshape(np.matmul(D, np.matmul(B1, ue[:, inde])), (3, 1))
            DB2 = np.reshape(np.matmul(D, np.matmul(B2, ue[:, inde])), (3, 1))
            DB3 = np.reshape(np.matmul(D, np.matmul(B3, ue[:, inde])), (3, 1))
            DB4 = np.reshape(np.matmul(D, np.matmul(B4, ue[:, inde])), (3, 1))

            sMat[:, self.enodeMat[inde,:]] += 1 / repmat(self.nele[inde,:].transpose(),\
                    3, 1) * np.hstack((DB1, DB2, DB3, DB4))

        self.stress = np.reshape(sMat, (sMat.size, 1), order="F")

    def plot_displacement(self):
        # check if displacements have been computed
        if not hasattr(self, "u"):
            raise ValueError("displacements have not yet been computed!")

        # compute displacement scale factor
        scale = max(self.he*self.nelx,self.he*self.nely)/(10*max(abs(self.u)))

        # select displacement in x and y
        u_x = scale*np.reshape(self.u[::2], (self.nelx + 1, self.nely + 1))
        u_y = scale*np.reshape(self.u[1::2], (self.nelx + 1, self.nely + 1))

        # define grid points
        xs = np.linspace(0, self.Lx, self.nelx+1)
        ys = np.linspace(self.Ly, 0, self.nely+1)

        # create lists of patches for both materials
        inner_patches = []
        outer_patches = []

        inner_patches.append(Rectangle((0, 0), self.Lx, self.Ly)) # add outline

        # add patch for each element
        for (i, _) in enumerate(xs[:-1]):
            for (j, _) in enumerate(ys[:-1]):
                a = np.array([ xs[i]   + u_x[i,   j  ], ys[j]   + u_y[i,   j  ]])
                b = np.array([ xs[i+1] + u_x[i+1, j  ], ys[j]   + u_y[i+1, j  ]])
                c = np.array([ xs[i+1] + u_x[i+1, j+1], ys[j+1] + u_y[i+1, j+1]])
                d = np.array([ xs[i]   + u_x[i,   j+1], ys[j+1] + u_y[i,   j+1]])
                patch = Polygon(np.vstack((a, b, c, d)))
                if j < self.nely/5 or j >= 4*self.nely/5: # if outer layer
                    outer_patches.append(patch)
                else: # if inner layer
                    inner_patches.append(patch)

        # plot!
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        ax.add_collection(PatchCollection(inner_patches, edgecolor=(0, 0, 0, 1), \
                facecolor=(1, 1, 1, 0)))
        ax.add_collection(PatchCollection(outer_patches, edgecolor=(0, 0, 0, 1), \
                facecolor=(.7, .7, .7, .5)))
        
        plt.axis('equal')
        plt.axis('off')
        plt.show()

    def plot_stress(self, ind):
        # check if stresses have been computed
        if not hasattr(self, "stress"):
            raise ValueError("stresses have not yet been computed!")

        si = self.stress[ind::3]
        smax = max(abs(self.stress))
        siM = np.reshape(si, (self.nely + 1, self.nelx + 1), order="F")
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        im = ax.imshow(siM, extent=[0, self.Lx, 0, self.Ly], interpolation="spline16", \
                vmin=-smax/2, vmax=smax/2)
        ax.set(xlim=(0, self.Lx), ylim=(0, self.Ly))
        plt.title("sigma_xx" if ind == 0 else "sigma_yy" if ind == 1 else "sigma_xy")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.5)
        plt.colorbar(im, cax=cax)
        plt.show()

#=======================================================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--elem", dest="he", type=float, default=0.025, \
            help="element size")
    parser.add_argument("-y", "--young", dest="E", type=float, nargs=2, \
            default=[200e9, 100e9], help="Young's modulus of material 1 and 2")
    parser.add_argument("-s", "--stress", dest="stress", action="store_true", \
            help="compute stress inside beam")
    parser.add_argument("-p", "--plot", dest="plot", action="store_true", \
            help="show plots of displacement and stress")

    args = parser.parse_args()

    # make a sandwich beam
    sw = SandwichBeam(he=args.he, E1=args.E[0], E2=args.E[1])

    # compute displacement and stresses
    sw.compute_displacement()
    if args.stress:
        sw.compute_stress()

    # QoI = displacement at bottom right corner
    print("displacement of bottom right corner is", sw.u[-1][0])
    
    # plot
    if args.plot:
        sw.plot_displacement()
        if args.stress:
            sw.plot_stress(0)
            sw.plot_stress(1)
            sw.plot_stress(2)

#=======================================================================================
if __name__== "__main__":
    main()
