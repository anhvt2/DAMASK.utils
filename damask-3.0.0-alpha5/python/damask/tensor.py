# Copyright 2011-2021 Max-Planck-Institut für Eisenforschung GmbH
# 
# DAMASK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
Tensor mathematics.

All routines operate on numpy.ndarrays of shape (...,3,3).

"""

import numpy as _np


def deviatoric(T):
    """
    Calculate deviatoric part of a tensor.

    Parameters
    ----------
    T : numpy.ndarray of shape (...,3,3)
        Tensor of which the deviatoric part is computed.

    Returns
    -------
    T' : numpy.ndarray of shape (...,3,3)
        Deviatoric part of T.

    """
    return T - spherical(T,tensor=True)


def eigenvalues(T_sym):
    """
    Eigenvalues, i.e. principal components, of a symmetric tensor.

    Parameters
    ----------
    T_sym : numpy.ndarray of shape (...,3,3)
        Symmetric tensor of which the eigenvalues are computed.

    Returns
    -------
    lambda : numpy.ndarray of shape (...,3)
        Eigenvalues of T_sym sorted in ascending order, each repeated
        according to its multiplicity.

    """
    return _np.linalg.eigvalsh(symmetric(T_sym))


def eigenvectors(T_sym,RHS=False):
    """
    Eigenvectors of a symmetric tensor.

    Parameters
    ----------
    T_sym : numpy.ndarray of shape (...,3,3)
        Symmetric tensor of which the eigenvectors are computed.
    RHS: bool, optional
        Enforce right-handed coordinate system. Defaults to False.

    Returns
    -------
    x : numpy.ndarray of shape (...,3,3)
        Eigenvectors of T_sym sorted in ascending order of their
        associated eigenvalues.

    """
    (u,v) = _np.linalg.eigh(symmetric(T_sym))

    if RHS:
        v[_np.linalg.det(v) < 0.0,:,2] *= -1.0
    return v


def spherical(T,tensor=True):
    """
    Calculate spherical part of a tensor.

    Parameters
    ----------
    T : numpy.ndarray of shape (...,3,3)
        Tensor of which the spherical part is computed.
    tensor : bool, optional
        Map spherical part onto identity tensor. Defaults to True.

    Returns
    -------
    p : numpy.ndarray of shape (...,3,3)
        unless tensor == False: shape (...,)
        Spherical part of tensor T. p is an isotropic tensor.

    """
    sph = _np.trace(T,axis2=-2,axis1=-1)/3.0
    return _np.einsum('...jk,...',_np.eye(3),sph) if tensor else sph


def symmetric(T):
    """
    Symmetrize tensor.

    Parameters
    ----------
    T : numpy.ndarray of shape (...,3,3)
        Tensor of which the symmetrized values are computed.

    Returns
    -------
    T_sym : numpy.ndarray of shape (...,3,3)
        Symmetrized tensor T.

    """
    return (T+transpose(T))*0.5


def transpose(T):
    """
    Transpose tensor.

    Parameters
    ----------
    T : numpy.ndarray of shape (...,3,3)
        Tensor of which the transpose is computed.

    Returns
    -------
    T.T : numpy.ndarray of shape (...,3,3)
        Transpose of tensor T.

    """
    return _np.swapaxes(T,axis2=-2,axis1=-1)
