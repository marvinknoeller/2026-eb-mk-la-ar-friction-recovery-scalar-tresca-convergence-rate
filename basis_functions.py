#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 22:54:16 2025

@author: marvinknoller
"""

import ufl
import numpy as np

tol = 1e-14
def create_fun(cos_coeffs, sin_coeffs, x):
    """
    Creates a linear combination of cosine and sine terms so that it can be used in forms.

    Parameters
    ----------
    cos_coeffs : fem.Constant
        The coefficients corresponding to the cosine terms.
    sin_coeffs : fem.Constant
        The coefficients corresponding to the sine terms.
    x : ufl.SpatialCoordinate
        The spatial coordinates corresponding to some mesh.

    Returns
    -------
    fun : ufl function
        The linear combination of cosines and sines on the spatial coordinates 
        according to the arrays in cos_coeffs and sin_coeffs.

    """
    fun = ufl.zero()
    for n in range(cos_coeffs.value.size):
        fun += 2/np.sqrt(np.pi) * cos_coeffs[n]*ufl.cos(n*ufl.atan2(x[1], x[0]))
    for n in range(1, sin_coeffs.value.size+1):
        fun += 2/np.sqrt(np.pi) * sin_coeffs[n-1]*ufl.sin(n*ufl.atan2(x[1], x[0]))
    return fun

def create_fun_interpol(cos_coeffs, sin_coeffs, x):
    """
    Creates a linear combination of cosine and sine terms so that it can be used with python's lambda function.

    Parameters
    ----------
    cos_coeffs : fem.Constant
        The coefficients corresponding to the cosine terms.
    sin_coeffs : fem.Constant
        The coefficients corresponding to the sine terms.
    x : input variable from lambda
        This is the x from lambda x : ...

    Returns
    -------
    fun : ufl function
        The linear combination of cosines and sines on the spatial coordinates 
        according to the arrays in cos_coeffs and sin_coeffs.

    """
    fun = ufl.zero()
    for n in range(cos_coeffs.size):
        fun += 2/np.sqrt(np.pi) * cos_coeffs[n]*np.cos(n*np.atan2(x[1], x[0]))
    for n in range(1, sin_coeffs.size+1):
        fun += 2/np.sqrt(np.pi) * sin_coeffs[n-1]*np.sin(n*np.atan2(x[1], x[0]))
    return fun


def create_fun_for_plot(cos_coeffs, sin_coeffs, nn=1000):
    """
    Creates a linear combination of cosine and sine terms just for easy plotting with matplotlib

    Parameters
    ----------
    cos_coeffs : np.ndarray
        The coefficients corresponding to the cosine terms.
    sin_coeffs : np.ndarray
        The coefficients corresponding to the sine terms.

    Returns
    -------
    xx : np.ndarray
        x values from 0 to 2*np.pi
    vals : TYPE
        fun(x), where fun is the combination of cosines and sines
        according to the arrays in cos_coeffs and sin_coeffs.

    """
    xx = np.linspace(0,2*np.pi,nn)
    vals = np.zeros(nn)
    
    for nn in range(cos_coeffs.size):
        vals += 2/np.sqrt(np.pi)*cos_coeffs[nn] * np.cos(nn*xx)
        
    for nn in range(1, sin_coeffs.size+1):
        vals += 2/np.sqrt(np.pi)*sin_coeffs[nn-1] * np.sin(nn*xx)
        
    return xx,vals


