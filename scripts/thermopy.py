# -*- coding: utf-8 -*-
"""
    Created on Wed Jan  7 14:02:28 2015
    
    @author: dvartany
    """

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import scipy.optimize as optimize

from constants import *
# Constants in cgs


def lam(r, q, f):
    return f*q**2*G*M/a*(a/(r-a))**4

def Omega(r):
    return np.sqrt(G*M/r**3)

def ftid(r, Sigma, q, f, off=False):
    if off:
        return 0.0
    else:
        return 0.5*(OmegaIn - Omega(r))*lam(r, q, f)*Sigma

def fv(r, T, Sigma):
    return 1.125 * Omega(r)*alpha*k*T/mu * Sigma

def Tirr(r, q):
    return 0.0 * r

def op(T, r, Sigma, idx):
    if idx == 1:
        kappa = 0.00330119*T**1.5
    elif idx == 2:
        kappa = 1.96231 * 10**8 *(Omega(r) * Sigma *(k/mu)**0.5)**0.0949916/T**3.48404
    elif idx == 3:
        kappa = 0.00144313* T**1.5
    elif idx == 4:
        kappa = 5.01187*10**17/T**5.832
    elif idx == 5:
        kappa = 1.14868 *10**-6 *T**2.129
    elif idx == 6:
        kappa = 1.25893*10**135*(Omega(r) * Sigma*(k/mu)**0.5)**1.312/T**42.324
    elif idx == 7:
        kappa = 9.71628 *10**-16*T**4.0625
    elif idx == 8:
        kappa = 8.51138*10**58*(Omega(r) * Sigma *(k/mu)**0.5)**0.676/T**18.142
    elif idx == 9:
        kappa = 1.01158*10**-14*(Omega(r) * Sigma *(k/mu)**0.5)**0.498*T**3.154
    elif idx == 10:
        kappa = 1.15878*10**-41*(Omega(r) * Sigma * (k/mu)**0.5)**0.382*T**10.381
    elif idx == 11:
        kappa = 1.0617*10**12 *(Omega(r) * Sigma * (k/mu)**0.5)**0.928/T**2.896
    elif idx == 12:
        kappa = 10**-.48
    else:
        raise ValueError("Check your idx input")

    try:
        if kappa < 1.41254e-17*T**3.586 and T < 1.0e4:
            kappa = 1.41254e-17*T**3.586
    except:
        floor = np.logical_and(T < 1.0e4, kappa < 1.41254e-17*T**3.586)
        if np.sum(floor) > 0:
            kappa[floor] = 1.41254e-17*T[floor]**3.586
    return kappa

def func(T, r, Sigma, q, f, kappa):
    return sigma*T**4 - (3*op(T , r, Sigma, kappa)*T**0.5*Sigma*0.0625 + 2/(op(T, r, Sigma,kappa)*Sigma*T**0.5))*(ftid(r,Sigma,q,f) +\
           fv(r,T,Sigma)) - sigma*Tirr(r,q)**4

def getBracket(r, Sigma, idx):
    if idx == 1:
        return 1.0e-3, 144.957* (Omega(r) * Sigma * (k/mu)**0.5)**0.019172
    
    elif idx == 2:
        return 144.957* (Omega(r) * Sigma * (k/mu)**0.5)**0.019172, 171.54*(Omega(r) * Sigma *(k/mu)**0.5)**0.019172
    
    elif idx == 3:
        return 171.54*(Omega(r) * Sigma *(k/mu)**0.5)**0.019172, 617.376
    
    elif idx == 4:
        return 617.376, 931.773
    
    elif idx == 5:
        return 931.773, 1584.42 *(Omega(r) * Sigma * (k/mu)**0.5)**0.027182
    
    elif idx == 6:
        return 1584.42 *(Omega(r) * Sigma * (k/mu)**0.5)**0.027182, 1719.07 * (Omega(r) * Sigma * (k/mu)**0.5)**0.0283976
    
    elif idx == 7:
        return 1719.07 * (Omega(r) * Sigma * (k/mu)**0.5)**0.0283976, 2137.71 * (Omega(r) * Sigma * (k/mu)**0.5)**0.0304569
    
    elif idx == 8:
        return 2137.71 * (Omega(r) * Sigma * (k/mu)**0.5)**0.0304569, 2656.1 * (Omega(r) * Sigma*(k/mu)**0.5)**0.00835476
    
    elif idx == 9:
        return 2656.1 * (Omega(r) * Sigma*(k/mu)**0.5)**0.00835476, 5345.15 * (Omega(r) * Sigma * (k/mu)**0.5)**0.0151134
    
    elif idx == 10:
        return 5345.15 * (Omega(r) * Sigma * (k/mu)**0.5)**0.0151134, 9767.78 * (Omega(r) * Sigma * (k/mu)**0.5)**0.0408163
    
    elif idx == 11:
        return 9767.78 * (Omega(r) * Sigma * (k/mu)**0.5)**0.0408163, 19529.8 *(Omega(r) * Sigma * (k/mu)**0.5)**0.325581
    
    elif idx == 12:
        return 19529.8 *(Omega(r) * Sigma * (k/mu)**0.5)**0.325581, 5.0e11
    else:
        raise ValueError("Opacity index out of range")

def Tfin(r, Sigma, q, f, idx, delta=0.0):
    Tmin, Tmax = getBracket(r, Sigma, idx)
    try:
        T = brentq(func, (1.0-delta)*Tmin, (1.0+delta)*Tmax, args=(r,Sigma,q,f,idx), maxiter=200)
    except ValueError, e:
        return Tfin(r, Sigma, q, f, idx+1, delta=delta)
    else:
        if (1.0-delta)*Tmin <= T <= (1.0+delta)*Tmax:
            return idx, T
        else:
            return Tfin(r, Sigma, q, f, idx+1, delta=delta)

def buildTempTable(rGrid, q=1.0, f=0.002, Sigmin=1.0e-5, Sigmax=1.0e6, Sigres=2000, delta=0.1,
                   rmStripe=False, smoothT=False, sigmaSigma=10, sigmaR=2, **kargs):
    """
        Return a table of precomputed temperatures as a function of radius and surface density.
        Arguments:
        rGrid: Grid of radii to compute the temperatures for, it has to be in cgs units.
        Keywords:
        q: Ratio of the masses of the stars in the binary, between 0 and 1.
        f: Fudge factor used to tune the magnitude of the tidal torque.
        Sigmin: Minimum value for the surface density to compute temperatures for.
        Sigmax: Maximum value for the surface density to compute temperatures for.
        Sigres: Number of points to use for surface density in the table, it will be logarthmically spaced
        in surface density.
        """
    
    SigmaGrid = np.logspace(np.log10(Sigmin), np.log10(Sigmax), Sigres)
    temp = np.zeros((len(rGrid), Sigres)) #create m x n array for temp
    idxs = np.zeros((Sigres, len(rGrid)))
    for i, r in enumerate(rGrid):
        for j, Sigma in enumerate(SigmaGrid):
            try:
                idxs[-j-1,i], temp[i,j] = Tfin(r, Sigma, q, f, 1)
            except ValueError, e:
                try:
                    idxs[-j-1,i], temp[i,j] = Tfin(r, Sigma, q, f, 1, delta=0.07)
                except ValueError, e:
                    try:
                        idxs[-j-1,i], temp[i,j] = Tfin(r, Sigma, q, f, 1, delta=0.2)
                    except ValueError, e:
                        raise
    if rmStripe:
        for i, j in zip(np.where(idxs==5)[0],np.where(idxs==5)[1]):
            idxs[i,j], temp[j,-i-1] = Tfin(rGrid[j],SigmaGrid[-i -1], q, f, 1, delta=0.07)    
    if smoothT:
        from scipy.ndimage.filters import gaussian_filter
        temp = gaussian_filter(temp, (sigmaR, sigmaSigma))

    # Return values in logspace for interpolation
    return np.log10(rGrid), np.log10(SigmaGrid), np.log10(temp), idxs
