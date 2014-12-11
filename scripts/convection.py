import argparse
import numpy as np
from scipy.optimize import root

from fipy import CylindricalGrid1D, CellVariable, FaceVariable, TransientTerm, ExponentialConvectionTerm

from constants import *

class circumbinary(object):
    def __init__(self, rmax=1.0e2, ncell=100, nstep=100, dt=1.0e-6, delta=1.0e-100,
                 nsweep=10, titer=10, fudge=1.0e-3, q=1.0, gamma=100, mDisk=0.1):
        self.rmax = rmax
        self.ncell = ncell
        self.nstep = nstep
        self.dt = dt
        self.delta = delta
        self.nsweep = nsweep
        self.titer = titer
        self.mDisk = mDisk
        Omega0 = (G*M/(gamma*a)**3)**0.5
        nu0 = alpha*cs**2/Omega0
        self.chi = 2*fudge*q**2*np.sqrt(G*M)/nu0/a*(gamma*a)**1.5
        self.T0 = mu*Omega0/alpha/k*nu0
        self.gamma = gamma
        self.fudge = fudge
        self.q = q
        self.nu0 = nu0
        self.t = 0.0
        self._genGrid()
        self.r = self.mesh.cellCenters.value[0]
        self.rF = self.mesh.faceCenters.value[0]
        self._genSigma()
        self._genTorque()
        self._genT()
        self._genVr()
        self._buildEq()

    def _genGrid(self):
        """Generate a logarithmically spaced grid"""
        logFaces = np.linspace(-np.log(self.gamma), np.log(self.rmax), num=self.ncell+1)
        logFacesLeft = logFaces[:-1]
        logFacesRight = logFaces[1:]
        dr = tuple(np.exp(logFacesRight) - np.exp(logFacesLeft))
        self.mesh = CylindricalGrid1D(dr=dr, origin=(1.0/self.gamma,))

    def _genSigma(self, width=0.1):
        """Create dependent variable Sigma"""
        # Gaussian initial condition
        value = self.mDisk*M/np.sqrt(2*np.pi)/(self.gamma*a*width)*\
                np.exp(-0.5*np.square(self.r-1.0)/width**2)/(2*np.pi*self.gamma*self.r*a)
        value /= M/(self.gamma*a)**2
        idxs = np.where(self.r < 0.1)
        value[idxs] = 0.0
        value = tuple(value)

        # Create the dependent variable and set the boundary conditions
        # to zero
        self.Sigma = CellVariable(name='Surface density',
                                 mesh=self.mesh, hasOld=True, value=value)
        self.Sigma.constrain(0, self.mesh.facesLeft)
        self.Sigma.constrain(0, self.mesh.facesRight)

    def _genTorque(self):
        """Generate Torque"""
        self.Lambda = FaceVariable(name='Torque at cell faces', mesh=self.mesh, rank=1)
        self.LambdaCell = CellVariable(name='Torque at cell centers', mesh=self.mesh)
        LambdaArr = np.zeros(self.rF.shape)
        LambdaArr[1:] = self.chi*np.power(1.0/(self.rF[1:]*self.gamma-1.0), 4)
        self.Lambda.setValue(LambdaArr)
        self.LambdaCell.setValue(self.chi*np.power(1.0/(self.r*self.gamma-1.0), 4))

    def _interpT(self):
        """
        Get an initial guess for T using an interpolation of the solutions for T
        in the various thermodynamic limits.
        """
        Lambda = self.Lambda/self.chi*self.fudge*self.q**2*G*M/a
        LambdaCell = self.LambdaCell/self.chi*self.fudge*self.q**2*G*M/a
        Sigma = self.Sigma*M/(self.gamma*a)**2
        r = self.r*a*self.gamma #In physical units (cgs)
        self.Omega = np.sqrt(G*M/r**3)
        self.TvThin = np.power(9.0/4*alpha*k/sigma/mu/kappa0*self.Omega, 1.0/(3.0+beta))
        self.TtiThin = np.power(1/sigma/kappa0*(OmegaIn-self.Omega)*LambdaCell, 1.0/(4.0+beta))
        self.Ti = np.power(np.square(eta/7*L/4/np.pi/sigma)*k/mu/G/M*r**(-3), 1.0/7)
        self.TvThick = np.power(27.0/64*kappa0*alpha*k/sigma/mu*self.Omega*Sigma**2, 1.0/(3.0-beta))
        self.TtiThick = np.power(3*kappa0/16/sigma*Sigma**2*(OmegaIn-self.Omega)*LambdaCell, 1.0/(4.0-beta))
        #return np.power(self.TvThin**4 + self.TvThick**4 + self.TtiThin**4 + self.TtiThick**4 + self.Ti**4, 1.0/4)/self.T0
        return np.power(self.TvThin**4 + self.TvThick**4 + self.Ti**4, 1.0/4)/self.T0

    def _genT(self):
        """Create a cell variable for temperature"""
        # Initialize T with the interpolation of the various thermodynamic limits
        #r = self.r*a #In physical units (cgs)
        #self.Omega = np.sqrt(G*M/r**3)
        #self.TvThin = np.power(9.0/4*alpha*k/sigma/mu/kappa0*self.Omega, 1.0/(3.0+beta))
        #self.Ti = np.power(np.square(eta/7*L/4/np.pi/sigma)*k/mu/G/M*r**(-3), 1.0/7)
        self.T = self._interpT()

    def _genVr(self):
        """Generate the face variable that stores the velocity values"""
        r = self.r #In dimensionless units (cgs)
        # viscosity at cell centers in cgs
        #nu = alpha*k*self.T/mu/self.Omega
        #self.nu = self.nu0
        self.nu = 1.0
        self.visc = r**0.5*self.nu*self.Sigma
        # I add the delta to avoid divisions by zero
        self.vrVisc = -3/self.rF**(0.5)/(self.Sigma.faceValue + self.delta)*self.visc.faceGrad
        self.vrTid = self.Lambda*np.sqrt(self.rF)

    def _buildEq(self):
        """
        Build the equation to solve, we can change this method to impelement other
        schemes, e.g. Crank-Nicholson.
        """
        # The current scheme is an implicit-upwind
        self.eq = TransientTerm(var=self.Sigma) == - ExponentialConvectionTerm(coeff=self.vrVisc + self.vrTid, var=self.Sigma)

    def singleTimestep(self, dt=None, update=True, emptyDt=False):
        """
        Evolve the system for a single timestep of size `dt`
        """
        if dt:
            self.dt = dt
        if emptyDt:
            vr = self.vrVisc.value[0] + self.vrTid.value[0]
            self.dt = 1.0*np.amin(np.absolute(self.mesh.cellVolumes/(self.rF[:-1]*vr[:-1])))
        try:
            for i in range(self.nsweep):
                res = self.eq.sweep(dt=self.dt)
            if update:
                self.Sigma.updateOld()
            self.t += self.dt
        except FloatingPointError:
            import ipdb; ipdb.set_trace()

    def evolve(self, **kargs):
        """
        Evolve the system according to the values in its initialization
        self.dt, self.nstep, and self.nsweep
        """
        for i in range(self.nstep):
            self.singleTimestep(**kargs)

    def revert(self):
        """
        Revert evolve method if update=False was used, otherwise
        it has no effect.
        """
        self.Sigma.setValue(self.Sigma.old.value)

def run(**kargs):
    circ = circumbinary(**kargs) 
    circ.evolve()
    print circ.Sigma
    return circ.Sigma

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
             description="Script that solves the convection problem in a cylindrical grid")
    parser.add_argument('--rmax', default=1.0e3, type=float,
                        help='The outer boundary of the grid in dimensionless units (r/rMin)')
    parser.add_argument('--ncell', default=100, type=int,
                        help='The number of cells to use in the grid')
    parser.add_argument('--nstep', default=100, type=int,
                        help='The number of time steps to do')
    parser.add_argument('--nsweep', default=10, type=int,
                        help='The number of sweeps to do')
    parser.add_argument('--titer', default=10, type=int,
                        help='The number of temprature iterations')
    parser.add_argument('--dt', default=1.0e-6, type=float,
                        help='The time step size (Constant for the moment)')
    parser.add_argument('--delta', default=1.0e-100, type=float,
                        help='Small number to add to avoid divisions by zero')
    kargs = vars(parser.parse_args())
    run(**kargs)
