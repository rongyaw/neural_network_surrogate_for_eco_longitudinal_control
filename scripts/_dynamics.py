################################
### Tyler Ard                ###
### Argonne National Lab     ###
### Vehicle Mobility Systems ###
### tard(at)anl(dot)gov      ###
################################

from math import sin, cos, sqrt, tan, atan, atan2, fmod, pi
import numpy as np

from _constants import * 

def dyn(t, x, u):
    '''x = [s, v, a], u = ua'''
    TAUINV = 1./0.275

    return np.array([
        x[1],
        x[2],
        TAUINV*(u[0]-x[2])
    ])

def RK2(f, dt, t, x, u):
    '''Single-step second-order Runge-Kutta integrator'''
    k1 = dt*f(t, x, u)
    k2 = dt*f(t+0.5*dt, x+0.5*k1, u)

    t = t+dt
    x = x+0.5*(k1+k2)
    return x

def RK4(f, dt, t, x, u):
    '''Single-step fourth-order Runge-Kutta integrator'''
    k1 = dt*f(t, x, u)
    k2 = dt*f(t+0.5*dt, x+0.5*k1, u)
    k3 = dt*f(t+0.5*dt, x+0.5*k2, u)
    k4 = dt*f(t+dt, x+k3, u)
    
    t = t+dt
    x = x+0.1666667*(k1+2.*k2+2.*k3+k4)
    return x
