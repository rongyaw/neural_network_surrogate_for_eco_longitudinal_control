################################
### Tyler Ard                ###
### Argonne National Lab     ###
### Vehicle Mobility Systems ###
### tard(at)anl(dot)gov      ###
################################

from math import inf, sin, cos, sqrt, tan, atan, atan2, fmod
import numpy as np

from _constants import * 

### Functions
def distance(x, x2, y, y2, theta=0.):
    '''Euclidean distance between points in forward direction from heading theta'''
    return (x2 - x)*cos(theta) + (y2 - y)*sin(theta)

def getFrontVeh(nvs, v):
    '''Get nv immediately in front'''
    nv_ds = 2000.
    nv_vr = 20.
    nv_a = 0.
    i = -1

    for k in (k for k in range(0, len(nvs)) if v.id != nvs[k].id):
        d0 = nvs[k].s - (v.s + v.len) # Ego front bumper to nv back bumper
        
        if d0 > -2-v.len and d0 < nv_ds: # Behind another neighboring vehicle in current lane
            nv_ds, nv_vr, nv_a, i = d0, nvs[k].v, nvs[k].a, k

    return nv_ds, nv_vr, nv_a, i

def getRearVeh(nvs, v):
    '''Get nv immediately behind'''
    nv_ds = -2000.
    nv_vr = 20.
    nv_a = 0.
    i = -1

    for k in (k for k in range(0, len(nvs)) if v.id != nvs[k].id):
        d0 = v.s - (nvs[k].s + nvs[k].len) # Ego back bumper to nv front bumper

        if d0 < 2+nvs[k].len and d0 > nv_ds: # In front of another neighboring vehicle in current lane
            nv_ds, nv_vr, i = d0, nvs[k].v, nvs[k].a, k

    return nv_ds, nv_vr, nv_a, i