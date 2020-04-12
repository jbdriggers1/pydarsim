# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:04:06 2020

@author: John
"""


from numpy import sin, cos, arctan2, arcsin, array, nan_to_num, matmul, identity, zeros, kron, sqrt, arccos, newaxis
from numpy.linalg import norm
from numpy import deg2rad as d2r
from numpy import rad2deg as r2d

from pdb import set_trace

z33 = zeros((3,3))
#earth_constant_eccen_sqr = 6.694379990141316e-003
earth_constant_eccen_sqr = 0  
#earth_constant_radius_eq = 6378137.0 
earth_constant_radius_eq = 6377388.0 


def rbe_to_enu(rbe, col=False):
    rbe = rbe.flatten()
    
    rng = rbe[0]
    bear = rbe[1]
    elev = rbe[2]
    
    east = rng * sin(bear) * cos(elev)
    north = rng * cos(bear) * cos(elev)
    up = rng * sin(elev)
    
    enu = array([east, north, up])
    enu = nan_to_num(enu)
    
    if col:
        return enu[:, newaxis]
    else:
        return enu


def enu_to_rbe(enu_pos, col=False):
    enu_pos = enu_pos.flatten()
    
    east = enu_pos[0]
    north = enu_pos[1]
    up = enu_pos[2]
    
    rng = norm(enu_pos)
    bear = arctan2(east,north)
    elev = arcsin(up/rng)
    
    rbe = array([rng, bear, elev])
    rbe = nan_to_num(rbe)
    
    if col:
        return rbe[:, newaxis]
    else:
        return rbe


def rbe_to_enu_cov(rbe_state, rbe_cov):
    
    rbe_state = rbe_state.flatten()
    
    r = rbe_state[0]
    az = rbe_state[1]
    el = rbe_state[2]
    
    J = array([[sin(az)*cos(el), r*cos(az)*cos(el),  -r*sin(az)*sin(el)],
               [cos(az)*cos(el), -r*sin(az)*cos(el), -r*cos(az)*sin(el)],
               [sin(el),                 0,          r*cos(el)]])
    
    enu_cov = matmul(J, matmul(rbe_cov, J.T))
    
    return enu_cov

        