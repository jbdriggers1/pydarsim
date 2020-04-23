# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:04:06 2020

@author: John
"""


from numpy import sin, cos, arctan2, arcsin, array, nan_to_num, matmul, identity, zeros, kron, sqrt, arccos, newaxis, eye, column_stack
from numpy.linalg import norm
from numpy import deg2rad as d2r
from numpy import rad2deg as r2d
from copy import deepcopy

from pdb import set_trace

z33 = zeros((3,3))
earth_constant_eccen_sqr = 6.694379990141316e-003
#earth_constant_eccen_sqr = 0  
earth_constant_radius_eq = 6378137.0 
#earth_constant_radius_eq = 6377388.0 


def rbe_to_enu(rbe):
    
    nrows = rbe.shape[0]
    enu = []
    
    if nrows == 3:  
        r = rbe[0,0]
        b = rbe[1,0]
        e = rbe[2,0]
    
    elif nrows == 6:
        r = rbe[0,0]
        dr = rbe[1,0]
        b = rbe[2,0]
        db = rbe[3,0]
        e = rbe[4,0]
        de = rbe[5,0]
    
    elif nrows == 9:
        r = rbe[0,0]
        dr = rbe[1,0]
        ddr = rbe[2,0]
        b = rbe[3,0]
        db = rbe[4,0]
        ddb = rbe[5,0]
        e = rbe[6,0]
        de = rbe[7,0]
        dde = rbe[8,0]
    
    else:
        raise('Wrong number of rows')
    
    cb = cos(b)
    ce = cos(e)
    sb = sin(b)
    se = sin(e)
        
    east = r * sb * ce
    north = r * cb * ce
    up = r * se
    
    enu.append([east, north, up])
    
    if nrows > 3:
        east_vel = (dr*sb*ce) + (r*db*cb*ce) - (r*de*sb*se)
        north_vel = (dr*cb*ce) - (r*db*sb*ce) - (r*de*cb*se)
        up_vel = (dr*se) + (r*de*ce)
        enu.append([east_vel, north_vel, up_vel])
    
    if nrows > 6:
        east_accel = (-r*sb*se*dde) + (r*cb*ce*ddb) + (ddr*sb*ce) - (2*sb*se*de*dr) + (2*cb*ce*db*dr) - (r*sb*ce*(db**2)) - (2*r*cb*se*db*de) - (r*sb*ce*(de**2))
        north_accel = (cb*ce*ddr) - (r*sb*ce*ddb) - (r*cb*se*dde) + (2*r*sb*se*db*de) - (r*cb*ce*(db**2)) - (r*cb*ce*(de**2)) - (2*sb*ce*db*dr) - (2*cb*se*de*dr)
        up_accel = (se*ddr) + (r*ce*dde) - (r*se*(de**2)) + (2*ce*de*dr)
        enu.append([east_accel, north_accel, up_accel])
    
    enu = column_stack(enu).reshape((nrows,1))
    return enu


def enu_to_rbe(enu_pos):
    
    nrows = enu_pos.shape[0]
    rbe = []
    
    if nrows == 3:
        e = enu_pos[0,0]
        n = enu_pos[1,0]
        u = enu_pos[2,0]
    elif nrows == 6:
        e = enu_pos[0,0]
        de = enu_pos[1,0]
        n = enu_pos[2,0]
        dn = enu_pos[3,0]
        u = enu_pos[4,0]
        du = enu_pos[5,0]
    elif nrows == 9:
        e = enu_pos[0,0]
        de = enu_pos[1,0]
        dde = enu_pos[2,0]
        n = enu_pos[3,0]
        dn = enu_pos[4,0]
        ddn = enu_pos[5,0]
        u = enu_pos[6,0]
        du = enu_pos[7,0]
        ddu = enu_pos[8,0]
    
    rng = (e**2 + n**2 + u**2)**0.5
    bear = arctan2(e,n)
    elev = arcsin(u/rng)
    
    rbe.append([rng, bear, elev])
    
    r2 = rng**2
    
    if nrows > 3:
        en = e**2 + n**2
        enu = rng**2
        drng = (e*de + n*dn + u*du) / rng
        dbear = ( n*de - e*dn ) / en
        delev = ( (en*du) - (e*u*de) - (n*u*dn) ) / ( ((en/enu)**2) * (enu**(2/3)) )
        rbe.append([drng, dbear, delev])
    
    if nrows > 6:
        ddrng = ( (r2 * (e*dde + n*ddn + u*ddu + (de**2) + (dn**2) + (du**2))) - ((e*de + n*dn + u*du)**2) ) / (r2**(3/2))
        ddbear = (-e*ddn/en) + (n*dde/en) + (2*(e**2)*de*dn/(en**2)) - (2*e*n*(de**2)/(en**2)) + (2*e*n*(dn**2)/(en**2)) - (2*(n**2)*de*dn/(en**2))
        t1 = 1 / ((1-((u**2)/enu))**0.5)
        t2 = ddu / rng
        t3 = ( u * (e*dde + u*ddu + n*ddn + (de**2) + (dn**2) + (du**2)) ) / (enu**(3/2))
        t4 = (2*e*de + 2*n*de + 2*u*de)
        t5 = (3*u*(t4**2)) / (4 * (enu**(5/2)))
        t6 = (du * t4) / (enu**(3/2))
        t7 = ((u**2) * t4) / (enu**2)
        t8 = (2*u*du) / enu
        t9 = du / rng
        t10 = (u * t4) / (2 * (enu**(3/2)))
        t11 = 2*((1 - (u**2)/enu)**(3/2))
        ddelev = ( t1 * (t2 - t3 + t5 - t6) ) - ( ((t7 - t8)*(t9 - t10)) / t11 )
        rbe.append([ddrng, ddbear, ddelev])
    
    rbe = column_stack(rbe).reshape((nrows,1))
    return rbe


def rbe_to_enu_cov(rbe_state, rbe_cov):
    
    r = rbe_state[0,0]
    az = rbe_state[1,0]
    el = rbe_state[2,0]
    
    J = array([[sin(az)*cos(el), r*cos(az)*cos(el),  -r*sin(az)*sin(el)],
               [cos(az)*cos(el), -r*sin(az)*cos(el), -r*cos(az)*sin(el)],
               [sin(el),                 0,          r*cos(el)]])
    
    enu_cov = matmul(J, matmul(rbe_cov, J.T))
    
    return enu_cov


def ecef_to_lla(ecef):
    
    radius_eq = earth_constant_radius_eq
    e2 = earth_constant_eccen_sqr
    
    a1 = radius_eq * e2
    a2 = a1 * a1
    a3 = a1 * e2 / 2
    a4 = 2.5 * a2
    a5 = a1 + a3
    a6 = 1 - e2
    
    thresh_val = 0.3
    
    x = ecef[0,0]
    y = ecef[1,0]
    z = ecef[2,0]
    
    zp = abs(z)
    w = norm([x,y])
    w2 = w * w
    
    r = norm([x, y, z])
    r2 = r * r
    s2 = (z*z)/r2
    c2 = w2/r2
    u = a2/r
    v = a3-a4/r
    
    lla = zeros((3,1))
    
    if (c2 > thresh_val):
        s = (zp / r) * (1 + c2 * (a1 + u + s2 * v) / r)
        lla[0,0] = arcsin(s)
        ss = s*s
        c = sqrt(1-ss)
    else:
        c = (w/r) * (1 - s2 * (a5 - u - c2 * v) / r)
        lla[0,0] = arccos(c)
        ss = 1 - (c*c)
        s = sqrt(ss)
    
    g = 1 - e2 * ss
    rg = radius_eq / sqrt(g)
    rf = a6 * rg
    u = w - rg * c
    v = zp - rf * s
    f = c * u + s * v
    m = c * v - s * u
    p = m / (rf / g + f)
    
    lla[0,0] = lla[0,0] + p
    
    if (z < 0):
        lla[0,0] = -lla[0,0]
    
    lla[2,0] = f + m * p / 2
    lla[1,0] = arctan2(y,x)
    
    return lla


def lla_to_ecef(lla):
    lat = lla[0,0]
    lon = lla[1,0]
    alt = lla[2,0]
    e2 = earth_constant_eccen_sqr
    
    temp = (earth_constant_radius_eq / sqrt(1 - (e2) * pow(sin(lat),2.0) ) ) + alt
    
    ecef = zeros((3,))
    ecef[0] = temp * cos(lat) * cos(lon)
    ecef[1] = temp * cos(lat) * sin(lon)
    temp = (temp - alt) * (1.0 - (e2)) + alt
    ecef[2] = temp * sin(lat)
    
    return ecef.reshape((3,1))


def enu_to_ecef(enu, lla):

    nrows = enu.shape[0]
    
    sin_lat = sin(lla[0,0])
    cos_lat = cos(lla[0,0])
    sin_lon = sin(lla[1,0])
    cos_lon = cos(lla[1,0])
    
    rot_val = zeros((3,3))
    
    rot_val[0,0] = -sin_lon
    rot_val[0,1] = -sin_lat * cos_lon
    rot_val[0,2] = cos_lat * cos_lon
    
    rot_val[1,0] = cos_lon
    rot_val[1,1] = -sin_lat * sin_lon
    rot_val[1,2] = cos_lat * sin_lon
    
    rot_val[2,0] = 0
    rot_val[2,1] = cos_lat
    rot_val[2,2] = sin_lat
    
    if nrows == 3:
        rot_mat = rot_val
    elif nrows == 6:
        rot_mat = kron(rot_val, eye(2))
    elif nrows == 9:
        rot_mat = kron(rot_val, eye(3))
    
    else:
        raise("nope")
    
    ecef = matmul(rot_mat, enu)
    
    lla_ecef = lla_to_ecef(lla)
    
    if nrows == 3:
        ecef = ecef + lla_ecef
    elif nrows == 6:
        ecef[0,0] += lla_ecef[0,0]
        ecef[2,0] += lla_ecef[1,0]
        ecef[4,0] += lla_ecef[2,0]
    elif nrows == 9:
        ecef[0,0] += lla_ecef[0,0]
        ecef[3,0] += lla_ecef[1,0]
        ecef[6,0] += lla_ecef[2,0]
    
    return ecef
    

def ecef_to_enu(ecef, lla):
    
    lla_ecef = lla_to_ecef(lla)
    
    nrows = ecef.shape[0]
    enu = deepcopy(ecef)
    
    if nrows == 3:
        enu -= lla_ecef
    elif nrows == 6:
        enu[0,0] -= lla_ecef[0,0]
        enu[2,0] -= lla_ecef[1,0]
        enu[4,0] -= lla_ecef[2,0]
    elif nrows == 9:
        enu[0,0] -= lla_ecef[0,0]
        enu[3,0] -= lla_ecef[1,0]
        enu[6,0] -= lla_ecef[2,0]
    
    sin_lat = sin(lla[0,0])
    cos_lat = cos(lla[0,0])
    sin_lon = sin(lla[1,0])
    cos_lon = cos(lla[1,0])
    
    rot_val = zeros((3,3))
    rot_val[0,0] = -sin_lon
    rot_val[0,1] = cos_lon
    rot_val[0,2] = 0.0
    
    rot_val[1,0] = -sin_lat * cos_lon
    rot_val[1,1] = -sin_lat * sin_lon
    rot_val[1,2] = cos_lat
    
    rot_val[2,0] = cos_lat * cos_lon
    rot_val[2,1] = cos_lat * sin_lon
    rot_val[2,2] = sin_lat
    
    if nrows == 3:
        rot_mat = rot_val
    elif nrows == 6:
        rot_mat = kron(rot_val, eye(2))
    elif nrows == 9:
        rot_mat = kron(rot_val, eye(3))
    else:
        raise('nope')
    
    enu = matmul(rot_mat, enu)
    
    return enu

    
