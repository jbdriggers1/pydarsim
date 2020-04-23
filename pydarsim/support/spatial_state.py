# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:24:57 2020

@author: John
"""


import numpy as np
from pydarsim.support import xfrm


class SpatialState(object):
    '''container for spatial information'''
    
    
    def __init__(self):
        '''initialize values to zero. using 3rd order column vector format [x, dx, ddx, y...]
        '''
        
        self.time = 0.0
        self.state = np.zeros((9,1))  # column vector
        self.speed = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw_rate = 0.0
        self.pitch_rate = 0.0
        self.roll_rate = 0.0
        self.axial_accel = 0.0
        self.lateral_accel = 0.0
        self.vertical_accel = 0.0
    
    
    def set_state(self, state):
        self.state = state
        vel = self.get_velocity()
        self.speed = np.linalg.norm(vel)
    
    
    def set_position(self, vec):
        self.state[0,0] = vec[0,0]
        self.state[3,0] = vec[1,0]
        self.state[6,0] = vec[2,0]
    
    
    def set_velocity(self, vec):
        self.state[1,0] = vec[0,0]
        self.state[4,0] = vec[1,0]
        self.state[7,0] = vec[2,0]
        
        self.speed = np.linalg.norm(vec)
    
    
    def set_acceleration(self, vec):
        self.state[2,0] = vec[0,0]
        self.state[5,0] = vec[1,0]
        self.state[8,0] = vec[2,0]
    
    
    def get_position(self):
        return np.array([[self.state[0,0]],
                         [self.state[3,0]],
                         [self.state[6,0]]])
    
    
    def get_velocity(self):
        return np.array([[self.state[1,0]],
                         [self.state[4,0]],
                         [self.state[7,0]]])
    
    
    def get_acceleration(self):
        return np.array([[self.state[2,0]],
                         [self.state[5,0]],
                         [self.state[8,0]]])
    
    
    def get_orientation(self):
        return np.array([[self.yaw],
                         [self.pitch],
                         [self.roll]])
    
    
    def set_orientation(self, ypr):
        self.yaw = ypr[0,0]
        self.pitch = ypr[1,0]
        self.roll = ypr[2,0]
    
    
    def get_ang_vel(self):
        return np.array([[self.yaw_rate],
                         [self.pitch_rate],
                         [self.roll_rate]])
    
    
    def set_ang_vel(self, dypr):
        self.yaw_rate = dypr[0,0]
        self.pitch_rate = dypr[1,0]
        self.roll_rate = dypr[2,0]
    
    
    def get_full_state_list(self):
        return [self.time,
                self.state[0,0],
                self.state[3,0],
                self.state[6,0],
                self.state[1,0],
                self.state[4,0],
                self.state[7,0],
                self.state[2,0],
                self.state[5,0],
                self.state[8,0],
                self.speed,
                self.yaw,
                self.pitch,
                self.roll,
                self.yaw_rate,
                self.pitch_rate,
                self.roll_rate,
                self.axial_accel,
                self.lateral_accel,
                self.vertical_accel]
    
    
    def get_course(self):
        return np.array([[self.speed],
                         [self.yaw],
                         [self.pitch]])
    
    
    
