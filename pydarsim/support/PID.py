# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:01:57 2020

@author: John
"""


class PID:


    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.perror = 0.0
        self.integral = 0.0
        self.derivative = 0.0


    def update(self, dt, error):
        self.integral += ( error * dt )
        self.derivative = (error - self.perror) / dt
        self.perror = error
        return self.kp*self.perror + self.ki*self.integral + self.kd*self.derivative


    def reset(self):
        self.integral = 0.0
        self.derivative = 0.0
        self.perror = 0.0
