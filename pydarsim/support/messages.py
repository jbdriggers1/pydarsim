# -*- coding: utf-8 -*-

import numpy as np


class Detection:
    ''' Class to generalize detection data format '''

    def __init__(self, coord_sys='RBE'):

        self.coord_sys = coord_sys  # RBE or RUV

        self.time = np.nan
        self.x = np.nan
        self.y = np.nan
        self.z = np.nan
        self.doppler = np.nan
        self.x_var = np.nan
        self.y_var = np.nan
        self.z_var = np.nan
        self.d_var = np.nan
        self.truth_doppler = np.nan
        self.sensor_lat = np.nan
        self.sensor_lon = np.nan
        self.sensor_alt = np.nan
        self.truth_x = np.nan
        self.truth_y = np.nan
        self.truth_z = np.nan
        self.x_err = np.nan
        self.y_err = np.nan
        self.z_err = np.nan
        self.d_err = np.nan


    def get_field_names_list():
        return ['time', 'coord', 'x', 'y', 'z', 'rdot', 'x_var', 'y_var', 'z_var',
                'rdot_var', 'sensor_lat', 'sensor_lon', 'sensor_alt',
                'truth_x', 'truth_y', 'truth_z', 'truth_rdot', 'x_err', 'y_err',
                'z_err', 'rdot_err']


    def to_list(self):
        return [self.time, self.coord_sys, self.x, self.y, self.z, self.doppler,
                self.x_var, self.y_var, self.z_var, self.d_var, self.sensor_lat,
                self.sensor_lon, self.sensor_alt, self.truth_x, self.truth_y,
                self.truth_z, self.truth_doppler, self.x_err, self.y_err,
                self.z_err, self.d_err]
