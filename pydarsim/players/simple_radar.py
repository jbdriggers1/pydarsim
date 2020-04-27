# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
from numpy import deg2rad as d2r
from numpy import rad2deg as r2d
import logging

import pydarsim.players.sensor as sensor
import pydarsim.support.spatial_state as spatial_state
import pydarsim.support.tools as tools
import pydarsim.support as support



class SimpleRadar(sensor.Sensor):
    ''' Simple noised-up-truth radar model '''

    log = logging.getLogger('default')

    def __init__(self):
        super().__init__()
        self.sensor_id = super().sensor_id
        self.config_processed = False
        self.sensor_model = 'SimpleRadar'
        SimpleRadar.log.info("SimpleRadar with sensor id {} initialized".format(self.sensor_id))


    def process_config(self, config_yaml_fp):
        ''' Once initialized, this will process the sensor configuration and define the parameters for the sensor instance '''

        SimpleRadar.log.info('Processing config for SimpleRadar id {}'.format(self.sensor_id))

        self.config_fp = config_yaml_fp
        self.config_dict = tools.load_yaml(self.config_fp)
        SimpleRadar.log.info('Loaded config from {}'.format(self.config_fp))

        # load general info
        self.sensor_name = self.config_dict['name']
        self.update_interval = self.config_dict['update_interval']
        self.spatial_config_fp = self.config_dict['spatial_config']
        assert os.path.exists(self.spatial_config_fp), "SimpleRadar {} spatial config file does not exist".format(self.sensor_name)
        self.spatial_model = tools.load_yaml(self.spatial_config_fp)['info']['model']

        # load detection limits
        dl = self.config_dict['detection_limits']
        self.range_min = 0.0 if dl['range_min'] == 'None' else dl['range_min']
        self.range_max = np.inf if dl['range_max'] == 'None' else dl['range_max']
        self.bearing_min = 0.0 if dl['bearing_min'] == 'None' else tools.map_0_to_2pi(d2r(dl['bearing_min']))
        self.bearing_max = 2*np.pi if dl['bearing_max'] == 'None' else tools.map_0_to_2pi(d2r(dl['bearing_max']))
        self.elevation_min = -np.pi if dl['elevation_min'] == 'None' else tools.map_pi_to_pi(d2r(dl['elevation_min']))
        self.elevation_max = np.pi if dl['elevation_max'] == 'None' else tools.map_pi_to_pi(d2r(dl['elevation_max']))
        self.rdot_min = 0.0 if dl['rdot_min'] == 'None' else dl['rdot_min']
        self.rdot_max = np.inf if dl['rdot_max'] == 'None' else dl['rdot_max']
        self.altitude_min = 0.0 if dl['altitude_min'] == 'None' else dl['altitude_min']
        self.altitude_max = np.inf if dl['altitude_max'] == 'None' else dl['altitude_max']

        # load detection parameters
        dp = self.config_dict['detection_parameters']
        self.range_error_mean = dp['range_error_mean']
        self.range_error_sigma = dp['range_error_sigma']
        self.range_error_sigma_report = dp['range_error_sigma_report']
        self.bearing_error_mean = dp['bearing_error_mean']
        self.bearing_error_sigma = dp['bearing_error_sigma']
        self.bearing_error_sigma_report = dp['bearing_error_sigma_report']
        self.elevation_error_mean = dp['elevation_error_mean']
        self.elevation_error_sigma = dp['elevation_error_sigma']
        self.elevation_error_sigma_report = dp['elevation_error_sigma_report']
        self.rdot_error_mean = dp['rdot_error_mean']
        self.rdot_error_sigma = dp['rdot_error_sigma']
        self.rdot_error_sigma_report = dp['rdot_error_sigma_report']

        self.config_processed = True
        SimpleRadar.log.info('Config processed for SimpleRadar id {}'.format(self.sensor_id))


    def __del__(self):
        SimpleRadar.log.info("SimpleRadar with sensor id {} destructed".format(self.sensor_id))
        super().__del__()

