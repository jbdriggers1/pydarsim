# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
from numpy import deg2rad as d2r
import logging

import pydarsim.players.sensor as sensor
import pydarsim.support.tools as tools
import pydarsim.support.messages as messages
import pydarsim.support.xfrm as xfrm


class SimpleRadar(sensor.Sensor):
    ''' Simple noised-up-truth radar model '''

    log = logging.getLogger('default')

    def __init__(self):
        super().__init__()
        self.sensor_id = super().sensor_id
        self.config_processed = False
        self.sensor_model = 'SimpleRadar'
        self.spatial_initialized = False
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
        self.bearing_max = 2*np.pi-sys.float_info.epsilon if dl['bearing_max'] == 'None' else tools.map_0_to_2pi(d2r(dl['bearing_max']))
        self.elevation_min = -np.pi if dl['elevation_min'] == 'None' else tools.map_pi_to_pi(d2r(dl['elevation_min']))
        self.elevation_max = np.pi if dl['elevation_max'] == 'None' else tools.map_pi_to_pi(d2r(dl['elevation_max']))
        self.rdot_min = 0 if dl['rdot_min'] == 'None' else dl['rdot_min']
        self.rdot_max = np.inf if dl['rdot_max'] == 'None' else dl['rdot_max']
        self.altitude_min = -np.inf if dl['altitude_min'] == 'None' else dl['altitude_min']
        self.altitude_max = np.inf if dl['altitude_max'] == 'None' else dl['altitude_max']

        assert (self.bearing_min == 'None' and self.bearing_max == 'None') or (self.bearing_min != 'None' and self.bearing_max != 'None'), "Must set both min and max if using bearing limits"

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


    def perform_detections_on_target(self, target_spatial):

        detections = []
        target_ended = False
        time = -self.update_interval

        while not target_ended:
            time += self.update_interval

            sensor_spatial_t = self.SpatialData.get_state(time, 'propagate')
            target_spatial_t = target_spatial.get_state(time, 'propagate')

            # if either spatial returns that the requested spatial at time t is outside the generated trajectory,
            # detections are over
            if target_spatial_t == 0 or sensor_spatial_t == 0:
                if target_spatial_t == 0:
                    SimpleRadar.log.info('Target {} exited scenario at time {}'.format(target_spatial.spatial_name, time))
                if sensor_spatial_t == 0:
                    SimpleRadar.log.info('Sensor {} exited scenario at time {}'.format(self.sensor_name, time))
                target_ended = True
                return detections

            detection = self.form_truth_into_detection(sensor_spatial_t, target_spatial_t)
            if self.detection_with_limits(detection):
                SimpleRadar.log.debug('Sensor {} Detection - Time: {} | Range: {} | Az: {} | El: {} | Rdot: {}'.format(self.sensor_name, detection.time, detection.x,
                                                                                                                       detection.y, detection.z, detection.doppler))
                detections.append(detection)
            else:
                SimpleRadar.log.debug('Sensor {} Missed Detection - Time: {} | Range: {} | Az: {} | El: {} | Rdot: {}'.format(self.sensor_name, detection.time, detection.x,
                                                                                                                              detection.y, detection.z, detection.doppler))


    def form_truth_into_detection(self, sensor_spatial_state, target_spatial_state):

        assert np.isclose(sensor_spatial_state.time, sensor_spatial_state.time), "These states do not go together in time"

        # create detection "message"
        det = messages.Detection(coord_sys='RBE')
        det.time = target_spatial_state.time

        # get sensor and target states
        sensor_ecef = sensor_spatial_state.get_state(order=2)
        target_ecef = target_spatial_state.get_state(order=2)

        # get target in rbe relative to sensor loc
        sensor_ecef_pos = sensor_spatial_state.get_state(order=1)
        sensor_lla = xfrm.ecef_to_lla(sensor_ecef_pos)
        target_rbe = xfrm.ecef_to_rbe(target_ecef, sensor_lla)

        # get sensor in rbe relative to target loc (this is just for getting doppler)
        target_ecef_pos = target_spatial_state.get_state(order=1)
        target_lla = xfrm.ecef_to_lla(target_ecef_pos)
        sensor_rbe = xfrm.ecef_to_rbe(sensor_ecef, target_lla)

        # save sensor loc to message and target truth position
        det.sensor_lat = sensor_lla[0,0]
        det.sensor_lon = sensor_lla[1,0]
        det.sensor_alt = sensor_lla[2,0]
        det.truth_x = target_rbe[0,0]
        det.truth_y = target_rbe[2,0]
        det.truth_z = target_rbe[4,0]

        # meas
        det.x = det.truth_x + self.range_error_mean + self.range_error_sigma*np.random.randn()
        det.y = tools.map_pi_to_pi(det.truth_y + self.bearing_error_mean + self.bearing_error_sigma*np.random.randn())
        det.z = tools.map_pi_to_pi(det.truth_z + self.elevation_error_mean + self.elevation_error_sigma*np.random.randn())

        # i think this is required since the sensor is not required to be stationary
        target_rdot = target_rbe[1,0]
        sensor_rdot = sensor_rbe[1,0]
        truth_rdot = target_rdot + sensor_rdot

        det.truth_doppler = truth_rdot
        det.doppler = truth_rdot + self.rdot_error_mean + self.rdot_error_sigma*np.random.randn()

        # var
        min_var = sys.float_info.epsilon
        det.x_var = max(self.range_error_sigma_report**2, min_var)
        det.y_var = max(self.bearing_error_sigma_report**2, min_var)
        det.z_var = max(self.elevation_error_sigma_report**2, min_var)
        det.d_var = max(self.rdot_error_sigma_report**2, min_var)

        # err
        det.x_err = det.x - det.truth_x
        det.y_err = tools.map_pi_to_pi(det.y - det.truth_y)
        det.z_err = tools.map_pi_to_pi(det.z - det.truth_z)
        det.d_err = det.doppler - det.truth_doppler

        return det


    def detection_with_limits(self, det):

        if det.x < self.range_min:
            return False
        if det.x > self.range_max:
            return False
        if not tools.between_angles(det.y, self.bearing_min, self.bearing_max):
            return False
        if det.z < self.elevation_min:
            return False
        if det.z > self.elevation_max:
            return False
        if abs(det.doppler) < self.rdot_min:
            return False
        if abs(det.doppler) > self.rdot_max:
            return False

        det_rbe = np.array([[det.x],
                            [det.y],
                            [det.z]])
        det_plat_lla = np.array([[det.sensor_lat],
                                 [det.sensor_lon],
                                 [det.sensor_alt]])
        det_alt = xfrm.ecef_to_lla(xfrm.rbe_to_ecef(det_rbe, det_plat_lla))[2,0]
        if det_alt < self.altitude_min:
            return False
        if det_alt > self.altitude_max:
            return False

        return True


    def __del__(self):
        SimpleRadar.log.info("SimpleRadar with sensor id {} destructed".format(self.sensor_id))
        super().__del__()
