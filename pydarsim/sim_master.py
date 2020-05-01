# -*- coding: utf-8 -*-

import os
import sys
import shutil
import logging
import itertools
import pandas as pd

import pydarsim.support.tools as tools
import pydarsim.support.my_logging as my_logging
import pydarsim.players.spatial as spatial
import pydarsim.players.sensor as sensor
import pydarsim.support.messages as messages


class SimMaster:
    ''' Simulation master/manager. The interface between all players, etc. in the simulation '''

    sim_id = 0

    def __init__(self):
        SimMaster.sim_id += 1
        self.sim_id = SimMaster.sim_id

        print("SimMaster id {} initialized".format(self.sim_id))

        self.sensor_players = []
        self.target_players = []

        self.meas_data = []  # where detections will be stored


    def process_config(self, fp):
        ''' process the simulation configuration which defines the sensors, targets, sim name, dir, duration, etc.'''

        self.config_fp = os.path.abspath(fp)
        print("SimMaster id {} processing config yaml at {}".format(self.sim_id, self.config_fp))

        self.config_dict = tools.load_yaml(self.config_fp)

        # Job Info
        self.job_name = self.config_dict['job_info']['name']
        self.output_dir = self.config_dict['job_info']['output_dir']
        self.sim_duration = self.config_dict['job_info']['sim_duration']
        self.log_level = self.config_dict['job_info']['log_level']
        self.verbose = self.config_dict['job_info']['verbose']


        # Player Configs
        self.sensor_config_fps = self.config_dict['players']['sensors']
        self.target_config_fps = self.config_dict['players']['targets']

        # Some assertions before continuing
        # assert len(self.sensor_config_fps) > 0, 'There must be at least 1 sensor'
        assert len(self.target_config_fps) > 0, 'There must be at least 1 target'
        assert self.sim_duration > 0, 'Sim Duration must be greater than 0'

        for path in self.sensor_config_fps:
            assert os.path.exists(path), 'Sensor config at {} does not exist'.format(path)
        for path in self.target_config_fps:
            assert os.path.exists(path), 'Target config at {} does not exist'.format(path)

        # Get job dir ready
        self.output_dir = tools.makedir(os.path.join(self.output_dir, self.job_name))
        tools.copy2dir(self.config_fp, self.output_dir)
        for path in self.sensor_config_fps:
            tools.copy2dir(path, self.output_dir)
        for path in self.target_config_fps:
            tools.copy2dir(path, self.output_dir)

        # Get logger set up
        self.log = my_logging.configure_logger('default', os.path.join(self.output_dir, 'logger.log'))
        self.log.setLevel(self.log_level)
        if not self.verbose:
            self.log = my_logging.remove_console_handler(self.log)

        self.log.info("SimMaster id {} initialized".format(self.sim_id))
        self.log.info("SimMaster id {} processing config yaml at {}".format(self.sim_id, self.config_fp))
        self.log.info("Created job dir at {}".format(self.output_dir))
        self.log.info("Config processed for job {}".format(self.job_name))


    def initialize_sensors(self):
        ''' initialize all sensors. process sensor config and initialize sensor spatial data '''

        self.log.info('Initializing Sensors')

        # loop through sensor configs, select sensor model, call its process_config function
        for i, sensor_config_fp in enumerate(self.sensor_config_fps):
            sensor_config = tools.load_yaml(sensor_config_fp)
            sensor_model = sensor_config['sensor']
            Sensor = sensor.SensorFactory.init_sensor_model(sensor_model)
            Sensor.process_config(sensor_config_fp)
            Sensor.initialize_sensor_spatial()
            Sensor.process_sensor_spatial_config()

            # limit sensor to sim duration
            if Sensor.SpatialData.stop_time > self.sim_duration:
                Sensor.SpatialData.stop_time = self.sim_duration

            tools.copy2dir(Sensor.config_fp, self.output_dir)
            tools.copy2dir(Sensor.spatial_config_fp, self.output_dir)
            self.sensor_players.append(Sensor)

        self.log.info('{} Sensors Initialized'.format(i+1))


    def initialize_targets(self):
        ''' initialize all targets. create spatial instances for all targets and process their configurations '''

        self.log.info('Initializing Targets')

        # loop through target configs, select sensor model, call its process_config function
        for i, target_config_fp in enumerate(self.target_config_fps):
            target_config = tools.load_yaml(target_config_fp)
            target_model = target_config['info']['spatial_model']
            Spatial = spatial.SpatialFactory.init_spatial_model(target_model)
            Spatial.process_config(target_config_fp)

            # limit target to sim duration
            if Spatial.stop_time > self.sim_duration:
                Spatial.stop_time = self.sim_duration

            tools.copy2dir(Spatial.config_fp, self.output_dir)
            self.target_players.append(Spatial)

        self.log.info('{} Targets Initialized'.format(i+1))


    def generate_sensor_spatial_trajectories(self):
        ''' generate all the sensor spatial trajectories '''

        self.log.info('Generating sensor spatial trajectories')

        for i, Sensor in enumerate(self.sensor_players):
            Sensor.SpatialData.generate_traj()
            Sensor.SpatialData.write_traj(os.path.join(self.output_dir, 'Sensor_{}_spatial.csv'.format(Sensor.sensor_name)), add_lla=True)

        self.log.info('Generated {} sensor trajectories'.format(i+1))


    def generate_target_spatial_trajectories(self):
        ''' generate all the target spatial trajectories '''

        self.log.info('Generating target spatial trajectories')

        for i, Target in enumerate(self.target_players):
            Target.generate_traj()
            Target.write_traj(os.path.join(self.output_dir, 'Target_{}_spatial.csv'.format(Target.spatial_name)), add_lla=True)

        self.log.info('Generated {} target trajectories'.format(i+1))


    def perform_sensor_target_pair_detections(sensor, target):
        ''' take the sensor model and the given target spatial and perform the sensor model's detection process,
            returning the detections '''

        detections = sensor.perform_detections_on_target(target)
        return detections


    def perform_all_sensor_target_pair_detections(self):
        ''' each sensor performs detections on each target '''

        self.log.info('Starting detection processes')

        for sensor, target in itertools.product(self.sensor_players, self.target_players):
            self.log.info('Detection Started - Sensor: {}, Target: {}'.format(sensor.sensor_name, target.spatial_name))
            detections = SimMaster.perform_sensor_target_pair_detections(sensor, target)
            self.meas_data = self.meas_data + detections

        self.log.info('Detection complete')


    def write_meas_output_log(self):
        ''' Get detections list, format, and write to output dir '''

        fp = os.path.join(self.output_dir, 'measdata.csv')
        self.log.info('Writing measdata log to {}'.format(fp))

        meas_data = [det.to_list() for det in self.meas_data]
        col_names = messages.Detection.get_field_names_list()
        meas_data = pd.DataFrame(meas_data, columns=col_names)
        meas_data.to_csv(fp, index=False)


    def write_output_logs(self):
        ''' Write any output logs. Only meas for now, add more later '''

        self.log.info('Writing output logs')

        self.write_meas_output_log()

        self.log.info('Finished writing output logs')


    def __del__(self):
        self.log.info("SimMaster id {} destructed".format(self.sim_id))
