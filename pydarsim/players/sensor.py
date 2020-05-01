# -*- coding: utf-8 -*-

import logging

import pydarsim.players.spatial as spatial


class SensorFactory:
    ''' For Generating the Requested Sensor Model'''

    @staticmethod
    def init_sensor_model(sensor_model):
        if sensor_model == 'SimpleRadar':
            import pydarsim.players.simple_radar as simple_radar  # hate that this is here, but was having circular import issues...
            return simple_radar.SimpleRadar()  # return class or instance?
        else:
            raise TypeError('No sensor model {} found'.format(sensor_model))


class Sensor:  # make abstract with ABC?
    ''' Abstract Sensor Class. Sensor model must inherit from this '''

    sensor_id = 0
    log = logging.getLogger('default')

    def __init__(self):
        Sensor.sensor_id += 1
        self.sensor_id = Sensor.sensor_id
        self.spatial_config_fp = None
        self.spatial_initialized = False
        self.spatial_config_processed = False
        self.spatial_traj_generated = False
        Sensor.log.info("Sensor id {} initialized".format(Sensor.sensor_id))


    def process_config(self):
        ''' specific sensor models must handle this. should i make this an abstract method? '''
        pass

    def perform_detections_on_target(self, target_spatial):
        ''' sensor model specific, abstract method '''
        pass


    def initialize_sensor_spatial(self):
        ''' if config is processed, this will initialize spatial data for the sensor model '''

        assert hasattr(self, 'config_processed'), "Cannot initialize spatial for Sensor {} without processing sensor config".format(self.sensor_id)
        assert self.config_processed == True, "Cannot initialize spatial for Sensor {} without processing sensor config".format(self.sensor_id)

        self.SpatialData = spatial.SpatialFactory.init_spatial_model(self.spatial_model)
        self.spatial_initialized = True
        Sensor.log.info("Spatial for Sensor ID {} initialized".format(self.sensor_id))


    def process_sensor_spatial_config(self):
        ''' if spatial_config_fp is defined, this will process it so the spatial will be ready to generate the sensor's trajectory '''

        assert self.spatial_config_fp != None, "Sensor id {} has no spatial config file to process".format(self.sensor_id)
        assert hasattr(self, 'SpatialData'), "Sensor id {} spatial not initialized".foramt(self.sensor_id)

        self.SpatialData.process_config(self.spatial_config_fp)
        self.spatial_config_processed = True
        Sensor.log.info("Spatial for Sensor ID {} processed".format(self.sensor_id))


    def __del__(self):
        Sensor.log.info("Sensor id {} destructed".format(self.sensor_id))
