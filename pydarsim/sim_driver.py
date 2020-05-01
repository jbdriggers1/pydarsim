# -*- coding: utf-8 -*-

import os
import sys

import pydarsim.sim_master as sim_master


def run(job_config_yaml_fp):
    ''' For the given sim config file, this intializes and drives the simulation '''

    # Create sim object
    Sim = sim_master.SimMaster()

    # process the sim configuration file
    Sim.process_config(job_config_yaml_fp)

    # initialize all the sensors, includes initializing their spatials
    Sim.initialize_sensors()

    # initialize target spatials
    Sim.initialize_targets()

    # Generate all trajectories and also write to output dir
    Sim.generate_sensor_spatial_trajectories()
    Sim.generate_target_spatial_trajectories()

    # Perform detections for all target/sensor pairs
    Sim.perform_all_sensor_target_pair_detections()

    # Write output logs
    Sim.write_output_logs()


if __name__ == '__main__':
    script_name = sys.argv[0]
    config_yaml_fp = sys.argv[1]
    #config_yaml_fp = 'C:/Users/John/Documents/Python_Scripts/pydarsim/pydarsim/test/test_jobs/test_job1.yaml'
    run(config_yaml_fp)
