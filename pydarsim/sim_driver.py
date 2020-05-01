# -*- coding: utf-8 -*-
'''#############################################################################
    sim_driver.py -
        Main driver of the simulation. This is what is called to kick off the
        whole thing. In a python environment with all the necessary packages,
        call 'python sim_driver.py <path to job config yaml>' to run the sim.

        Format of configuration yaml: (# comments not necessary)

~~~~~~Start YAML~~~~~~
job_info:
    name: test_job1  # name of directory created to put all output in
    output_dir: "./test/test_jobs"  # where this directory will be put
    sim_duration: 300  # seconds, duration of simulation
    log_level: DEBUG  # level above which to log. DEBUG, INFO, WARNING, ERROR
    verbose: True  # print log data to console if True

# this example has 1 sensor and 1 target
players:  # add players in given dash (-) format, on a new line and indented
    sensors:  # paths to sensor config YAMLs define new sensors
        - './test/test_jobs/Radar1.yaml'

    targets:  # paths to spatial YAMLs define new targets
        - './test/test_jobs/straight_and_level.yaml'
~~~~~~Stop YAML~~~~~~

#############################################################################'''


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
