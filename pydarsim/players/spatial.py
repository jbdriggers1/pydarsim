# -*- coding: utf-8 -*-
'''#############################################################################
    spatial.py -
        Spatial classes to generate trajectories (for targets or sensors or
        whatever). Only one generated now is SpatialEarth. Generates constant
        altitude targets with speed, heading, and altitude maneuvers. Requires
        configuration yaml for define parameters of trajectory.

        Format of configuration yaml:
        (# comments are not necessary)

~~~~~~Start YAML~~~~~~
info:
    name: Target  # string name of Spatial entity to use

initial:  # initial location and kinematic data
    start_time: 0.0  # seconds
    stop_time: 300.0  # seconds
    latitude: 2.0  # degrees
    longitude: 0.0  # degrees
    altitude: 10000.0  # meters
    heading: 180.0  # degrees (0 to 360)
    pitch_angle: 0.0  # degrees (just leave at 0)
    speed: 300.0  # m/s

parameters:  # some kinematic limitations
    update_rate: 0.1  # reported state rate (internally generated at 200 Hz)
    max_lateral_gs: 3.0  # Gs, affects heading change maneuvers
    max_vertical_gs: 3.0 # Gs, affects altitude maneuvers
    max_gs: 3.0  # Gs, affects speed maneuvers
    autopilot: True  # True For PD loop method of maneuvers (preferred True)
    accel_time_const: 0.25  # smaller for more abrupt accelerations, default 0.25

# speed, heading, and altitude maneuvers in a single maneuver are okay, but
# try not to overlap consecutive maneuvers (1: needs to finish before 2: starts)
maneuvers:  # single maneuver example, add more maneuvers with 2:, 3:, etc...
    1:
        time: 100.0  # time when maneuver starts (s)
        final_speed: 500.0  # speed at end of maneuver (delete line if no speed change desired)
        final_heading: 0.0  # degrees, heading at end of maneuver (delete line if no heading change desired)
        final_altitude: 5000  # meters, altitude at end of maneuver (delete line if no altitude change desired)
        climb_rate: 50  # m/s, climb or descent rate limitation
~~~~~~Stop YAML~~~~~~

#############################################################################'''


import os
import numpy as np
from numpy import deg2rad as d2r
from numpy import rad2deg as r2d
from pandas import DataFrame
from copy import deepcopy
import logging
import pandas as pd

import pydarsim.support.tools as tools
import pydarsim.support as support
import pydarsim.support.spatial_state as spatial_state
import pydarsim.support.PID as PID

from pdb import set_trace


class SpatialFactory:
    ''' Generate requested spatial model '''

    @staticmethod
    def init_spatial_model(spatial_model):
        if spatial_model == 'SpatialEarth':
            return SpatialEarth()
        else:
            raise TypeError('No spatial model {} found'.format(spatial_model))


class SpatialEarth:
    '''
    Class for generating simple spatial trajectories with parameters defined in a
    yaml configuration file. Supported maneuvers are simple heading change, speedup,
    and pitch maneuvers.

    Issues:
        Check github for issues, there any many
    '''

    spatial_id = 0
    log = logging.getLogger('default')

    def __init__(self):

        SpatialEarth.spatial_id = SpatialEarth.spatial_id + 1
        self.spatial_id = SpatialEarth.spatial_id

        self.config_processed = False
        self.traj_generated = False

        self.integration_step = 0.005

        SpatialEarth.log.info('SpatialEarth object number {} initialized'.format(SpatialEarth.spatial_id))


    def __str__(self):

        states = self.get_all_states(add_lla=True)
        start_time = states.time.min()
        stop_time = states.time.max()
        duration = stop_time - start_time
        max_speed = states.speed.max()
        min_speed = states.speed.min()
        max_gs = max((states.ddx**2 + states.ddy**2 + states.ddz**2)**0.5) / 9.81
        min_heading, max_heading = r2d(states.yaw.min()), r2d(states.yaw.max())
        min_pitch, max_pitch = r2d(states.pitch.min()), r2d(states.pitch.max())
        min_x, max_x = states.x.min(), states.x.max()
        min_y, max_y = states.y.min(), states.y.max()
        min_z, max_z = states.z.min(), states.z.max()
        min_lat, max_lat = r2d(states.LAT.min()), r2d(states.LAT.max())
        min_lon, max_lon = r2d(states.LON.min()), r2d(states.LON.max())
        min_alt, max_alt = states.ALT.min(), states.ALT.max()

        s = ''
        s += "SpatialEarth Name: {}\n".format(self.spatial_name)
        s += 'SpatialEarth ID: {}\n'.format(self.spatial_id)

        if self.traj_generated:
            s += 'Start time {:.2f}\n'.format(start_time)
            s += 'Stop time {:.2f}\n'.format(stop_time)
            s += 'Duration {:.2f}\n'.format(duration)
            s += 'Min/Max Speed: ({:.2f}, {:.2f})\n'.format(min_speed, max_speed)
            s += 'Max Total Gs: {:.2f}\n'.format(max_gs)
            s += 'Min/Max Heading: ({:.2f}, {:.2f})\n'.format(min_heading, max_heading)
            s += 'Min/Max Pitch: ({:.2f}, {:.2f})\n'.format(min_pitch, max_pitch)
            s += 'Min/Max X: ({:.2f}, {:.2f})\n'.format(min_x, max_x)
            s += 'Min/Max Y: ({:.2f}, {:.2f})\n'.format(min_y, max_y)
            s += 'Min/Max Z: ({:.2f}, {:.2f})\n'.format(min_z, max_z)
            s += 'Min/Max Latitude: ({:.4f}, {:.4f})\n'.format(min_lat, max_lat)
            s += 'Min/Max Longitude: ({:.4f}, {:.4f})\n'.format(min_lon, max_lon)
            s += 'Min/Max Altitude: ({:.2f}, {:.2f})\n'.format(min_alt, max_alt)

        return s


    def process_config(self, spatial_config_yaml_fp):
        '''
        Process spatial configuration yaml and set object attributes from contents.

        Args:
            spatial_config_yaml_fp: (str) filepath to config yaml

        Attributes Created:
            config_fp: (str) filepath to config yaml
            config_yaml: (dict) parsed configuration
            init_params: (dict) initial kinematic parameters for spatial
            kinematic_params: (dict) kinematic limitations and such
            maneuvers: (dict) if exists, set of maneuvers and their specs

        Returns:
            None
        '''

        # load the file into a dict
        self.config_fp = spatial_config_yaml_fp
        self.config_yaml = tools.load_yaml(self.config_fp)

        # check to make sure we have what we need an separate them
        assert 'info' in self.config_yaml, 'SpatialEarth id {} config has no info section'.format(self.spatial_id)
        self.info_params = self.config_yaml['info']

        assert 'initial' in self.config_yaml, 'SpatialEarth id {} config has no initial section'.format(self.spatial_id)
        self.init_params = self.config_yaml['initial']

        assert 'parameters' in self.config_yaml, 'SpatialEarth id {} config has no parameters section'.format(self.spatial_id)
        self.kinematic_params = self.config_yaml['parameters']

        if 'maneuvers' in self.config_yaml and self.config_yaml['maneuvers'] is not None:
            self.maneuvers = self.config_yaml['maneuvers']
        else:
            self.maneuvers = {}


        # Process config

        # info parameters
        self.spatial_name = self.info_params['name']

        # Generate sampling times
        self.start_time = self.init_params['start_time']
        self.stop_time = self.init_params['stop_time']
        self.update_rate = self.kinematic_params['update_rate']  # update period in seconds
        self.sample_times = np.arange(self.start_time, self.stop_time+self.update_rate, self.integration_step)
        self.report_times = np.arange(self.start_time, self.stop_time+self.update_rate, self.update_rate)

        # unpack max gs info
        self.max_gs = self.kinematic_params['max_gs']
        self.max_lat_gs = self.kinematic_params['max_lateral_gs']
        self.max_vert_gs = self.kinematic_params['max_vertical_gs']
        self.accel_time_const = self.kinematic_params['accel_time_const']

        # get initial position
        latitude = d2r(self.init_params['latitude'])
        longitude = d2r(self.init_params['longitude'])
        altitude = self.init_params['altitude']
        lla = np.array([[latitude, longitude, altitude]]).T
        ecef_pos = support.xfrm.lla_to_ecef(lla)
        enu_pos = support.xfrm.ecef_to_enu(ecef_pos, lla)

        # get initial velocity
        self.initial_speed = self.init_params['speed']
        self.initial_heading = tools.map_0_to_2pi(d2r(self.init_params['heading']))
        self.initial_pitch_angle = tools.map_pi_to_pi(d2r(self.init_params['pitch_angle']))
        enu_vel = support.xfrm.rbe_to_enu(np.array([[self.initial_speed],
                                            [self.initial_heading],
                                            [self.initial_pitch_angle]]))

        self.initial_acceleration = np.zeros((3,1))

        enu_state = np.column_stack((enu_pos, enu_vel, self.initial_acceleration)).reshape((9,1))
        ecef_state = support.xfrm.enu_to_ecef(enu_state, lla)

        # form initial state
        self.initial_state = ecef_state

        # mark maneuvers an incomplete
        for man_id in self.maneuvers:
            self.maneuvers[man_id]['finished'] = False
            self.maneuvers[man_id]['finished_time'] = np.inf

        self.heading_maneuver_in_progress = False
        self.speed_maneuver_in_progress = False
        self.altitude_maneuver_in_progress = False

        # initialize PIDs if using autopilot (don't like this being here but don't feel like rewriting this whole thing)
        if self.kinematic_params['autopilot']:
            self.speed_PID = PID.PID(kp=0.8, ki=0.0, kd=0.2)
            self.climb_PID = PID.PID(kp=0.8, ki=0.0, kd=0.2)
            self.vert_accel_PID = PID.PID(kp=0.8, ki=0.0, kd=0.2)
            self.yaw_rate_PID = PID.PID(kp=0.8, ki=0.0, kd=0.2)

        # fill in current-state variables
        self.State = spatial_state.SpatialState()
        self.State.time = self.start_time
        self.State.state = self.initial_state
        self.State.yaw_rate = 0.0
        self.State.pitch_rate = 0.0
        self.State.speed = self.initial_speed
        self.State.yaw = self.initial_heading
        self.State.pitch = self.initial_pitch_angle
        self.State.axial_accel = 0.0
        self.State.lateral_accel = 0.0
        self.State.vertical_accel = 0.0

        self.spatial_states = []  # where generated spatial states will be stored

        self.config_processed = True

        SpatialEarth.log.info('SpatialEarth object {}, number {} config processed.'.format(self.spatial_name, SpatialEarth.spatial_id))


    def generate_traj(self):
        '''
        Generate the SpatialEarth Trajectories. Basically just calls '__propagate' for all times and parameters set up by
        'process_config'

        Args:
            None

        Attributes Created:
            spatial_states (list): spatial states stored after propagation at that time has occured

        Returns:
            None
        '''

        assert self.config_processed, 'SpatialEarth id {} trying to generate trajectory before config is processed'.format(self.spatial_id)

        SpatialEarth.log.info('Generating SpatialEarth object {} trajectory'.format(self.spatial_name))

        for time in self.sample_times:

            # propagate spatial to time t
            self.__propagate(time)

            # append spatial state to list of states
            self.spatial_states.append(self.State)


        # trim to only reported spatial_states
        self.spatial_states = self.spatial_states[0::int(self.update_rate/self.integration_step)]

        self.traj_generated = True
        SpatialEarth.log.info('SpatialEarth object {} trajectory generated'.format(self.spatial_name))


    def __propagate(self, time):
        '''
        Propagate current spatial state to time 'time'. Right now, the maneuvers are treated separately than straight
        propagation. By that I mean the maneuvers don't use a propagation matrix, which doesn't feel right for some
        reason, but it seems to work so I'm going to roll with it for now

        Args:
            time (float): time to propagate current state to

        Attributes Created:
            None, just updates a bunch

        Returns:
            None
        '''

        # make current state a deep copy of the previous state, otherwise we will modify the previously saved states
        self.State = deepcopy(self.State)

        dt = time - self.State.time

        # only propagate if the time given was later than current time
        if dt > 0:

            man_performed = self.__perform_maneuvers(time)

            # if a maneuver was performed, use the new velocity vector to calculate new acceleration and propagate pos.
            # not sure if splitting up maneuver and straight propagation is the right way to do this...
            if man_performed:

                old_spatial_lla = support.xfrm.ecef_to_lla(self.State.get_position())

                # speed, heading, and pitch angle are calculated according to maneuvers, convert them to an enu veloctiy
                course = self.State.get_course()
                new_velocity_enu = support.xfrm.rbe_to_enu(course)

                old_enu_state = support.xfrm.ecef_to_enu(self.State.state, old_spatial_lla)
                old_velocity_enu = np.array([[old_enu_state[1,0]],
                                             [old_enu_state[4,0]],
                                             [old_enu_state[7,0]]])

                # now use the new and old velocity to upate acceleration
                new_accel_enu = (new_velocity_enu - old_velocity_enu) / dt

                # now use new vel to porpogate old position enu
                old_pos_enu = np.array([[old_enu_state[0,0]],
                                        [old_enu_state[3,0]],
                                        [old_enu_state[6,0]]])
                new_pos_enu = old_pos_enu + (new_velocity_enu * dt)
                new_state_enu = np.column_stack((new_pos_enu, new_velocity_enu, new_accel_enu)).reshape((9,1))
                new_state_ecef = support.xfrm.enu_to_ecef(new_state_enu, old_spatial_lla)

                # update current state and speed
                self.State.set_state(new_state_ecef)

            # no maneuver was performed. straight propagation of current state
            else:
                self.State.set_ang_vel(np.zeros((3,1)))
                self.State.set_acceleration(np.zeros((3,1)))

                old_spatial_lla = support.xfrm.ecef_to_lla(self.State.get_position())
                course = self.State.get_course()
                enu_vel = support.xfrm.rbe_to_enu(course)
                enu_vel[2,0] = 0.0
                new_enu_pos_relative_to_body = enu_vel * dt
                new_enu_state = np.column_stack((new_enu_pos_relative_to_body, enu_vel, np.zeros((3,1)))).reshape((9,1))
                new_ecef_state = support.xfrm.enu_to_ecef(new_enu_state, old_spatial_lla)

                self.State.set_state(new_ecef_state)
                #self.State.speed = course[0,0]


            self.State.time = time


    def __perform_maneuvers(self, time):
        ''' Perform a manueuver if deemed necessary. Looks through maneuver dictionary to check if its time to do a
        maneuver and makes sure it hasn't already been performed. Maneuver can have 3 parts: speed change, heading
        change, or pitch/climb angle change. Right now, the total Gs performed by combining types of maneuvers is not
        considered so the total Gs can exceed the individual Gs for each dimension. Don't really like this, but
        what I have now is good enough for my purposes.

        Right now, speed maneuvers are handled first since speed is needed for heading/pitch maneuvers. Not sure if this
        is right (yes, I am unsure of many things)

        Args:
            time (float): time to do maneuver at (if any)

        Attribute Created:
            None

        Returns:
            man_performed (bool): True if a maneuver was performed. Used to determine type of propagation to do.
        '''

        man_performed = False

        for man_id in self.maneuvers:

            man_time = self.maneuvers[man_id]['time']
            man_finished = self.maneuvers[man_id]['finished']
            finished_time = self.maneuvers[man_id]['finished_time']
            if time <= man_time or man_finished or time >= finished_time:
                continue

            man_performed = True  # past the conditions, so a maneuver is going to be performed

            # maneuvers can have speed, heading, and pitch componenets. Right now they are treated individually
            # so the total gs can exceed each individual g component. Don't really like this...
            #!!! Maybe the speed should be updated last? !!!
            parts_finished = []  # keep track of if each part of maneuver is finished
            if 'final_speed' in self.maneuvers[man_id]:
                final_speed = self.maneuvers[man_id]['final_speed']
                if self.kinematic_params['autopilot']:
                    speed_man_finished = self.__speed_autopilot(time, final_speed)
                else:
                    speed_man_finished = self.__speed_maneuver(time, final_speed)
                parts_finished.append(speed_man_finished)

            if 'final_heading' in self.maneuvers[man_id]:
                final_heading = tools.map_0_to_2pi(d2r(self.maneuvers[man_id]['final_heading']))
                if self.kinematic_params['autopilot']:
                    heading_man_finished = self.__heading_change_autopilot(time, final_heading)
                else:
                    heading_man_finished = self.__heading_change_maneuver(time, final_heading)
                parts_finished.append(heading_man_finished)

            # if 'final_pitch_angle' in self.maneuvers[man_id]:
            #     final_pitch_angle = tools.map_pi_to_pi(d2r(self.maneuvers[man_id]['final_pitch_angle']))
            #     pitch_finished = self.__pitch_maneuver(time, final_pitch_angle)
            #     parts_finished.append(pitch_finished)

            if 'final_altitude' in self.maneuvers[man_id]:
                final_altitude = self.maneuvers[man_id]['final_altitude']
                climb_rate = self.maneuvers[man_id]['climb_rate']
                if self.kinematic_params['autopilot']:
                    altitude_finished = self.__altitude_autopilot(time, final_altitude, climb_rate)
                else:
                    altitude_finished = self.__altitude_maneuver(time, final_altitude, climb_rate)
                parts_finished.append(altitude_finished)

            # if all parts are finished, the maneuver is finished
            if all(parts_finished):
                self.maneuvers[man_id]['finished'] = True
                self.maneuvers[man_id]['finished_time'] = time + self.integration_step

        return man_performed


    def __speed_maneuver(self, time, final_speed):
        ''' Based on the current state, max gs set, and our final_speed goal, calculate what our speed will be at 'time'

        Args:
            time (float): time to perform the maneuver through
            final_speed (float): our speed we desire to reach (how close we get this iteration depends on max gs)

        Attribute Created:
            None

        Returns:
            maneuver_finished (bool): If we reached our final speed, mark as complete
        '''

        dt = time - self.State.time
        maneuver_finished = False

        if self.State.speed == final_speed:
            maneuver_finished = True
            return

        max_accel = self.max_gs * 9.81
        self.State.axial_accel = max_accel

        # going up or down?
        speed_increase = True if final_speed > self.State.speed else False

        # going up
        if speed_increase:
            speed = self.State.speed + (self.State.axial_accel * dt)  # new speed (maybe)

            # went up too much, we're at our threshold
            if speed > final_speed:
                speed = final_speed
                maneuver_finished = True
                self.State.axial_accel = 0

            self.State.speed = speed

        # going down
        else:
            speed = self.State.speed - (self.State.axial_accel * dt)  # new speed (maybe)

            # went down too much, we're at our threshold
            if speed < final_speed:
                speed = final_speed
                maneuver_finished = True
                self.State.axial_accel = 0

            self.State.speed = speed

        return maneuver_finished


    def __speed_autopilot(self, time, final_speed):
        ''' Based on the current state, max gs set, and our final_speed goal, calculate what our speed will be at 'time'.
        Uses a PID (really PD) loop and lag filter on axial acceleration to approach desired speed.

        Args:
            time (float): time to perform the maneuver through
            final_speed (float): our speed we desire to reach (how close we get this iteration depends on max gs)

        Attribute Created:
            None

        Returns:
            maneuver_finished (bool): If we reached our final speed, mark as complete
        '''

        # if first step in maneuver, mark as in progress and initialize counter/timer to 0 that tracks if the maneuver is complete
        if self.speed_maneuver_in_progress == False:
            self.speed_maneuver_in_progress = True
            self.speed_within_error_tolerance_duration = 0

        # get time difference since last update and set return flag to default False
        dt = time - self.State.time
        maneuver_finished = False

        # get error between current speed and final speed goal
        current_speed = self.State.speed
        error = final_speed - current_speed

        # if we're within a small fraction of our desired speed, increment the timer that keeps track of how long we've been within the bounds
        if abs(error) < 0.00001 * final_speed:
            self.speed_within_error_tolerance_duration += dt

        # get commanded acceleration from PID loop
        cmd_acc = self.speed_PID.update(dt, error)

        # apply limits to acceleration
        if cmd_acc >= 0:
            cmd_acc = min(self.max_gs*9.81, cmd_acc)
        else:
            cmd_acc = max(-self.max_gs*9.81, cmd_acc)

        # apply time first order filter lag filter
        prev_acc = self.State.axial_accel
        cmd_acc = prev_acc + ( (cmd_acc - prev_acc) * (1-np.exp(-dt/self.accel_time_const)) )

        # propogate speed based on computed acceleration
        new_speed = current_speed + (cmd_acc * dt)

        self.State.speed = new_speed
        self.State.axial_accel = cmd_acc

        # if we've been within our bounds for at least 1 second, we're done. reset everything.
        if self.speed_within_error_tolerance_duration >= 1:
            maneuver_finished = True
            self.speed_maneuver_in_progress = False
            self.speed_within_error_tolerance_duration = 0
            self.speed_PID.reset()
            self.State.speed = final_speed
            self.State.axial_accel = 0.0

        return maneuver_finished


    def __heading_change_maneuver(self, time, final_heading):
        ''' Based on the current state, max gs set, and our final_heading goal, calculate what our heading will
        be at 'time'

        Args:
            time (float): time to perform the maneuver through
            final_heading (float): heading we desire to reach (how close we get this iteration depends on max gs)

        Attribute Created:
            None

        Returns:
            maneuver_finished (bool): If we reached our final heading, mark as complete
        '''

        dt = time - self.State.time
        maneuver_finished = False

        max_lat_accel = self.max_lat_gs * 9.81
        self.State.lateral_accel = max_lat_accel

        if self.State.yaw == final_heading:
            self.State.yaw_rate = 0.0
            maneuver_finished = True
            return maneuver_finished

        # determine direction of turn
        delta = tools.map_pi_to_pi(self.State.yaw - final_heading)
        direction = 'right' if delta < 0 else 'left'

        if direction == 'right':
            yaw_rate = self.State.lateral_accel / self.State.speed
            heading = self.State.yaw + (yaw_rate * dt)

            # too far, set to final
            if heading > final_heading:
                yaw_rate = (final_heading - self.State.yaw) / dt
                heading = final_heading
                maneuver_finished = True
                self.State.lateral_accel = 0

            self.State.yaw_rate = yaw_rate
            self.State.yaw = tools.map_0_to_2pi(heading)

        else:
            yaw_rate = -self.State.lateral_accel / self.State.speed
            heading = self.State.yaw + (yaw_rate * dt)

            # too far, set to final
            if heading < final_heading:
                yaw_rate = (final_heading - self.State.yaw) / dt
                heading = final_heading
                maneuver_finished = True
                self.State.lateral_accel = 0

            self.State.yaw_rate = yaw_rate
            self.State.yaw = tools.map_0_to_2pi(heading)

        return maneuver_finished


    def __heading_change_autopilot(self, time, final_heading):
        ''' Based on the current state, max gs set, and our final_heading goal, calculate what our heading will
        be at 'time'. Uses a PID (really PD) loop and lag filter of lateral acceleration to approach desired heading.

        Args:
            time (float): time to perform the maneuver through
            final_heading (float): heading we desire to reach (how close we get this iteration depends on max gs)

        Attribute Created:
            None

        Returns:
            maneuver_finished (bool): If we reached our final heading, mark as complete
        '''

        # if first step in maneuver, mark as in progress and initialize counter/timer to 0 that tracks if the maneuver is complete
        if self.heading_maneuver_in_progress == False:
            self.heading_maneuver_in_progress = True
            self.heading_within_error_tolerance_duration = 0

        # time diff and default maneuver not finished
        dt = time - self.State.time
        maneuver_finished = False

        # how far off are we from our desired heading?
        current_heading = self.State.yaw
        error = tools.map_pi_to_pi(final_heading - current_heading)

        # if we're within a small fraction of our desired heading, increment the timer that keeps track of how long we've been within the bounds
        if abs(error) < 0.00001 * final_heading:
            self.heading_within_error_tolerance_duration += dt
        else:
            self.heading_within_error_tolerance_duration = 0  # else, reset it

        # update PID to get commanded yaw rate. use it and speed to compute lateral acceleration
        cmd_yaw_rate = self.yaw_rate_PID.update(dt, error)
        speed = self.State.speed
        cmd_lat_acc = cmd_yaw_rate * speed

        # limit our commanded acceleration based on user defined limits
        max_lat_accel = self.max_lat_gs * 9.81
        if cmd_lat_acc > 0:
            cmd_lat_acc = min(cmd_lat_acc, max_lat_accel)
        else:
            cmd_lat_acc = max(cmd_lat_acc, -max_lat_accel)

        # apply time first order filter lag filter to acceleration
        prev_lat_acc = self.State.lateral_accel
        cmd_lat_acc = prev_lat_acc + ( (cmd_lat_acc - prev_lat_acc) * (1-np.exp(-dt/self.accel_time_const)) )

        # update yaw rate and heading
        yaw_rate = cmd_lat_acc / speed
        heading = current_heading + (yaw_rate * dt)

        # save to State
        self.State.yaw_rate = yaw_rate
        self.State.yaw = tools.map_0_to_2pi(heading)
        self.State.lateral_accel = cmd_lat_acc

        # if we've been within our bounds for at least 1 second, we're done. reset everything.
        if self.heading_within_error_tolerance_duration >= 1:
            maneuver_finished = True
            self.heading_maneuver_in_progress = False
            self.heading_within_error_tolerance_duration = 0
            self.yaw_rate_PID.reset()
            self.State.yaw = tools.map_0_to_2pi(final_heading)
            self.State.yaw_rate = 0.0

        return maneuver_finished


    # def __pitch_maneuver(self, time, final_pitch_angle):
    #     ''' Based on the current state, max gs set, and our final_pitch_angle goal, calculate what our angle will
    #     be at 'time'

    #     Args:
    #         time (float): time to perform maneuver through
    #         final_pitch_angle (float): the pitch_angle we're aiming for

    #     Attribute Created:
    #         None

    #     Returns:
    #         maneuver_finished (bool): If we reached our final pitch angle, mark as complete
    #     '''

    #     dt = time - self.State.time
    #     maneuver_finished = False

    #     if self.State.pitch == final_pitch_angle:
    #         self.State.pitch_rate = 0.0
    #         maneuver_finished = True
    #         return maneuver_finished

    #     max_vert_accel = self.max_vert_gs * 9.81

    #     self.State.vertical_accel = max_vert_accel + ( (self.State.vertical_accel - max_vert_accel) * (np.exp(-dt/self.accel_time_const)) )
    #     if self.State.vertical_accel > max_vert_accel:
    #         self.State.vertical_accel = max_vert_accel

    #     # determine direction of turn
    #     delta = tools.map_pi_to_pi(self.State.pitch - final_pitch_angle)
    #     direction = 'down' if delta > 0 else 'up'

    #     if direction == 'up':
    #         pitch_rate = self.State.vertical_accel / self.State.speed
    #         pitch_angle = self.State.pitch + (pitch_rate * dt)

    #         # too far, set to final
    #         if pitch_angle > final_pitch_angle:
    #             pitch_rate = (final_pitch_angle - self.State.pitch) / dt
    #             pitch_angle = final_pitch_angle
    #             maneuver_finished = True
    #             self.State.vertical_accel = 0

    #         self.State.pitch_rate = pitch_rate
    #         self.State.pitch = tools.map_pi_to_pi(pitch_angle)

    #     else:
    #         pitch_rate = -self.State.vertical_accel / self.State.speed
    #         pitch_angle = self.State.pitch + (pitch_rate * dt)

    #         # too far, set to final
    #         if pitch_angle < final_pitch_angle:
    #             pitch_rate = (final_pitch_angle - self.State.pitch) / dt
    #             pitch_angle = final_pitch_angle
    #             maneuver_finished = True
    #             self.State.vertical_accel = 0

    #         self.State.pitch_rate = pitch_rate
    #         self.State.pitch = tools.map_pi_to_pi(pitch_angle)

    #     return maneuver_finished


    def __altitude_maneuver(self, time, final_altitude, climb_rate):
        ''' Perform a change of altitude maneuver, limited by climb_rate.

        Args:
            time (float): time to perform the maneuver through
            final_altitude (float): altitude at end of maneuver (m)
            climb_rate (float): limit maximum climb (or decent) rate

        Attribute Created:
            None

        Returns:
            maneuver_finished (bool): If we reached our final altitude, mark as complete
        '''
        dt = time - self.State.time
        maneuver_finished = False

        if self.altitude_maneuver_in_progress == False:
            self.altitude_maneuver_in_progress = True
            self.altitude_within_error_tolerance_duration = 0

        # get current altitude of state
        current_altitude = support.xfrm.ecef_to_lla(self.State.get_position())[2,0]


        max_vert_accel = self.max_vert_gs * 9.81
        self.State.vertical_accel = max_vert_accel

        alt_error = final_altitude - current_altitude

        if abs(alt_error) < 0.001:
            self.altitude_within_error_tolerance_duration += dt
        else:
            self.altitude_within_error_tolerance_duration = 0

        cvv = alt_error / 1

        vvmax = climb_rate

        if abs(cvv) > vvmax:
            cvv = -vvmax if cvv < 0 else vvmax

        cpitch = np.arcsin(cvv/self.State.speed)

        perror = cpitch - self.State.pitch

        pitch_rate = perror / dt

        #pitchRateLimit = self.max_vert_gs*9.81/self.State.speed
        pitchRateLimit = self.State.vertical_accel/self.State.speed
        if abs(pitch_rate) > pitchRateLimit:
            pitch_rate = pitchRateLimit if pitch_rate > 0 else -pitchRateLimit

        self.State.pitch_rate = pitch_rate
        self.State.pitch = self.State.pitch + (pitch_rate * dt)

        if self.altitude_within_error_tolerance_duration >= 1:
            maneuver_finished = True
            self.altitude_maneuver_in_progress = False
            self.altitude_within_error_tolerance_duration = 0
            self.State.pitch = 0.0
            self.State.pitch_rate = 0.0

        return maneuver_finished


    def __altitude_autopilot(self, time, final_altitude, climb_rate):
        ''' Perform a change of altitude maneuver, limited by climb_rate.
        This is the preferred method of doing altitude maneuvers.

        Args:
            time (float): time to perform the maneuver through
            final_altitude (float): altitude at end of maneuver (m)
            climb_rate (float): limit maximum climb (or decent) rate

        Attribute Created:
            None

        Returns:
            maneuver_finished (bool): If we reached our final altitude, mark as complete
        '''

        # get dt since that update and default maneuver to not finished
        dt = time - self.State.time
        maneuver_finished = False

        # some current attributes we need
        current_pitch = self.State.pitch
        current_altitude = support.xfrm.ecef_to_lla(self.State.get_position())[2,0]
        max_vert_accel = self.max_vert_gs * 9.81
        speed = self.State.speed
        current_alt_rate = speed * np.sin(current_pitch)

        # if this is the first step in the maneuver, mark as in progress and get "timer" going
        if self.altitude_maneuver_in_progress == False:
            self.altitude_maneuver_in_progress = True
            self.altitude_within_error_tolerance_duration = 0  # consecutive time counter that maneuver is within desired bounds

        alt_error = final_altitude - current_altitude

        # if the altitude error is within 1mm, increment timer, else reset it to 0
        if abs(alt_error) <= 0.001:
            self.altitude_within_error_tolerance_duration += dt
        else:
            self.altitude_within_error_tolerance_duration = 0

        # update climb rate PID to get commanded climb rate
        cmd_alt_rate = self.climb_PID.update(dt, alt_error)

        # apply limits based on user specified climb rate
        if cmd_alt_rate >= 0:
            cmd_alt_rate = min(climb_rate, cmd_alt_rate)
        else:
            cmd_alt_rate = max(-climb_rate, cmd_alt_rate)

        # how far out are we of the desired climb rate?
        alt_rate_error = cmd_alt_rate - current_alt_rate

        # how do we need to adjust our up acceleration to achieve the desired climb rate?
        cmd_up_accel = self.vert_accel_PID.update(dt, alt_rate_error)

        # apply limits to vertical acceleration defined in "parameters" keyword
        if cmd_up_accel > 0:
            cmd_up_accel = min(cmd_up_accel, max_vert_accel)
        else:
            cmd_up_accel = max(cmd_up_accel, -max_vert_accel)

        # now we have our final vertical acceleration, use it to compute new pitch rate and pitch
        pitch_rate = cmd_up_accel / speed
        pitch = current_pitch + (pitch_rate * dt)

        # save the states
        self.State.pitch_rate = pitch_rate
        self.State.pitch = pitch

        # if we've been within our error tolerance for at least 1 second, we're done. Reset everything.
        if self.altitude_within_error_tolerance_duration >= 1:
            maneuver_finished = True
            self.altitude_maneuver_in_progress = False
            self.altitude_within_error_tolerance_duration = 0
            self.State.pitch = 0.0
            self.State.pitch_rate = 0.0
            self.climb_PID.reset()
            self.vert_accel_PID.reset()

        return maneuver_finished


    def write_traj(self, fp, **kwargs):
        ''' just writing the current contents of spatial_states to file fp

        Args:
            fp (str): filepath where contents of spatial state will be saved
            numeric (bool): If True, writes our column names and spatial name. If false, completely numeric data file

        Attribute Created:
            None

        Returns:
            None
        '''

        states = self.get_all_states(**kwargs)
        if isinstance(states, pd.core.frame.DataFrame):
            states.to_csv(fp, index=False)
        elif isinstance(state, np.ndarray):
            states.to_csv(fp, header=False, index=False)
        else:
            raise Exception("What type is states? Not DataFrame of ndarray")

        SpatialEarth.log.info('SpatialEarth object {} trajectory written to {}'.format(self.spatial_name, fp))


    def get_all_states(self, numeric=False, add_lla=False, enu_ref_lla=None):
        ''' return all states in numpy array or dataframe form

        Args:
            numeric (bool): if numeric return numpy array, else return dataframe

        Attribute Created:
            None

        Returns:
            all spatial states
        '''

        flattened_states = []
        for State in self.spatial_states:
            flattened_states.append(State.get_full_state_list())

        if not numeric:
            cols = ['time', 'x', 'y', 'z', 'dx', 'dy', 'dz', 'ddx', 'ddy',
                    'ddz', 'speed', 'yaw', 'pitch', 'roll', 'yaw_rate', 'pitch_rate', 'roll_rate', 'axial_accel',
                    'lateral_accel', 'vertical_accel']


            states = DataFrame(flattened_states, columns=cols)
            states.insert(0, 'name', self.spatial_name)
            states.insert(0, 'id', self.spatial_id)

            if add_lla:
                for row in states.itertuples():
                    ecef = np.array([[row.x, row.y, row.z]]).T
                    lla = support.xfrm.ecef_to_lla(ecef)
                    states.loc[row.Index, 'LAT'] = r2d(lla[0,0])
                    states.loc[row.Index, 'LON'] = r2d(lla[1,0])
                    states.loc[row.Index, 'ALT'] = lla[2,0]

            if enu_ref_lla is not None:
                ref_lla = np.zeros((3,1))
                ref_lla[0,0] = d2r(enu_ref_lla[0])
                ref_lla[1,0] = d2r(enu_ref_lla[1])
                ref_lla[2,0] = enu_ref_lla[2]

                for row in states.itertuples():
                    ecef = np.array([[row.x, row.dx, row.ddx, row.y, row.dy, row.ddy, row.z, row.dz, row.ddz]]).T
                    enu = support.xfrm.ecef_to_enu(ecef, ref_lla)
                    states.loc[row.Index, 'east'] = enu[0,0]
                    states.loc[row.Index, 'east_vel'] = enu[1,0]
                    states.loc[row.Index, 'east_accel'] = enu[2,0]
                    states.loc[row.Index, 'north'] = enu[3,0]
                    states.loc[row.Index, 'north_vel'] = enu[4,0]
                    states.loc[row.Index, 'north_accel'] = enu[5,0]
                    states.loc[row.Index, 'up'] = enu[6,0]
                    states.loc[row.Index, 'up_vel'] = enu[7,0]
                    states.loc[row.Index, 'up_accel'] = enu[8,0]

            return states
        else:
            states = DataFrame(flattened_states)
            states.insert(0, 'temp', self.spatial_id)
            states = states.values
            return states


    def get_state(self, prop_time, method='interp', prop_rate=0.005):
        ''' For getting a state at the specified prop time.

        Args:
            prop_time (float): time to get state at
            method (str): 'interp' or 'propagate'. linearly interpolate between states or kinematic propagation
            prop_rate (float): if 'propagate', integration interval to use. smaller is slower but more accurate.

        Returns:
            SpatialState object of the spatial spate of the trajectory at time prop_Time
        '''

        # prop time requested must be a time when spatial existed
        if (prop_time < self.spatial_states[0].time) or (prop_time > self.spatial_states[-1].time):
            return 0

        # linearly interpolate between states at requested time
        if method == 'interp':
            State = spatial_state.SpatialState()
            states = self.get_all_states(numeric=True)
            State.state[0,0] = np.interp(prop_time, states[:,1], states[:,2])
            State.state[3,0] = np.interp(prop_time, states[:,1], states[:,3])
            State.state[6,0] = np.interp(prop_time, states[:,1], states[:,4])
            State.state[1,0] = np.interp(prop_time, states[:,1], states[:,5])
            State.state[4,0] = np.interp(prop_time, states[:,1], states[:,6])
            State.state[7,0] = np.interp(prop_time, states[:,1], states[:,7])
            State.state[2,0] = np.interp(prop_time, states[:,1], states[:,8])
            State.state[5,0] = np.interp(prop_time, states[:,1], states[:,9])
            State.state[8,0] = np.interp(prop_time, states[:,1], states[:,10])
            State.speed = np.interp(prop_time, states[:,1], states[:,11])
            State.yaw = np.interp(prop_time, states[:,1], states[:,12])
            State.pitch = np.interp(prop_time, states[:,1], states[:,13])
            State.roll = np.interp(prop_time, states[:,1], states[:,14])
            State.yaw_rate = np.interp(prop_time, states[:,1], states[:,15])
            State.pitch_rate = np.interp(prop_time, states[:,1], states[:,16])
            State.roll_rate = np.interp(prop_time, states[:,1], states[:,17])
            State.axial_accel = np.interp(prop_time, states[:,1], states[:,18])
            State.lateral_accel = np.interp(prop_time, states[:,1], states[:,19])
            State.vertical_accel = np.interp(prop_time, states[:,1], states[:,20])

            return deepcopy(State)

        # ~more accurate propogation between states
        elif method == 'propagate':

            # get most recent state before time t and set that state as current SpatialState
            for i, State in enumerate(self.spatial_states):
                # compare to state time
                # implemented the isclose because I had issues with the propagated state not matching the interpolated
                # equivalent at times that were like ~0.00000000001 seconds away from a reported state time.
                if np.isclose(prop_time, State.time, atol=1e-07, rtol=0):
                    return State
                elif prop_time > State.time:  # not far enough, keep going
                    continue
                else:  # went too far
                    break

            NextState = deepcopy(State)
            self.State = deepcopy(self.spatial_states[i-1])  # grabbing the one before we went too far

            # reset maneuvers to incomplete in case a time is called where they need to be run
            for man_id in self.maneuvers:
                self.maneuvers[man_id]['finished'] = False

            # 1000 Hz propogation to time t
            for time in np.arange(self.State.time, prop_time+prop_rate, prop_rate):
                if time > prop_time: break
                self.__propagate(time)
                final_time = time

            # just linearly interpolate the last bit
            if final_time < prop_time:
                self.State.time = prop_time
                self.State.state[0,0] = np.interp(prop_time, [self.State.time, NextState.time], [self.State.state[0,0], NextState.state[0,0]])
                self.State.state[1,0] = np.interp(prop_time, [self.State.time, NextState.time], [self.State.state[1,0], NextState.state[1,0]])
                self.State.state[2,0] = np.interp(prop_time, [self.State.time, NextState.time], [self.State.state[2,0], NextState.state[2,0]])
                self.State.state[3,0] = np.interp(prop_time, [self.State.time, NextState.time], [self.State.state[3,0], NextState.state[3,0]])
                self.State.state[4,0] = np.interp(prop_time, [self.State.time, NextState.time], [self.State.state[4,0], NextState.state[4,0]])
                self.State.state[5,0] = np.interp(prop_time, [self.State.time, NextState.time], [self.State.state[5,0], NextState.state[5,0]])
                self.State.state[6,0] = np.interp(prop_time, [self.State.time, NextState.time], [self.State.state[6,0], NextState.state[6,0]])
                self.State.state[7,0] = np.interp(prop_time, [self.State.time, NextState.time], [self.State.state[7,0], NextState.state[7,0]])
                self.State.state[8,0] = np.interp(prop_time, [self.State.time, NextState.time], [self.State.state[8,0], NextState.state[8,0]])
                self.State.speed = np.interp(prop_time, [self.State.time, NextState.time], [self.State.speed, NextState.speed])
                self.State.yaw = np.interp(prop_time, [self.State.time, NextState.time], [self.State.yaw, NextState.yaw])
                self.State.pitch = np.interp(prop_time, [self.State.time, NextState.time], [self.State.pitch, NextState.pitch])
                self.State.roll = np.interp(prop_time, [self.State.time, NextState.time], [self.State.roll, NextState.roll])
                self.State.yaw_rate = np.interp(prop_time, [self.State.time, NextState.time], [self.State.yaw_rate, NextState.yaw_rate])
                self.State.pitch_rate = np.interp(prop_time, [self.State.time, NextState.time], [self.State.pitch_rate, NextState.pitch_rate])
                self.State.roll_rate = np.interp(prop_time, [self.State.time, NextState.time], [self.State.roll_rate, NextState.roll_rate])
                self.State.axial_accel = np.interp(prop_time, [self.State.time, NextState.time], [self.State.axial_accel, NextState.axial_accel])
                self.State.lateral_accel = np.interp(prop_time, [self.State.time, NextState.time], [self.State.lateral_accel, NextState.lateral_accel])
                self.State.vertical_accel = np.interp(prop_time, [self.State.time, NextState.time], [self.State.vertical_accel, NextState.vertical_accel])

            return deepcopy(self.State)

        else:
            raise("Method should be eithr 'propagate' or 'interp'")

    def plot_states(self, directory=None, enu_ref_lla=(0.0, 0.0, 0.0)):
        ''' generate some informative plots about our trajectory

        Args:
            directory (str): if not None, save figures to this location. if None, plot to screen

        Attribute Created:
            None

        Returns:
            None. But either generates some interactive plots or saves to directory
        '''
        from matplotlib.pyplot import subplots, ion, ioff, close, show

        if directory is not None:
            directory = tools.makedir(directory)
            ioff()
        else:
            ion()

        states = self.get_all_states()

        ref_lla = np.zeros((3,1))
        ref_lla[0,0] = d2r(enu_ref_lla[0])
        ref_lla[1,0] = d2r(enu_ref_lla[1])
        ref_lla[2,0] = enu_ref_lla[2]
        for row in states.itertuples():
            ecef = np.array([[row.x, row.y, row.z]]).T
            lla = support.xfrm.ecef_to_lla(ecef)
            states.loc[row.Index, 'LAT'] = r2d(lla[0,0])
            states.loc[row.Index, 'LON'] = r2d(lla[1,0])
            states.loc[row.Index, 'ALT'] = lla[2,0]

            ecef = np.array([[row.x, row.dx, row.ddx, row.y, row.dy, row.ddy, row.z, row.dz, row.ddz]]).T
            enu = support.xfrm.ecef_to_enu(ecef, ref_lla)
            states.loc[row.Index, 'east'] = enu[0,0]
            states.loc[row.Index, 'east_vel'] = enu[1,0]
            states.loc[row.Index, 'east_accel'] = enu[2,0]
            states.loc[row.Index, 'north'] = enu[3,0]
            states.loc[row.Index, 'north_vel'] = enu[4,0]
            states.loc[row.Index, 'north_accel'] = enu[5,0]
            states.loc[row.Index, 'up'] = enu[6,0]
            states.loc[row.Index, 'up_vel'] = enu[7,0]
            states.loc[row.Index, 'up_accel'] = enu[8,0]

        fig1, ax = subplots()
        ax.set_xlabel('EAST')
        ax.set_ylabel('NORTH')
        ax.set_aspect('equal')
        ax.plot(states.east, states.north)

        fig1a, ax = subplots()
        ax.set_xlabel('LON')
        ax.set_ylabel('LAT')
        ax.set_aspect('equal')
        ax.plot(states.LON, states.LAT)

        fig2, ax = subplots()
        ax.set_xlabel('TIME')
        ax.set_ylabel('ALT')
        ax.plot(states.time, states.ALT)

        fig3, (ax1, ax2, ax3, ax4) = subplots(nrows=4, sharex=True)
        ax4.set_xlabel('TIME')
        ax1.set_ylabel('EAST VEL')
        ax2.set_ylabel('NORTH VEL')
        ax3.set_ylabel('UP VEL')
        ax4.set_ylabel('SPEED')
        ax1.plot(states.time, states.east_vel)
        ax2.plot(states.time, states.north_vel)
        ax3.plot(states.time, states.up_vel)
        ax4.plot(states.time, states.speed)

        fig4, (ax1, ax2, ax3, ax4) = subplots(nrows=4, sharex=True)
        ax4.set_xlabel('TIME')
        ax1.set_ylabel('EAST Gs')
        ax2.set_ylabel('NORTH Gs')
        ax3.set_ylabel('UP Gs')
        ax4.set_ylabel('TOTAL Gs')
        ax1.plot(states.time, states.east_accel/9.81)
        ax2.plot(states.time, states.north_accel/9.81)
        ax3.plot(states.time, states.up_accel/9.81)
        ax4.plot(states.time, ((states.east_accel**2 + states.north_accel**2 + states.up_accel**2)**0.5)/9.81)

        fig5, (ax1, ax2) = subplots(nrows=2, sharex=True)
        ax2.set_xlabel('TIME')
        ax1.set_ylabel('HEADING')
        ax2.set_ylabel('PITCH ANGLE')
        ax1.plot(states.time, np.rad2deg(states.yaw))
        ax2.plot(states.time, np.rad2deg(states.pitch))

        if directory is not None:
            fig1.savefig(os.path.join(directory, '{}_east_north.csv'.format(self.spatial_name)))
            fig1a.savefig(os.path.join(directory, '{}_lat_lon.csv'.format(self.spatial_name)))
            fig2.savefig(os.path.join(directory, '{}_time_alt.csv'.format(self.spatial_name)))
            fig3.savefig(os.path.join(directory, '{}_veloctiy.csv'.format(self.spatial_name)))
            fig4.savefig(os.path.join(directory, '{}_acceleration.csv'.format(self.spatial_name)))
            fig5.savefig(os.path.join(directory, '{}_heading_pitch_angle.csv'.format(self.spatial_name)))
            close(fig1)
            close(fig2)
            close(fig3)
            close(fig4)
            close(fig5)
        else:
            show()


    def maneuver_info(self):

        assert self.traj_generated, 'SpatialEarth id {} trying to print maneuver info before trajectory is generated'.format(self.spatial_id)

        s = ''

        for man_id in self.maneuvers:
            start = self.maneuvers[man_id]['time']
            stop = self.maneuvers[man_id]['finished_time']
            finished = self.maneuvers[man_id]['finished']

            states = self.get_all_states(add_lla=True)

            for t in states.time.tolist():
                nearest_reported_time = t
                if t >= stop:
                    break

            states = states[(states.time >= start) & (states.time <= nearest_reported_time)]
            total_gs = max((states.ddx**2 + states.ddy**2 + states.ddz**2)**0.5) / 9.81

            start_state = self.get_state(start)
            start_speed = start_state.speed
            start_heading = r2d(start_state.yaw)
            start_alt = support.xfrm.ecef_to_lla(start_state.get_position())[2,0]
            stop_state = self.get_state(nearest_reported_time)
            stop_speed = stop_state.speed
            stop_heading = r2d(stop_state.yaw)
            stop_alt = support.xfrm.ecef_to_lla(stop_state.get_position())[2,0]

            if finished:
                s += '\n\nManeuver {}\n'.format(man_id)
            else:
                s += '\n\nManeuver {} (Incomplete)\n'.format(man_id)
            s += '\tStart Time: {:.2f}\n'.format(start)
            s += '\tStop Time: {:.2f}\n'.format(nearest_reported_time)
            s += '\tDuration: {:.2f}\n'.format(nearest_reported_time-start)
            s += '\tPeak Total Gs: {:.2f}\n'.format(total_gs)
            s += '\tStart/Stop Speed: {:.2f}/{:.2f}\n'.format(start_speed, stop_speed)
            s += '\tStart/Stop Heading: {:.2f}/{:.2f}\n'.format(start_heading, stop_heading)
            s += '\tStart/Stop Altitude: {:.2f}/{:.2f}\n'.format(start_alt, stop_alt)

        return s


    @staticmethod
    def make_prop_matrix(dt):
        ''' propagation matrix'''

        phi = np.array([[1, dt, (dt**2)/2],
                        [0, 1, dt],
                        [0, 0, 1]])
        phi = np.kron(np.eye(3), phi)
        return phi



if __name__ == '__main__':

    players_path = os.path.realpath(__file__)
    pydarsim_path = players_path[:players_path.rfind('\players')]


    ''' Test Trajectories '''
    config_file = 'straight_and_level'
    # config_file = 'lazy_turn'
    # config_file = 'lazy_turn_autopilot'
    # config_file = 'fast_and_slow'
    # config_file = 'fast_and_slow_autopilot'
    # config_file = 'speedup'
    # config_file = 'speedup_autopilot'
    # config_file = 'turn_and_run_and_dive'
    # config_file = 'turn_and_run_and_dive_autopilot'
    # config_file = 'two_turns'
    # config_file = 'two_turns_autopilot'
    # config_file = 'up_and_down'
    # config_file = 'up_and_down_autopilot'


    ''' Run it '''
    config_fp = os.path.join(pydarsim_path, 'test/test_spatials/{}.yaml'.format(config_file))
    csv_fp = os.path.join(pydarsim_path, 'test/test_spatials/{}.csv'.format(config_file))
    spatial = SpatialEarth(config_fp, process=True)
    states = spatial.get_all_states(add_lla=True, enu_ref_lla=(0.0, 0.0, 0.0))
    spatial.write_traj(csv_fp)
    spatial.plot_states()
    print(spatial)
    print(spatial.maneuver_info())
