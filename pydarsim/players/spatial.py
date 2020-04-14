# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:06:33 2020

@author: John
"""

import os
import numpy as np
from numpy import deg2rad as d2r
from numpy import rad2deg as r2d
from pandas import DataFrame
from copy import deepcopy

from pydarsim.support.tools import load_yaml, map_pi_to_pi, map_0_to_2pi, makedir
from pydarsim.support import xfrm
from pydarsim.support.spatial_state import SpatialState

from pdb import set_trace

class Spatial(object):
    '''
    Class for generating simple spatial trajectories with parameters defined in a
    yaml configuration file. Supported maneuvers are simple heading change, speedup,
    and pitch maneuvers.
    
    Issues:
        Check github for issues, there any many
    '''
    
    spatial_num = 0
    
    
    def __init__(self, config_fp=None, process=True):
        
        Spatial.spatial_num = Spatial.spatial_num + 1
        self.spatial_num = Spatial.spatial_num
        
        self.config_loaded = False
        self.config_processed = False
        self.traj_generated = False
        
        if config_fp is not None:
            self.load_config(config_fp, process=process)
    
    
    def __str__(self):
        
        states = self.get_all_states()
        start_time = states.time.min()
        stop_time = states.time.max()
        duration = stop_time - start_time
        max_speed = states.speed.max()
        min_speed = states.speed.min()
        max_gs = max((states.east_accel**2 + states.north_accel**2 + states.up_accel**2)**0.5) / 9.81
        min_heading, max_heading = r2d(states.yaw.min()), r2d(states.yaw.max())
        min_pitch, max_pitch = r2d(states.pitch.min()), r2d(states.pitch.max())
        min_east, max_east = states.east.min(), states.east.max()
        min_north, max_north = states.north.min(), states.north.max()
        min_up, max_up = states.up.min(), states.up.max()
        
        s = ''
        s += "Spatial Name: {}\n".format(self.spatial_name)
        s += 'Spatial ID: {}\n'.format(self.spatial_num)
        
        if self.traj_generated:
            s += 'Start time {:.2f}\n'.format(start_time)
            s += 'Stop time {:.2f}\n'.format(stop_time)
            s += 'Duration {:.2f}\n'.format(duration)
            s += 'Min/Max Speed: ({:.2f}, {:.2f})\n'.format(min_speed, max_speed)
            s += 'Max Total Gs: {:.2f}\n'.format(max_gs)
            s += 'Min/Max Heading: ({:.2f}, {:.2f})\n'.format(min_heading, max_heading)
            s += 'Min/Max Pitch: ({:.2f}, {:.2f})\n'.format(min_pitch, max_pitch)
            s += 'Min/Max East: ({:.2f}, {:.2f})\n'.format(min_east, max_east)
            s += 'Min/Max North: ({:.2f}, {:.2f})\n'.format(min_north, max_north)
            s += 'Min/Max Up: ({:.2f}, {:.2f})\n'.format(min_up, max_up)
        
        return s
            
        
    def load_config(self, spatial_config_yaml_fp, process=True):
        '''
        Load spatial configuration yaml and set object attributes from contents.
        Calls 'process_config' when finished
                
        Args:
            spatial_config_yaml_fp: (str) filepath to config yaml
            
        Attributes Created:
            yaml_fp: (str) filepath to config yaml
            config_yaml: (dict) parsed configuration
            init_params: (dict) initial kinematic parameters for spatial
            kinematic_params: (dict) kinematic limitations and such
            maneuvers: (dict) if exists, set of maneuvers and their specs
            
        Returns:
            None            
        '''
        
        # load the file into a dict
        self.yaml_fp = spatial_config_yaml_fp
        self.config_yaml = load_yaml(self.yaml_fp)

        # check to make sure we have what we need an separate them
        assert('info' in self.config_yaml)
        self.info_params = self.config_yaml['info']
        
        assert('initial' in self.config_yaml)
        self.init_params = self.config_yaml['initial']
        
        assert('parameters' in self.config_yaml)
        self.kinematic_params = self.config_yaml['parameters']
        
        if 'maneuvers' in self.config_yaml and self.config_yaml['maneuvers'] is not None:
            self.maneuvers = self.config_yaml['maneuvers']
        else:
            self.maneuvers = {}
            
        self.config_loaded = True
        
        # process the configuration to pull out and set parameters
        if process:
            self.process_config()
    
    
    def process_config(self):
        '''
        Process the configuration dict and set up attributes for generating the trajectory

        Args:
            None
            
        Attributes Created:
            a lot... don't feel like describing them all. so whats the point of this docstring? ¯\_(ツ)_/¯
            
        Returns:
            None            
        '''
        
        assert(self.config_loaded)
        
        # info parameters
        self.spatial_name = self.info_params['name']
        
        # Generate sampling times
        self.start_time = self.init_params['start_time']
        self.stop_time = self.init_params['stop_time']
        self.update_rate = self.kinematic_params['update_rate']  # update period in seconds
        self.sample_times = np.arange(self.start_time, self.stop_time+self.update_rate, 0.005)
        self.report_times = np.arange(self.start_time, self.stop_time+self.update_rate, self.update_rate)
        
        # unpack max gs info
        self.max_gs = self.kinematic_params['max_gs']
        self.max_lat_gs = self.kinematic_params['max_lateral_gs']
        self.max_vert_gs = self.kinematic_params['max_vertical_gs']
        self.accel_time_const = self.kinematic_params['accel_time_const'] 
        
        # get initial position
        east = self.init_params['east']
        north = self.init_params['north']
        up = self.init_params['up']
        self.initial_position = np.array([[east],
                                          [north],
                                          [up]])
        
        # get initial velocity
        self.initial_speed = self.init_params['speed']
        self.initial_heading = map_0_to_2pi(d2r(self.init_params['heading']))
        self.initial_pitch_angle = map_pi_to_pi(d2r(self.init_params['pitch_angle']))
        self.initial_velocity = xfrm.rbe_to_enu(np.array([[self.initial_speed],
                                                          [self.initial_heading],
                                                          [self.initial_pitch_angle]]), col=True)
        
        # form initial state
        self.initial_acceleration = np.zeros((3,1))
        self.initial_state = np.column_stack((self.initial_position, self.initial_velocity, self.initial_acceleration)).reshape((9,1))
    
    
        # maneuver parameters
        self.initial_yaw_rate = 0.0
        self.initial_pitch_rate = 0.0
        
        # mark maneuvers an incomplete
        for man_id in self.maneuvers:
            self.maneuvers[man_id]['finished'] = False
            self.maneuvers[man_id]['finished_time'] = np.inf
        
        # fill in current-state variables
        self.State = SpatialState()
        self.State.time = self.start_time
        self.State.state = self.initial_state
        self.State.yaw_rate = self.initial_yaw_rate
        self.State.pitch_rate = self.initial_pitch_rate
        self.State.speed = self.initial_speed
        self.State.yaw = self.initial_heading
        self.State.pitch = self.initial_pitch_angle
        self.State.axial_accel = 0.0
        self.State.lateral_accel = 0.0
        self.State.vertical_accel = 0.0
        
        self.spatial_states = []  # where generated spatial states will be stored
        
        self.config_processed = True
              
        # generate trajectory based on configuration
        self.__generate_traj()
        
     
    def __generate_traj(self):
        '''
        Generate the Spatial Trajectories. Basically just calls '__propagate' for all times and parameters set up by
        'process_config'

        Args:
            None
            
        Attributes Created:
            spatial_states (list): spatial states stored after propagation at that time has occured
            
        Returns:
            None        
        '''
        
        assert(self.config_loaded)
        assert(self.config_processed)
        
        for time in self.sample_times:
            
            # propagate spatial to time t
            self.__propagate(time)
            
            # append spatial state to list of states
            self.spatial_states.append(self.State)
            
        
        # trim to only reported spatial_states
        self.spatial_states = self.spatial_states[0::int(200*self.update_rate)]
        
        self.traj_generated = True
        
        
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
                
                # speed, heading, and pitch angle are calculated according to maneuvers, convert them to an enu veloctiy
                new_velocity_enu = xfrm.rbe_to_enu(np.array([[self.State.speed],
                                                              [self.State.yaw],
                                                              [self.State.pitch]]), col=True)
                
                # now use the new and old velocity to upate acceleration
                old_velocity_enu = self.State.get_velocity()
                new_accel_enu = (new_velocity_enu - old_velocity_enu) / dt
                
                # now use new vel to porpogate old position enu
                old_pos_enu = self.State.get_position()
                new_pos_enu = old_pos_enu + (new_velocity_enu * dt)
                
                # update current state and speed
                self.State.set_state(np.column_stack((new_pos_enu, new_velocity_enu, new_accel_enu)).reshape((9,1)))
            
            # no maneuver was performed. straight propagation of current state
            else:
                self.State.set_ang_vel(np.zeros((3,1)))
                self.State.set_acceleration(np.zeros((3,1)))
                prop_matrix = Spatial.make_prop_matrix(dt)  # prop matrix of order 3
            
                self.State.set_state( np.matmul(prop_matrix, self.State.state) )
                
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
            parts_finished = []  # keep track of if each part of maneuver is finished
            if 'final_speed' in self.maneuvers[man_id]:
                final_speed = self.maneuvers[man_id]['final_speed']
                speed_man_finished = self.__speed_maneuver(time, final_speed)
                parts_finished.append(speed_man_finished)
                
            if 'final_heading' in self.maneuvers[man_id]:
                final_heading = map_0_to_2pi(d2r(self.maneuvers[man_id]['final_heading']))
                heading_man_finished = self.__heading_change_maneuver(time, final_heading)
                parts_finished.append(heading_man_finished)
                
            if 'final_pitch_angle' in self.maneuvers[man_id]:
                final_pitch_angle = map_pi_to_pi(d2r(self.maneuvers[man_id]['final_pitch_angle']))
                pitch_finished = self.__pitch_maneuver(time, final_pitch_angle)
                parts_finished.append(pitch_finished)
            
            # if all parts are finished, the maneuver is finished
            if all(parts_finished):
                self.maneuvers[man_id]['finished'] = True
                self.maneuvers[man_id]['finished_time'] = time + 0.005
    
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
        
        # this is technically only part of what I should do. I then need to propagate the state based on this...
        # but the time interval is so short i can probably get away with it for my purposes, though it is not robust
        # (applies to each maneuver)
        self.State.axial_accel = max_accel + ( (self.State.axial_accel - max_accel) * (np.exp(-dt/self.accel_time_const)) )
        if self.State.axial_accel > max_accel:
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
                self.State.axial_accel = 0  # should probably change from this immediate decceleration to a time constant based one
            
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
        
        if self.State.yaw == final_heading:
            self.State.yaw_rate = 0.0
            maneuver_finished = True
            return maneuver_finished
        
        self.State.lateral_accel = max_lat_accel + ( (self.State.lateral_accel - max_lat_accel) * (np.exp(-dt/self.accel_time_const)) )
        if self.State.lateral_accel > max_lat_accel:
            self.State.lateral_accel = max_lat_accel
        
        # determine direction of turn
        delta = map_pi_to_pi(self.State.yaw - final_heading)
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
            self.State.yaw = map_0_to_2pi(heading)
        
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
            self.State.yaw = map_0_to_2pi(heading)
        
        return maneuver_finished
    
    
    def __pitch_maneuver(self, time, final_pitch_angle):
        ''' Based on the current state, max gs set, and our final_pitch_angle goal, calculate what our angle will
        be at 'time'
        
        Args:
            time (float): time to perform maneuver through
            final_pitch_angle (float): the pitch_angle we're aiming for
        
        Attribute Created:
            None
        
        Returns:
            maneuver_finished (bool): If we reached our final pitch angle, mark as complete
        '''
        
        dt = time - self.State.time
        maneuver_finished = False
        
        if self.State.pitch == final_pitch_angle:
            self.State.pitch_rate = 0.0
            maneuver_finished = True
            return maneuver_finished
        
        max_vert_accel = self.max_vert_gs * 9.81
        
        self.State.vertical_accel = max_vert_accel + ( (self.State.vertical_accel - max_vert_accel) * (np.exp(-dt/self.accel_time_const)) )
        if self.State.vertical_accel > max_vert_accel:
            self.State.vertical_accel = max_vert_accel
        
        # determine direction of turn
        delta = map_pi_to_pi(self.State.pitch - final_pitch_angle)
        direction = 'down' if delta > 0 else 'up'
        
        if direction == 'up':
            pitch_rate = self.State.vertical_accel / self.State.speed
            pitch_angle = self.State.pitch + (pitch_rate * dt)
            
            # too far, set to final
            if pitch_angle > final_pitch_angle:
                pitch_rate = (final_pitch_angle - self.State.pitch) / dt
                pitch_angle = final_pitch_angle
                maneuver_finished = True
                self.State.vertical_accel = 0
            
            self.State.pitch_rate = pitch_rate
            self.State.pitch = map_pi_to_pi(pitch_angle)
        
        else:
            pitch_rate = -self.State.vertical_accel / self.State.speed
            pitch_angle = self.State.pitch + (pitch_rate * dt)
            
            # too far, set to final
            if pitch_angle < final_pitch_angle:
                pitch_rate = (final_pitch_angle - self.State.pitch) / dt
                pitch_angle = final_pitch_angle
                maneuver_finished = True
                self.State.vertical_accel = 0
            
            self.State.pitch_rate = pitch_rate
            self.State.pitch = map_pi_to_pi(pitch_angle)
        
        return maneuver_finished
    
    
    def write_traj(self, fp, numeric=False):
        ''' just writing the current contents of spatial_states to file fp
        
        Args:
            fp (str): filepath where contents of spatial state will be saved
            numeric (bool): If True, writes our column names and spatial name. If false, completely numeric data file
        
        Attribute Created:
            None
        
        Returns:
            None
        '''
        
        states = self.get_all_states(numeric)
        if not numeric:          
            states.to_csv(fp)
        else:
            states.to_csv(fp, header=False, index=False)
    
    
    def get_all_states(self, numeric=False):
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
            cols = ['time', 'east', 'north', 'up', 'east_vel', 'north_vel', 'up_vel', 'east_accel', 'north_accel',
                    'up_accel', 'speed', 'yaw', 'pitch', 'roll', 'yaw_rate', 'pitch_rate', 'roll_rate', 'axial_accel',
                    'lateral_accel', 'vertical_accel']
            
            states = DataFrame(flattened_states, columns=cols)
            states.insert(0, 'name', self.spatial_name)
            states.insert(0, 'id', self.spatial_num)
            return states
        else:
            states = DataFrame(flattened_states)
            states.insert(0, 'temp', self.spatial_num)
            states = states.values
            return states
        
    
    def get_state(self, prop_time, method='interp', prop_rate=0.001):
        ''' For getting a state at the specified prop time.
        
        Args:
            prop_time (float): time to get state at
            method (str): 'interp' or 'propagate'. linearly interpolate between states or kinematic propagation
            prop_rate (float): if 'propagate', integration interval to use. smaller is slower but more accurate.
            
        Returns:
            SpatialState object of the spatial spate of the trajectory at time prop_Time
        '''
        
        # prop time requested must be a time when spatial existed
        assert(prop_time >= self.spatial_states[0].time and prop_time <= self.spatial_states[-1].time)
        
        # linearly interpolate between states at requested time
        if method == 'interp':           
            State = SpatialState()
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
    
    def plot_states(self, directory=None):
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
            directory = makedir(directory)
            ioff()
        else:
            ion()
        
        states = self.get_all_states()
        
        fig1, ax = subplots()
        ax.set_xlabel('EAST')
        ax.set_ylabel('NORTH')
        ax.set_aspect('equal')
        ax.plot(states.east, states.north)
        
        fig2, ax = subplots()
        ax.set_xlabel('TIME')
        ax.set_ylabel('UP')
        ax.plot(states.time, states.up)
        
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
            fig2.savefig(os.path.join(directory, '{}_time_up.csv'.format(self.spatial_name)))
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
        
        try:
            assert(self.traj_generated)
        except AssertionError:
            print('Trajectory not generated yet')
            
        s = ''
        
        for man_id in self.maneuvers:
            start = self.maneuvers[man_id]['time']
            stop = self.maneuvers[man_id]['finished_time']
            
            states = self.get_all_states()
            
            for t in states.time.tolist():
                if t < stop:
                    continue
                else:
                    nearest_reported_time = t
                    break
            
            states = states[(states.time >= start) & (states.time <= nearest_reported_time)]
            total_gs = max((states.east_accel**2 + states.north_accel**2 + states.up_accel**2)**0.5) / 9.81
            
            start_state = self.get_state(start)
            start_speed = start_state.speed
            start_heading = r2d(start_state.yaw)
            start_pitch = r2d(start_state.pitch)
            stop_state = self.get_state(nearest_reported_time)
            stop_speed = stop_state.speed
            stop_heading = r2d(stop_state.yaw)
            stop_pitch = r2d(stop_state.pitch)
            
            
            s += '\n\nManeuver {}\n'.format(man_id)
            s += '\tStart Time: {:.2f}\n'.format(start)
            s += '\tStop Time: {:.2f}\n'.format(nearest_reported_time)
            s += '\tDuration: {:.2f}\n'.format(nearest_reported_time-start)
            s += '\tPeak Total Gs: {:.2f}\n'.format(total_gs)
            s += '\tStart/Stop Speed: {:.2f}/{:.2f}\n'.format(start_speed, stop_speed)
            s += '\tStart/Stop Heading: {:.2f}/{:.2f}\n'.format(start_heading, stop_heading)
            s += '\tStart/Stop Pitch: {:.2f}/{:.2f}\n'.format(start_pitch, stop_pitch)
    
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
    config_fp = os.path.join(pydarsim_path, 'test/test_spatials/target1.yaml')
    csv_fp = os.path.join(pydarsim_path, 'test/test_spatials/target1_traj.yaml')
    
    spatial = Spatial(config_fp, process=True)
    states = spatial.get_all_states()
    spatial.write_traj(csv_fp)
    spatial.plot_states()
    print(spatial)
    print(spatial.maneuver_info())
    
    config_fp2 = os.path.join(pydarsim_path, 'test/test_spatials/target1_10Hz.yaml')
    csv_fp2 = os.path.join(pydarsim_path, 'test/test_spatials/target1_10Hz_traj.yaml')
    
    spatial2 = Spatial(config_fp2, process=True)
    states2 = spatial2.get_all_states()
    spatial2.write_traj(csv_fp2)
    spatial2.plot_states()
    print(spatial2)
    print(spatial2.maneuver_info())
    