# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:38:49 2020

@author: John

attempt at ghk filter
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GHK:


    def __init__(self, xp0, t0, use_accel=True, g=None, h=None, k=None, eta=None, dt_limit=None):
        self.xp0 = xp0  # state prediction
        self.t0 = t0  # time of state
        self.r = np.array([[np.nan]])  # residual
        self.i = 0  # number of updates
        self.use_accel = use_accel  # True for alpha-beta-gamma filter (GHK), False for alpha-beta (GH)
        self.states = []  # preserve a list of states at updates
        self.dt_limit = dt_limit

        # gain parameters
        self.g = g
        self.h = h
        self.k = k
        self.eta = eta

        # if eta is specified, override gains to critically dampled filter solution (mahafza, pg 447, 450)
        if eta is not None:
            if self.use_accel:
                self.g = 1 - (self.eta**3)
                self.h = 1.5 * ((1-eta)**2) * (1+eta)
                self.k = (1-eta)**3
            else:
                self.g = 1 - (eta**2)
                self.h = (1-eta)**2
                self.k = None


    def update(self, xm, t1):


        dt = t1 - self.t0  # get time differnece simple last update

        state = np.zeros(self.xp0.shape)  # empty state

        ind_mul = 3 if self.use_accel else 2  # index multiplier depending on gh or ghk

        '''   Initialization Process  (not sure about this...) '''

        # first update
        if self.i == 0:
            self.obs_dim = xm.shape[0]  # number of dimensions in the observation

            # fill in first state positions with obs positions
            for index in range(self.obs_dim):
                state[index*ind_mul, 0] = xm[index, 0]

            # fill in velocities with 0
            for index in range(self.obs_dim):
                state[(index*ind_mul)+1, 0] = 0

            if self.use_accel:
                # fill in accelerations with 0
                for index in range(self.obs_dim):
                    state[(index*ind_mul)+2, 0] = 0

            # update state and push to list
            self.i += 1  # increment update counter
            self.t0 = t1
            self.xp0 = state
            self.states.append((t1, state))  # time and state
            self.r = np.zeros(xm.shape)
            return self.xp0, self.r

        # second update
        elif self.i == 1:

            if self.dt_limit is not None and dt < self.dt_limit:
                return self.xp0, self.r

            assert(xm.shape[0] == self.obs_dim)

            # fill in state positions with obs positions
            for index in range(self.obs_dim):
                state[index*ind_mul, 0] = xm[index, 0]

            # fill in velocities (use previous vel to compute range rate)
            for index in range(self.obs_dim):
                d_pos = state[(index*ind_mul), 0] - self.xp0[(index*ind_mul), 0]
                vel = d_pos / dt
                state[(index*ind_mul)+1, 0] = vel

            if self.use_accel:
                # fill in accelerations with 0
                for index in range(self.obs_dim):
                    state[(index*ind_mul)+2, 0] = 0

            # update state and push to list
            self.i += 1  # increment update counter
            self.t0 = t1
            self.xp0 = state
            self.states.append((t1, state))  # time and state
            self.r = np.zeros(xm.shape)
            return self.xp0, self.r

        # third update
        elif self.i == 2 and self.use_accel:

            if self.dt_limit is not None and dt < self.dt_limit:
                return self.xp0, self.r

            assert(xm.shape[0] == self.obs_dim)

            # fill in state positions with obs positions
            for index in range(self.obs_dim):
                state[index*ind_mul, 0] = xm[index, 0]

            # fill in velocities (use previous vel to compute range rate)
            for index in range(self.obs_dim):
                d_pos = state[(index*ind_mul), 0] - self.xp0[(index*ind_mul), 0]
                vel = d_pos / dt
                state[(index*ind_mul)+1, 0] = vel

            # fill in accelerations
            for index in range(self.obs_dim):
                _, first_state = self.states[-2]
                _, second_state = self.states[-1]

#                num = state[(index*ind_mul), 0] + first_state[(index*ind_mul), 0] - 2*second_state[(index*ind_mul), 0]
#                accel = num / (dt**2)

                d_vel = state[(index*ind_mul)+1, 0] - self.xp0[(index*ind_mul)+1, 0]
                accel = d_vel / dt
                state[(index*ind_mul)+2, 0] = accel

            # update state and push to list
            self.i += 1  # increment update counter
            self.t0 = t1
            self.xp0 = state
            self.states.append((t1, state))  # time and state
            self.r = np.zeros(xm.shape)
            return self.xp0, self.r


        '''   End Initialization Process   '''



        assert(xm.shape[0] == self.obs_dim)  # number of dimensions in observation passed
                                             # needs to be consistent with intial obs.

        self.t0 = t1

        # if dt is small, return to previous state and recompute ghk, choose state with smaller residual (?)
        if self.dt_limit is not None and dt < self.dt_limit:
            prev_dt = 0
            index = -1
            while prev_dt < 0.5:
                try:
                    prev_t, prev_state = self.states[index]
                    prev_dt = t1 - prev_t
                    index -= 1
                except IndexError:
                    # no previous states with dt > 0.5 exist
                    return self.xp0, self.r

            phi = GHK.make_phi_matrix(prev_dt, xm.shape[0], self.use_accel)  # get transition matrix based on dt
            pred_state = np.matmul(phi, prev_state)  # propogate state prediction forward

            if self.use_accel:
                pred_pos = pred_state[0::3]  # get predicted state positions
            else:
                pred_pos = pred_state[0::2]
            new_res = xm - pred_pos  # calculate residual

            if new_res < self.r:
                gain = GHK.make_gain_matrix(prev_dt, self.g, self.h, self.k)
                temp = gain * new_res.flatten()
                temp = np.reshape(temp, (temp.size, 1), order='F')  # stack the columns into nx1 vector
                new_pred = pred_state + temp  # add prediction to gain-adjusted residual

                self.xp0 = new_pred
                self.states.append((t1, self.xp0))
                self.i += 1  # increment update counter
                return self.xp0, self.r

            else:
                return self.xp0, self.r


        else:

            phi = GHK.make_phi_matrix(dt, xm.shape[0], self.use_accel)  # get transition matrix based on dt
            self.xp1 = np.matmul(phi, self.xp0)  # propogate state prediction forward

            if self.use_accel:
                pred_pos = self.xp1[0::3]  # get predicted state positions
            else:
                pred_pos = self.xp1[0::2]
            self.r = xm - pred_pos  # calculate residual

            gain = GHK.make_gain_matrix(dt, self.g, self.h, self.k)
            temp = gain * self.r.flatten()
            temp = np.reshape(temp, (temp.size, 1), order='F')  # stack the columns into nx1 vector
            new_pred = self.xp1 + temp  # add prediction to gain-adjusted residual

            self.xp0 = new_pred  # new prediction becomes old (current)
            self.states.append((t1, self.xp0))
            self.i += 1  # increment update counter


            return self.xp0, self.r


    @staticmethod
    def make_phi_matrix(dt, dim, accel=True):
        ''' propogation matrix'''

        if accel:
            phi = np.array([[1, dt, (dt**2)/2],
                            [0, 1, dt],
                            [0, 0, 1]])
        else:
            phi = np.array([[1, dt],
                            [0, 1]])

        phi = np.kron(np.eye(dim), phi)

        return phi


    @staticmethod
    def make_gain_matrix(dt, g, h, k=None):
        a = g
        b = h/dt

        if k is not None:
            c = (2*k) / (dt**2)

            return np.array([[a],
                             [b],
                             [c]])
        else:
            return np.array([[a],
                             [b]])






''' Testbed '''

if __name__ == '__main__':
    from pdb import set_trace

    samples = pd.read_csv("C:/Users/John/Documents/Python Scripts/radar/studies/testing/mixed_traj_1_detections.csv")
    samples = samples[['time', 'range', 'azimuth', 'elevation', 'truth_range', 'truth_azimuth', 'truth_elevation']]
    samples = samples.sort_values('time').reset_index(drop=True)
    samples['rdot'] = samples.truth_range.diff() / samples.time.diff()

    use_accel = True
    for row in samples.itertuples():
        if row.Index == 0:
            if use_accel:
                initial = np.array([[row.range],
                                    [0],
                                    [0],
                                    [row.azimuth],
                                    [0],
                                    [0],
                                    [row.elevation],
                                    [0],
                                    [0]])
            else:
                initial = np.array([[row.range],
                                    [0],
                                    [row.azimuth],
                                    [0],
                                    [row.elevation],
                                    [0]])
            ghk = GHK(initial, row.time, use_accel=use_accel, eta=0.85)
            continue
        else:
            meas = np.array([[row.range],
                             [row.azimuth],
                             [row.elevation]])
            state, residual = ghk.update(meas, row.time)
            if use_accel:
                samples.loc[row.Index,'pred_range'] = state[0,0]
                samples.loc[row.Index,'pred_az'] = state[3,0]
                samples.loc[row.Index,'pred_el'] = state[6,0]
                samples.loc[row.Index, 'pred_rdot'] = state[1,0]
                samples.loc[row.Index,'range_res'] = residual[0,0]
                samples.loc[row.Index,'az_res'] = residual[1,0]
                samples.loc[row.Index,'el_res'] = residual[2,0]
            else:
                samples.loc[row.Index,'pred_range'] = state[0,0]
                samples.loc[row.Index,'pred_az'] = state[2,0]
                samples.loc[row.Index,'pred_el'] = state[4,0]
                samples.loc[row.Index, 'pred_rdot'] = state[1,0]
                samples.loc[row.Index,'range_res'] = residual[0,0]
                samples.loc[row.Index,'az_res'] = residual[1,0]
                samples.loc[row.Index,'el_res'] = residual[2,0]


    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
    ax1.scatter(samples.time, samples.range-samples.truth_range, s=5, color='b')
    ax2.scatter(samples.time, samples.azimuth-samples.truth_azimuth, s=5, color='b')
    ax3.scatter(samples.time, samples.elevation-samples.truth_elevation, s=5, color='b')
    ax1.plot(samples.time, samples.pred_range-samples.truth_range, c='r')
    ax2.plot(samples.time, samples.pred_az-samples.truth_azimuth, c='r')
    ax3.plot(samples.time, samples.pred_el-samples.truth_elevation, c='r')
    plt.show()

    fig2, (ax4, ax5, ax6) = plt.subplots(nrows=3)
    ax4.plot(samples.time, samples.range_res, color='g')
    ax5.plot(samples.time, samples.az_res, color='g')
    ax6.plot(samples.time, samples.el_res, color='g')
    plt.show()

    fig3, ax7 = plt.subplots()
    ax7.plot(samples.time, (samples.range.diff() / samples.time.diff()) - samples.rdot, color='r')
    ax7.plot(samples.time, (samples.pred_range.diff() / samples.time.diff()) - samples.rdot, color='b')
    ax7.plot(samples.time, samples.pred_rdot - samples.rdot, color='g')
    plt.show()
