# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:46:39 2020

@author: John
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Kalman(object):
    ''' tried to use notation convention of Kalman Filter wikipedia entry, except i use k+1 and k, instead of k and k-1 '''
    
    
    def __init__(self, qvar, order=2, ndim=3):
        assert(order==2 or order==3)  # haven't implemented order n yet.
        
        self.i = 0  # updates counter
        self.ndim = ndim  # number of dimensions
        self.order = order  # order of filter (keep it at 2 or 3 please...)
        self.xk = np.zeros((self.ndim*self.order, 1))
        self.Pk = np.zeros((self.ndim*self.order, self.ndim*self.order))
        self.qvar = qvar  # process noise variance
        self.yk1 = np.zeros((self.ndim, 1))
        self.yk1_k1 = np.zeros((self.ndim, 1))
        
        self.states = []  # keep history of states
        
    
    def update(self, tk1, zk1, Rk1):
        ''' tk1: time of update (scalar)
            zk1: measurement vector [ndim x 1]
            Rk1: measurement cov [ndim x ndim]
        '''
        
        assert(zk1.shape == (self.ndim,1))
        assert(Rk1.shape == (self.ndim, self.ndim))
        
        self.i += 1  # increment update counter
        
        # initialize position on first update
        if self.i == 1:
            for j in range(self.ndim):
                self.xk[j*self.order,0] = zk1[j,0]
            
            self.Pk = np.eye(self.ndim*self.order) * 1000000
            for (x,y), value in np.ndenumerate(Rk1):
                self.Pk[x*self.order,y*self.order] = value
                
            if self.order == 3:
                for j in range(self.ndim):
                    self.Pk[j*3+2,j*3+2] = 9.81**2
        
        # # initialize velocity and cov
        # elif self.i == 2:
        #     dt = tk1 - self.tk
            
        #     for j in range(self.ndim):
        #         #self.xk[(j*self.order)+1,0] = ( zk1[j,0] - self.xk[j*self.order,0] ) / dt
        #         #self.xk[(j*self.order)+1,0] = 0
        #         self.xk[j*self.order,0] = zk1[j,0]
            
        #     # not sure about this
        #     # Pk = np.zeros((self.ndim*self.order, self.ndim*self.order))
        #     # for (x,y), value in np.ndenumerate(Rk1):
        #     #     Pk[x*self.order,y*self.order] = value
        #     F = Kalman.make_phi_matrix(dt, self.ndim, self.order)
        #     G = Kalman.make_state_uncert_vector(dt, self.ndim, self.order)
        #     Q = np.matmul(G, G.T) * self.qvar
        #     self.Pk = np.matmul( np.matmul(F, self.Pk), F.T) + Q
                
        # Filter initialized, normal predict/update cycle
        elif self.i >= 2:
        
            ''' PREDICT '''
            
            xk1_k, Pk1_k = self.predict(tk1)
            
            
            ''' UPDATE '''
            
            H = Kalman.make_obs_model_matrix(self.ndim, self.order)  # map state to observation space matrix
            
            # Calculate Kalman Gain
            temp1 = np.matmul(Pk1_k, H.T)
            temp2 = np.matmul(H, temp1) + Rk1
            temp3 = np.linalg.matrix_power(temp2, -1)
            Kk = np.matmul(temp1, temp3)
            
            # calculate innovation
            self.yk1 = zk1 - np.matmul(H, xk1_k)
            
            # update state estimate
            xk1_k1 = xk1_k + np.matmul(Kk, self.yk1)  # xk1_k1 -> x k+1 given k+1, or x k+1 | k+11 notation
            
            # update covariance estimate
            temp1 = np.matmul(Kk, H)
            I = np.eye(*temp1.shape)
            temp2 = I - temp1
            Pk1_k1 = np.matmul(temp2, Pk1_k)
            
            # post-fit residual
            self.yk1_k1 = zk1 - np.matmul(H, xk1_k1)
            
            
            # update object variables
            self.xk = xk1_k1  # new state become old state
            self.Pk = Pk1_k1  # new state cov becomes old state cov
        
        
        # final storage/cleanup
        self.tk = tk1  # new timestamp become old timestamp
        self.states.append((self.i, self.tk, self.xk, self.Pk, self.yk1, self.yk1_k1))
            
        return self.states[-1] 
        

    def predict(self, t):
        ''' propogate existing state
            t: time to propogate to
        '''
        
        i, tk, xk, Pk, _, __ = self.most_recent_update(t)
        
        dt = t - tk
        
        F = Kalman.make_phi_matrix(dt, self.ndim, self.order)  # state propogation matrix
        G = Kalman.make_state_uncert_vector(dt, self.ndim, self.order)  # state uncert vector
        xk_uncert = np.repeat(np.random.normal(0.0, self.qvar**0.5, (self.ndim,1)), self.order, axis=0)  # random acceleration in each dimension to distrub predicted state
        wk = G * xk_uncert  # acceleration disturbance vector
        Q = np.matmul(G, G.T) * self.qvar  # additional uncertainty in state due to acceleration disturbance
        
        # prediction equation (no control vector u or B control-input model present)
        xk1_k = np.matmul(F, xk) + wk  # simple constant velocity propogation with small, random acceleration disturbance
        #xk1_k = np.matmul(F, xk)  # simple constant velocity propogation with small, random acceleration disturbance
        
        # propogate uncertainty
        Pk1_k = np.matmul( np.matmul(F, Pk), F.T) + Q
        
        return xk1_k, Pk1_k
    
    
    def most_recent_update(self, t):
        '''find most recently updated state based on desired time'''
        
        assert(t >= self.states[0][1])  # time must occur on or before first state time
        
        if t >= self.states[-1][1]:  # first check if this is after the most recent state
            return self.states[-1]
        
        else:
            for index, (i, tk, xk, Pk, _, __) in enumerate(self.states):
                if t < tk:
                    return self.states[index-1]  # if t is greater than tk, the last state was the most recent one
        
        raise('What are you doing here?... O.o')
    
    
    @staticmethod
    def make_phi_matrix(dt, num_dim, order):
        ''' propogation matrix
            dt: time difference to propogate over
            num_dim: number of dimensions in observation
            accel: (bool) do accel+vel prop or just vel
        '''

        if order == 3:
            phi = np.array([[1, dt, (dt**2)/2],
                            [0, 1, dt],
                            [0, 0, 1]])
        elif order == 2:
            phi = np.array([[1, dt],
                            [0, 1]])
        else:
            raise ValueError('Must enter order 2 or 3, n not implemented yet... Feel free to do that.')

        phi = np.kron(np.eye(num_dim), phi)

        return phi


    @staticmethod
    def make_obs_model_matrix(num_dim, num_states):
        ''' observation model matrix. maps state to obs space
            num_dim: number of dimension in observation
            num_states: number of unique states per dimension (2 for pos+vel, 3 for pos+vel+accel
        '''
        
        arr1 = np.eye(num_dim)
        arr2 = np.zeros((1, num_states))
        arr2[0,0] = 1
        return np.kron(arr1, arr2)
    
    
    @staticmethod
    def make_state_uncert_vector(dt, num_dim, order):
        ''' make state uncert vector. assume simple acceleration disturbance '''
        
        if order == 2:
            v = np.array([[(dt**2)/2],
                          [dt]])
        elif order == 3:
            v = np.array([[0.5 * (dt**2)],
                          [dt],
                          [1]])
        else:
            raise ValueError('Must enter order 2 or 3, n not implemented yet... Feel free to do that.')
        
        v2 = np.ones((num_dim, 1))
        
        return np.kron(v2, v)




if __name__ == '__main__':
    

    ''' read in detections and filter '''
    
    # det = pd.read_csv("C:/Users/John/Documents/Python Scripts/radar/studies/testing/target_0_to_200_constant_vel_1Hz_detections.csv")
    det = pd.read_csv("C:/Users/John/Documents/Python Scripts/radar/studies/testing/mixed_traj_1_detections_small_err.csv")

    qvar = (9.81 / 50) ** 2
    # qvar = 0
    order=3
    ndim=3
    KF = Kalman(qvar, order=order, ndim=ndim)
    
    for row in det.itertuples():
        print(row.time)
        
        tk1 = row.time
        zk1 = np.array([[row.meas_east],
                        [row.meas_north],
                        [row.meas_up]])
        Rk1 = np.array([[row.EE_cov, row.EN_cov, row.EU_cov],
                        [row.EN_cov, row.NN_cov, row.NU_cov],
                        [row.EU_cov, row.NU_cov, row.UU_cov]])
        i, updateTime, state, cov, inn, res = KF.update(tk1, zk1, Rk1)
        
        if order==2:
            det.loc[row.Index, 'track_east'] = state[0,0]
            det.loc[row.Index, 'track_north'] = state[2,0]
            det.loc[row.Index, 'track_up'] = state[4,0]
            det.loc[row.Index, 'track_east_vel'] = state[1,0]
            det.loc[row.Index, 'track_north_vel'] = state[3,0]
            det.loc[row.Index, 'track_up_vel'] = state[5,0]
            
            det.loc[row.Index, 'Track_EE_cov'] = cov[0,0]
            det.loc[row.Index, 'Track_NN_cov'] = cov[2,2]
            det.loc[row.Index, 'Track_UU_cov'] = cov[4,4]
            det.loc[row.Index, 'Track_dEE_cov'] = cov[1,1]
            det.loc[row.Index, 'Track_dNN_cov'] = cov[3,3]
            det.loc[row.Index, 'Track_dUU_cov'] = cov[5,5]
        elif order==3:
            det.loc[row.Index, 'track_east'] = state[0,0]
            det.loc[row.Index, 'track_north'] = state[3,0]
            det.loc[row.Index, 'track_up'] = state[6,0]
            det.loc[row.Index, 'track_east_vel'] = state[1,0]
            det.loc[row.Index, 'track_north_vel'] = state[4,0]
            det.loc[row.Index, 'track_up_vel'] = state[7,0]
            det.loc[row.Index, 'track_east_accel'] = state[2,0]
            det.loc[row.Index, 'track_north_accel'] = state[5,0]
            det.loc[row.Index, 'track_up_accel'] = state[8,0]
            
            det.loc[row.Index, 'Track_EE_cov'] = cov[0,0]
            det.loc[row.Index, 'Track_NN_cov'] = cov[3,3]
            det.loc[row.Index, 'Track_UU_cov'] = cov[6,6]
            det.loc[row.Index, 'Track_dEE_cov'] = cov[1,1]
            det.loc[row.Index, 'Track_dNN_cov'] = cov[4,4]
            det.loc[row.Index, 'Track_dUU_cov'] = cov[7,7]



    ''' plotting '''
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    
    ax1.set_ylabel('East (m)')
    ax2.set_ylabel('North (m)')
    ax3.set_ylabel('Up (m)')
    ax3.set_xlabel('Time (s)')
    
    # ax1.scatter(det.time, det.east_error, label='meas', c='b', s=3)
    # ax2.scatter(det.time, det.north_error, label='', c='b', s=3)
    # ax3.scatter(det.time, det.up_error, label='', c='b', s=3)
    
    ax1.plot(det.time, det.track_east - det.truth_east, label='track', c='r', lw=2)
    ax2.plot(det.time, det.track_north - det.truth_north, label='', c='r', lw=2)
    ax3.plot(det.time, det.track_up - det.truth_up, label='', c='r', lw=2)
    
    ax1.legend(loc=(1.04, 0.8), ncol=1)



    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    
    ax1.set_ylabel('East Vel (m/s)')
    ax2.set_ylabel('North Vel (m/s)')
    ax3.set_ylabel('Up Vel (m/s)')
    ax3.set_xlabel('Time (s)')
    
    # ax1.scatter(det.time, det.meas_east_vel_error, label='meas', c='b', s=3)
    # ax2.scatter(det.time, det.meas_north_vel_error, label='', c='b', s=3)
    # ax3.scatter(det.time, det.meas_up_vel_error, label='', c='b', s=3)
    
    ax1.plot(det.time, det.track_east_vel - det.truth_east_vel, label='track', c='r', lw=2)
    ax2.plot(det.time, det.track_north_vel - det.truth_north_vel, label='', c='r', lw=2)
    ax3.plot(det.time, det.track_up_vel - det.truth_up_vel, label='', c='r', lw=2)
    
    ax1.legend(loc=(1.04, 0.8), ncol=1)
    
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    
    ax1.set_ylabel('East (m)')
    ax2.set_ylabel('North (m)')
    ax3.set_ylabel('Up (m)')
    ax3.set_xlabel('Time (s)')
    
    ax1.scatter(det.time, det.EE_cov**0.5, label='meas', c='b', s=3)
    ax2.scatter(det.time, det.NN_cov**0.5, label='', c='b', s=3)
    ax3.scatter(det.time, det.UU_cov**0.5, label='', c='b', s=3)
    
    ax1.plot(det.time, det.Track_EE_cov**0.5, label='track', c='r', lw=2)
    ax2.plot(det.time, det.Track_NN_cov**0.5, label='', c='r', lw=2)
    ax3.plot(det.time, det.Track_UU_cov**0.5, label='', c='r', lw=2)
    
    ax1.legend(loc=(1.04, 0.8), ncol=1)
    
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    
    ax1.set_ylabel('East Vel (m/s)')
    ax2.set_ylabel('North Vel (m/s)')
    ax3.set_ylabel('Up Vel (m/s)')
    ax3.set_xlabel('Time (s)')
    
    ax1.plot(det.time, det.Track_dEE_cov**0.5, label='track', c='r', lw=2)
    ax2.plot(det.time, det.Track_dNN_cov**0.5, label='', c='r', lw=2)
    ax3.plot(det.time, det.Track_dUU_cov**0.5, label='', c='r', lw=2)
    
    ax1.legend(loc=(1.04, 0.8), ncol=1)
    
    
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
        
        ax1.set_ylabel('East Accel (m/s2)')
        ax2.set_ylabel('North Accel (m/s2)')
        ax3.set_ylabel('Up Accel (m/s2)')
        ax3.set_xlabel('Time (s)')
        
        ax1.scatter(det.time, det.truth_east_vel.diff() / det.time.diff(), label='meas', c='b', s=3)
        ax2.scatter(det.time, det.truth_north_vel.diff() / det.time.diff(), label='', c='b', s=3)
        ax3.scatter(det.time, det.truth_up_vel.diff() / det.time.diff(), label='', c='b', s=3)
        
        ax1.plot(det.time, det.track_east_accel, label='track', c='r', lw=2)
        ax2.plot(det.time, det.track_north_accel, label='', c='r', lw=2)
        ax3.plot(det.time, det.track_up_accel, label='', c='r', lw=2)
    except:
        pass
    
    

    # ''' testing prediction at higher update rate '''
    
    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    
    # ax1.set_ylabel('East (m)')
    # ax2.set_ylabel('North (m)')
    # ax3.set_ylabel('Up (m)')
    # ax3.set_xlabel('Time (s)')
    
    # ax1.scatter(det.time, det.east_error, label='meas', c='b', s=3)
    # ax2.scatter(det.time, det.north_error, label='', c='b', s=3)
    # ax3.scatter(det.time, det.up_error, label='', c='b', s=3)
    
    # ax1.plot(det.time, det.track_east - det.truth_east, label='track', c='r', lw=2)
    # ax2.plot(det.time, det.track_north - det.truth_north, label='', c='r', lw=2)
    # ax3.plot(det.time, det.track_up - det.truth_up, label='', c='r', lw=2)
    
    # for t in np.arange(0.1, 200.0, 0.1):
    #     state, _ = KF.predict(t)
    
    #     meas_east = state[0,0]
    #     meas_north = state[2,0]
    #     meas_up = state[4,0]
        
    #     truth_east = np.interp(t, det.time, det.truth_east)
    #     truth_north = np.interp(t, det.time, det.truth_north)
    #     truth_up = np.interp(t, det.time, det.truth_up)
        
    #     east_err = meas_east - truth_east
    #     north_err = meas_north - truth_north
    #     up_err = meas_up - truth_up
    
    #     ax1.scatter(t, east_err, c='g', s=5, label='')
    #     ax2.scatter(t, north_err, c='g', s=5)
    #     ax3.scatter(t, up_err, c='g', s=5)
    
        
        