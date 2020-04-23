
import xfrm
import numpy as np
import pandas as pd

update_period = 0.005
end_time = 200
times = np.arange(0, end_time, update_period)

heading         = 0.0      # degrees
speed           = 200.0     # meters per second
start_latitude  = 0.0       # degrees
start_longitude = 0.0       # degrees
altitude        = 1000      # meters

start_lla = np.array([np.radians(start_latitude), np.radians(start_longitude), altitude]).reshape((3,1))
target_ecef = xfrm.lla_to_ecef(start_lla)

hdg_rad = np.radians(heading)
east_vel = speed*np.sin(hdg_rad)
north_vel = speed*np.cos(hdg_rad)
up_vel = 0.0
enu_velocity = np.array([east_vel, north_vel, up_vel]).reshape((3,1))

data = []

for time in times[1:]:
    old_target_lla = xfrm.ecef_to_lla(target_ecef)
    
    new_enu_relative_to_body = update_period*enu_velocity
    
    temp = np.column_stack((new_enu_relative_to_body, enu_velocity, np.zeros((3,1)))).reshape((9,1))
    new_target_ecef = xfrm.enu_to_ecef(temp, old_target_lla)
    
    new_target_lla = xfrm.ecef_to_lla(np.array([[new_target_ecef[0]],
                                                [new_target_ecef[3]],
                                                [new_target_ecef[6]]]))
    data.append((time, new_target_ecef[0], new_target_ecef[3], new_target_ecef[6],
                 new_target_ecef[1], new_target_ecef[4], new_target_ecef[7],
                 new_target_ecef[2], new_target_ecef[5], new_target_ecef[8],
                 np.rad2deg(new_target_lla[0]), np.rad2deg(new_target_lla[1]), new_target_lla[2]))
    

    target_ecef = np.array([[new_target_ecef[0]],
                            [new_target_ecef[3]],
                            [new_target_ecef[6]]])


df = pd.DataFrame(data, columns=['time', 'x', 'y', 'z', 'dx', 'dy', 'dz', 'ddx', 'ddy', 'ddz', 'lat', 'lon', 'alt'])