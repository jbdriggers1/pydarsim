# -*- coding: utf-8 -*-
'''#############################################################################
    tools.py
        Just a random collections of common functions needed here and there

#############################################################################'''


import os
import numpy as np
import pandas as pd
import yaml
import shutil


def makedir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
        os.chmod(d, 0o777)
    return d


def copy2dir(filepath, directory):
    new_filepath = os.path.join(directory, os.path.basename(filepath))
    _ = makedir(directory)
    shutil.copyfile(filepath, new_filepath)


def round_down(num, divisor):
    return num - (num % divisor)


def sliding_window(data, sample_step, window_width, metric=np.mean):
    sample_col = data.columns[0]
    data_col = data.columns[1]

    data = data.sort_values(by=sample_col)  # meaningless if not sorted

    # get first sample value, rounded down to nearest whole step size
    window_middle = round_down(data[sample_col].min(), sample_step)

    output_df = pd.DataFrame([])
    index = 0
    while window_middle <= data[sample_col].max():


        # get data within [sample_time - 1/2 width, sample_time + 1/2 width)
        window_data = data[ ( data[sample_col] >= (window_middle-(0.5*window_width)) ) &
                            ( data[sample_col] < (window_middle+(0.5*window_width)) ) ]

        agg = window_data[data_col].agg(metric)

        output_df.loc[index, 'sample'] = window_middle
        output_df.loc[index, 'metric'] = agg

        index += 1
        window_middle += sample_step

    return output_df


def load_yaml(fp):
    ''' load the yaml at filepath fp '''

    with open(fp, 'r') as f:
        try:
            yaml_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict


def map_0_to_2pi(angle):
    return (angle + 2 * np.pi) % (2 * np.pi)


def map_pi_to_pi(angle):
    angle = map_0_to_2pi(angle)
    if angle > np.pi:
        angle -= 2*np.pi
    elif angle < -np.pi:
        angle += 2*np.pi
    return angle


def between_angles(a, start, stop):
    ''' check if a is between start and stop angles (radians) '''

    stop = stop - start + 2*np.pi if (stop - start) < 0.0 else stop - start
    a = a - start + 2*np.pi if (a - start) < 0.0 else a - start
    return a <= stop
