#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_trajectory(file_path):
    """
    Load trajectory data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file containing trajectory data.
    """
    sim_t = []
    a = []
    v = []
    s = []
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split(',')
            sim_t.append(float(values[0]))
            a.append(float(values[1]))
            v.append(float(values[2]))
            s.append(float(values[3]))
    return sim_t, a, v, s

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('leading_speed_profile', choices=['Nyc', 'Hwy', 'Ftp', 'US06','FTPsec1','FTPsec2','FTPsec3'], help='Choose leading vehicles speed profile')
    parser.add_argument("control_type", choices=['Preview', 'Predict','IDM'], help='Choose control method for traffic vehicles')
    
    spd_profile = parser.parse_args().leading_speed_profile
    control_type = parser.parse_args().control_type
    
    mpc_file_name = f"{spd_profile}_{control_type}MPC.csv"
    nn_file_name = f"{spd_profile}_{control_type}NN.csv"

    mpc_sim_t, mpc_a, mpc_v, mpc_s = load_trajectory(mpc_file_name)
    nn_sim_t, nn_a, nn_v, nn_s = load_trajectory(nn_file_name)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(mpc_sim_t, mpc_a, label='MPC Acceleration', color='blue')
    plt.plot(nn_sim_t, nn_a, label='NN Acceleration', color='orange')
    plt.ylabel('Acceleration (m/s²)')
    plt.title(f'Trajectory Comparison: {spd_profile} - {control_type}')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(mpc_sim_t, mpc_v, label='MPC Velocity', color='blue')
    plt.plot(nn_sim_t, nn_v, label='NN Velocity', color='orange')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'Trajectory Comparison: {spd_profile} - {control_type}')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(mpc_sim_t, mpc_s, label='MPC Position', color='blue')
    plt.plot(nn_sim_t, nn_s, label='NN Position', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title(f'Trajectory Comparison: {spd_profile} - {control_type}')
    plt.legend()
    
    plt.show()
