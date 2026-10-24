#! /usr/bin/env python3

import os
import sys
import argparse
import traci
import traci.constants as tc
import matplotlib.pyplot as plt
import traci.exceptions
from utils import *
from _controller import *
from _constants import *
import time
import random

class sumo_sim():
    def __init__(self, sumo_config_name):
        self.sumoBinary = "/usr/bin/sumo-gui"
        self.sumoconfig = sumo_config_name
        self.vehID_list = []
        self.num_veh = 0
        self.step = 0
    
    def add_vehicle_to_traffic_fleet(self):
        veh_id = [int(id[3:]) for id in self.vehID_list]
        max_veh_id = np.max(veh_id)
        self.num_veh = len(self.vehID_list) + 1
        try:
            front_state = self.sumo_veh[-1].getVehicleStates()
            if np.abs(front_state[2] - 12) > 0.5 and front_state[2] > 0:
                veh = SUMO_vehicles(vehicle_ID="veh" + str(max_veh_id + 1), init_s=50, init_lane=0, route_ID="route0", lane_change_mode=0, sumo_brake=False)
                self.sumo_veh.append(veh)
        except:
            pass
        
    def init_vehicles_large_map(self, num_vehicle, gap):
        self.num_veh = num_vehicle
        self.sumo_veh = [None]*num_vehicle
        for i in range(int(self.num_veh)):
            self.sumo_veh[i] = SUMO_vehicles(vehicle_ID="veh" + str(i), init_s=1300 - gap*i, init_lane=0, route_ID="route0", lane_change_mode=0, sumo_brake=False)

    def start_Sumo(self):
        sumoCmd = [self.sumoBinary, "-c", self.sumoconfig, "--quit-on-end"]
        traci.start(sumoCmd)
    
    def simulationStepForward(self):
        traci.simulationStep()
        self.vehID_list = traci.vehicle.getIDList()
        self.step += 1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logging_sim", help="whether to save the simulation data", action="store_true")
    parser.add_argument("--num_sv", type=int, default=5.0, help="Number of vehicles in the traffic")
    parser.add_argument("--flow_measure_interval", type=int, default=10.0, help="Number of vehicles in the traffic")
    parser.add_argument("control_type", choices=['NN', 'IDM'], help='Choose control method for traffic vehicles')
    args = parser.parse_args()
    
    # Traffic control setting
    if args.control_type == 'MPC':
        USING_ONLINE_MPC = 1 # If using online MPC to track front vehicle
        USING_NEURAL_NETWORK = 0 # If using neural network controller to track front vehicle
        USING_IDM = 0 # If using IDM to traffic front vehicle
    elif args.control_type == 'NN':
        USING_ONLINE_MPC = 0 # If using online MPC to track front vehicle
        USING_NEURAL_NETWORK = 1 # If using neural network controller to track front vehicle
        USING_IDM = 0 # If using IDM to traffic front vehicle
    else:
        USING_ONLINE_MPC = 0 # If using online MPC to track front vehicle
        USING_NEURAL_NETWORK = 0 # If using neural network controller to track front vehicle
        USING_IDM = 1 # If using IDM to traffic front vehicle
        
    num_veh = args.num_sv
    gap = 1000 / num_veh
    
    veh_sim_t = []
    
    runtime_record = []
    
    current_dirname = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dirname, os.pardir))
    
    sumo_sim_manager = sumo_sim(sumo_config_name=parent_dir + "/sumo/Custom/custom.sumocfg")
    sumo_sim_manager.start_Sumo()
    
    # Initialize controller
    dirname = os.path.dirname(__file__)
    nn_pt_filename = dirname + '/traffic_following_control_dc_trained.pt'
    
    # Setup controller
    if USING_NEURAL_NETWORK:
        FCN_control = NN_controller(nn_pt_file=nn_pt_filename, input_num=3)
        controller_name = 'Neural_Network'
        print('Use neural network to control traffic vehicles')
    elif USING_IDM:
        IDM_control = IDM(a=4, b=5, s0=3, v0=30, T=1)
        controller_name = 'Intelligent_Driving_Model'
        print('Use IDM to control traffic vehicles')
    else:
        print('No controller for all vehicles')
    
    Avg_spd_traffic = []
    Avg_density_traffic = []
    Avg_speed_traffic = []
    Avg_traffic_flow_traffic = []
    Gap_to_front = []
    Ego_speed = []
    
    flow_dt = 0.0
    veh_flow_count = 0
    num_veh_inflow = 0
    
    ego_v = []
    pv_v = []
    inflow_period = gap / 20
    inflow_timer = 0.0

    sumo_sim_manager.init_vehicles_large_map(num_vehicle=num_veh, gap=gap)
    
    while sumo_sim_manager.step * 0.1 < 50:
        sumo_sim_manager.simulationStepForward()
        sim_t = sumo_sim_manager.step * 0.1
        
        # Check if new vehicle is needed to be added
        if inflow_timer < inflow_period:
            inflow_timer += 0.1
        else:
            sumo_sim_manager.add_vehicle_to_traffic_fleet()
            inflow_timer = 0.0
            
        s_vt_traffic = []
        pv_vt_traffic = []
        
        s_st_traffic = []
        pv_st_traffic = []
        
        s_at_traffic = []
        pv_at_traffic = []
        
        start_t = time.time()
        
        num_veh = len(sumo_sim_manager.vehID_list)
        
        for i in range(0, num_veh):
            # Check if vehicle still in the traffic fleet
            if not sumo_sim_manager.sumo_veh[i].ID in sumo_sim_manager.vehID_list:
                sumo_sim_manager.sumo_veh.pop(0)
                continue
            
            if i == 0:
                [veh_1_acc_t, veh_1_spd_t, veh_1_dist_t] = sumo_sim_manager.sumo_veh[i].getVehicleStates()
                veh_0_acc_t = 0.0
                veh_0_spd_t = 30.0
                veh_0_dist_t = veh_1_dist_t + 30
            else:
                [veh_0_acc_t, veh_0_spd_t, veh_0_dist_t] = sumo_sim_manager.sumo_veh[i-1].getVehicleStates()
                [veh_1_acc_t, veh_1_spd_t, veh_1_dist_t] = sumo_sim_manager.sumo_veh[i].getVehicleStates()
                Gap_to_front.append(veh_0_dist_t - veh_1_dist_t)
                Ego_speed.append(veh_1_spd_t)
            
            if USING_NEURAL_NETWORK:
                acc_traffic_step_t = FCN_control.step_forward(s_vt=np.array([veh_1_spd_t]), pv_vt=np.array([veh_0_spd_t]),
                                                            s_st=np.array([veh_1_dist_t]), pv_st=np.array([veh_0_dist_t]),
                                                            s_at=np.array([veh_1_acc_t]), pv_at=np.array([veh_0_acc_t]),
                                                            use_prediction_horizon=True, sim_t=sim_t)
            elif USING_IDM:
                acc_traffic_step_t = IDM_control.IDM_acceleration(front_v=np.array([veh_0_spd_t]),
                                                                  ego_v=np.array([veh_1_spd_t]),
                                                                  front_s=np.array([veh_0_dist_t]),
                                                                  ego_s=np.array([veh_1_dist_t]))
            else:
                acc_traffic_step_t = np.zeros(3)
                
            acc = acc_traffic_step_t[0]
            
            # Assign the acceleration to ego vehicle
            sumo_sim_manager.sumo_veh[i].assignTargetAcceleration(acc, v_max=30)
        
        runtime_record.append(time.time() - start_t)
        
        if args.logging_sim:
            data_logger(sim_t=sim_t, ego_a=veh_1_acc_t, ego_v=veh_1_spd_t, ego_s=veh_1_dist_t,
                        pv_a=veh_0_acc_t, pv_v=veh_0_spd_t, pv_s=veh_0_dist_t, filename= args.leading_speed_profile + "_" + controller_name + ".csv")
        
        # Record the traffic density and traffic flow
        unique_vehicle_id = traci.edge.getLastStepVehicleIDs(edgeID="E2")
        if len(unique_vehicle_id) == 0:
            continue
        edge_len = traci.lane.getLength(laneID="E2_0")
        flow_interval = args.flow_measure_interval
        
        if flow_dt < flow_interval:
            flow_dt += 0.1
        else:
            num_veh_inflow = int(unique_vehicle_id[0][3:]) - veh_flow_count
            veh_flow_count = int(unique_vehicle_id[0][3:])
            flow_dt = 0.0
            Avg_traffic_flow_traffic.append(num_veh_inflow/flow_interval*3600)
            Avg_density_traffic.append(len(unique_vehicle_id)/edge_len*1000)
            Avg_speed_traffic.append(np.mean(Ego_speed))
            print('Traffic flow: ' + str(num_veh_inflow/flow_interval*3600) + " with density of " + str(len(unique_vehicle_id)/edge_len*1000))
        time.sleep(0.01)
    
    traci.close(False)
    
    with open(controller_name + "_" + str(num_veh) + ".csv", 'w') as f:
        csv_writer = csv.writer(f)
        data_x = np.array(Avg_density_traffic)
        data_y = np.array(Avg_traffic_flow_traffic)
        data_z = np.array(Avg_speed_traffic)
        data = np.vstack((data_x[1:], data_y[1:], data_z[1:])).T
        csv_writer.writerow(['Traffic_Density', 'Traffic_Flow', 'Average_Speed'])
        csv_writer.writerows(data)
    
    print('Average runtime is: ', str(round(np.mean(runtime_record) * 1000, 4)), 'ms')
    print('Average speed of traffic vehicles is: ', str(round(np.mean(Ego_speed), 4)), 'm/s')
    print('Runtime standard deviation is: ', str(round(np.std(runtime_record) * 1000, 4)), 'ms')
    
    plt.scatter(Avg_density_traffic, Avg_traffic_flow_traffic, s=15, c='red', label='Gap to front vs Ego speed')
    plt.show(block=False)
    plt.pause(1)
    plt.close()