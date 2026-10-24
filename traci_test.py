import os
import sys
import traci
import traci.constants as tc
import matplotlib.pyplot as plt

sumoBinary  = "/usr/bin/sumo-gui"
sumoCmd = [sumoBinary, "-c", "sumo/CMI/cmi.sumocfg"]
veh_0 = 'veh0'
# sumoCmd = [sumoBinary, "-c", "sumo/on_ramp/onramp.sumocfg"]
# veh_0 = 'mainlane.2'

traci.start(sumoCmd)
step = 0

veh_0_dist = []
veh_0_spd = []
veh_0_lane = []
veh_0_acc = []
veh_0_sim_t = []

while step < 10000:
    traci.simulationStep()
    vehicle_list = traci.vehicle.getIDList()
    # Check if vehicle name in the vehicle list
    if veh_0 in vehicle_list:
        veh_0_pos_t = traci.vehicle.getPosition(vehID=veh_0)
        veh_0_spd_t = traci.vehicle.getSpeed(vehID=veh_0)
        veh_0_dist_t = traci.vehicle.getDistance(vehID=veh_0)
        veh_0_lane_t = traci.vehicle.getLaneID(vehID=veh_0)
        veh_0_acc_t = traci.vehicle.getAcceleration(vehID=veh_0)
        veh_0_t = traci.simulation.getTime()
        
        # Record the data
        veh_0_spd.append(veh_0_spd_t)
        veh_0_dist.append(veh_0_dist_t)
        veh_0_lane.append(veh_0_lane_t)
        veh_0_acc.append(veh_0_acc_t)
        veh_0_sim_t.append(veh_0_t)
    step += 1

traci.close(False)

plt.figure(1)
plt.subplot(2,2,1)
plt.plot(veh_0_sim_t, veh_0_spd)
plt.xlabel('Time [s]')
plt.ylabel('Vehicle speed [m/s]')

plt.subplot(2,2,2)
plt.plot(veh_0_sim_t, veh_0_lane)
plt.xlabel('Time [s]')
plt.ylabel('Lane ID')

plt.subplot(2,2,3)
plt.plot(veh_0_sim_t, veh_0_acc)
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s^2]')

plt.subplot(2,2,4)
plt.plot(veh_0_sim_t, veh_0_dist)
plt.xlabel('Time [s]')
plt.ylabel('Distance [m]')
plt.show()