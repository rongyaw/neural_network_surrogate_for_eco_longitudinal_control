import os
import sys
import traci
import traci.constants as tc

sumoBinary  = "/usr/bin/sumo-gui"
sumoCmd = [sumoBinary, "-c", "sumo/on_ramp/onramp.sumocfg"]

traci.start(sumoCmd)
step = 0

while step < 1000:
    traci.simulationStep()
    vehicle_list = traci.vehicle.getIDList()
    step += 1

traci.close(False)