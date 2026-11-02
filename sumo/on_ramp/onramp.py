import traci
import traci.constants as tc
import xml.etree.ElementTree as ET
import os
import sys
import subprocess
import pickle
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_vis as vis
import macro
import onramp_calibrate as onramp
import csv


def run_simulation(readfilename):
    
    traci.start(["sumo", "-c", readfilename+".sumocfg"])

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

    traci.close()


def get_vehicle_ids_from_routes(route_file):
    tree = ET.parse(route_file)
    root = tree.getroot()

    vehicle_ids = []
    for route in root.findall('.//vehicle'):
        vehicle_id = route.get('id')
        vehicle_ids.append(vehicle_id)

    return vehicle_ids





def write_vehicle_trajectories_to_csv(readfilename, writefilename):
    traci.start(["sumo", "-c", readfilename+".sumocfg"])
    
    route_file_path = readfilename+".rou.xml"
    predefined_vehicle_ids = get_vehicle_ids_from_routes(route_file_path)

    print("List of Predefined Vehicle IDs:", predefined_vehicle_ids)

    
    with open(writefilename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Column 1:	Vehicle ID
        # Column 2:	Frame ID
        # Column 3:	Lane ID
        # Column 4:	LocalY
        # Column 5:	Mean Speed
        # Column 6:	Mean Acceleration
        # Column 7:	Vehicle length
        # Column 8:	Vehicle Class ID
        # Column 9:	Follower ID
        # Column 10: Leader ID

        writer.writerow(["VehicleID", "Time", "LaneID", "LocalY", "MeanSpeed", "MeanAccel", "VehLength", "VehClass", "FollowerID", "LeaderID"])
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            simulation_time = traci.simulation.getTime()

            vehicle_ids = traci.vehicle.getIDList()

            for vehicle_id in vehicle_ids:
                position = traci.vehicle.getPosition(vehicle_id)
                laneid = traci.vehicle.getLaneID(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)
                accel = traci.vehicle.getAcceleration(vehicle_id)
                cls = traci.vehicle.getVehicleClass(vehicle_id)

                try:
                    lead_id, lead_dist = traci.vehicle.getLeader(vehicle_id)
                except TypeError:
                    # no leader
                    lead_id = -1
                foll_id, foll_dist = traci.vehicle.getFollower(vehicle_id)
                length = traci.vehicle.getLength(vehicle_id)

                writer.writerow([vehicle_id, simulation_time, laneid, position[0], speed, accel, length, cls, foll_id, lead_id])

            traci.simulationStep()
            step += 1

    traci.close()
    print("Complete!")

    return

def reorder_by_id(trajectory_file, link_names="all", lane_name=""):
    # read the trajectory file to see if it is ordered by time (time increases from previous row)
    # assumption: trajectory_file is either ordered by time or by vehicleID
    # separate files by laneID if bylane is True
    
    prev_time = -1
    ordered_by_time = True
    with open(trajectory_file, mode='r') as file:
        csv_reader = csv.reader(file)
        _ = next(csv_reader)
        for columns in csv_reader:
            try:
                curr_time = float(columns[1]) # int(columns[0])
            except IndexError:
                columns = columns[0].split(',')
                curr_time = float(columns[1]) 
                
            if curr_time < prev_time:
                print(trajectory_file + " is NOT ordered by time")
                ordered_by_time = False
                return
            prev_time = curr_time
    
    # reorder this file by ID
    if ordered_by_time:
        print(trajectory_file + " is ordered by time, reordering it by vehicleID...")
        with open(trajectory_file, mode='r') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)
            rows = [row for row in csv_reader]
        print("Start time: ", rows[0][1])
        print("End time: ", rows[-1][1])
        print("Start pos: ", rows[0][3]) # TODO may not be accurate
        print("End pos: ", rows[-1][3])

        rows.sort(key=lambda x: (x[0], float(x[1])))

            
        if link_names == "all":
            output_file = trajectory_file.replace(".csv", "_byid.csv")
            with open(output_file, mode='w', newline="") as file:
                csv_writer = csv.writer(file)
                file.write(",".join(str(num) for num in headers)+"\n")  # Write the headers
                for row in rows:
                    csv_writer.writerow(row)

        else: # write selected link names
            output_file = trajectory_file.replace(".csv", f"_{lane_name}.csv")
            with open(output_file, mode='w') as file:
                file.write(",".join(str(num) for num in headers)+"\n")  # Write the headers
                for row in rows:
                    lane_id = row[2]
                    if lane_id in link_names:
                        file.write(" ".join(str(num) for num in row)+"\n")

    return

if __name__ == "__main__":
    reorder_by_id(trajectory_file="trajs_gt.csv")
