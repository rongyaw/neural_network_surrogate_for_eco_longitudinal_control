import math
import csv
import numpy as np
import scipy
import traci
from _controller import IDM

class IDM():
    def __init__(self, a, b, s0, v0, T):
        self.a = a
        self.b = b
        self.s0 = s0
        self.v0 = v0
        self.T = T

    def IDM_acceleration(self, front_v, ego_v, front_s, ego_s):
        s_safe = self.s0 + ego_v * self.T + front_v * \
            (ego_v - front_v) / (2 * (self.a * self.b)**0.5)
        acc = self.a * (1 - (ego_v / self.v0) ** 4 -
                        (s_safe / (front_s - ego_s - 5)) ** 2)
        acc = np.clip(acc, -6, 4)
        return acc

class SUMO_Traffic_Light():
    def __init__(self, s_TL, t_TL, status_TL, red_duration, amber_duration, green_duration):
        self.s_TL = s_TL
        self.status_TL = status_TL
        self.TL_timing = t_TL
        self.RD_duration = red_duration
        self.AM_duration = amber_duration
        self.GR_duration = green_duration
    
    def TL_update(self, TL_id):
        # Update traffic signal timing
        TL_t = self.TL_timing[TL_id]
        # Check the status and duration
        TL_status = self.status_TL[TL_id]
        # Update traffic light status
        if TL_status == 0:
            if TL_t > self.RD_duration:
                TL_status = 2
                TL_t = 0.0
            else:
                TL_status = 0
                TL_t += 0.1
        elif TL_status == 1:
            if TL_t > self.AM_duration:
                TL_status = 0
                TL_t = 0.0
            else:
                TL_status = 1
                TL_t += 0.1
        elif TL_status == 2:
            if TL_t > self.GR_duration:
                TL_status = 1
                TL_t = 0.0
            else:
                TL_status = 2
                TL_t += 0.1
        else:
            print("Something is wrong about this traffic light...(-_-`)")
            
        self.TL_timing[TL_id] = TL_t
        self.status_TL[TL_id] = TL_status

class SUMO_vehicles():
    def __init__(self, vehicle_ID, init_s, init_lane, route_ID, lane_change_mode, sumo_brake):
        self.ID = vehicle_ID
        self.pv_ID = None
        self.v = 0.0
        self.a = 0.0
        self.s = init_s
        self.init_dist = init_s
        self.lane_ID = init_lane
        self.pTL_s = None
        self.pTL_id = None
        self.preview_s = None
        self.preview_v = None
        self.preview_a = None
        self.preview_dt = None
        self.preview_sim_step = None
        self.preview_source = None
        self.preview_num_steps = 0
        self.preview_ref_s = None
        self.preview_ref_v = None
        self.pv_s_prev = None
        self.pv_v_prev = None

        traci.vehicle.add(self.ID, route_ID, typeID = 'electricCar', departLane=str(self.lane_ID), departPos=self.s, departSpeed=0)
        traci.vehicle.setParameter(objectID=self.ID, key='vClass', value='evehicle')
        traci.vehicle.setLaneChangeMode(vehID=self.ID, laneChangeMode=lane_change_mode)
        if not sumo_brake:
            traci.vehicle.setSpeedMode(vehID=self.ID, speedMode=96)
    
    def update_preceding_traffic_light(self, TL_s):
        # Find the traffic that is in front of the traffic light
        [_, _, ego_s] = self.getVehicleStates()
        TL_s = np.array(TL_s)
        TL_id = np.where(TL_s > ego_s)[0]
        if len(TL_id) > 0:
            self.pTL_id = TL_id[0].tolist()
            self.pTL_s = TL_s[TL_id[0]].tolist() - ego_s
        else:
            self.pTL_s = None
            self.pTL_id = None
    
    def getVehicleStates(self):
        veh_v_t = traci.vehicle.getSpeed(vehID=self.ID)
        veh_s_t = traci.vehicle.getDistance(vehID=self.ID) + self.init_dist
        veh_a_t = traci.vehicle.getAcceleration(vehID=self.ID)
        
        return [veh_a_t, veh_v_t, veh_s_t]
    
    def getVehicleTrafficStates(self):
        veh_v_t = traci.vehicle.getSpeed(vehID=self.ID)
        veh_s_t = traci.vehicle.getLanePosition(vehID=self.ID)
        veh_a_t = traci.vehicle.getAcceleration(vehID=self.ID)
        
        return [veh_a_t, veh_v_t, veh_s_t]
    
    def assignTargetSpeed(self, tgt_spd):
        traci.vehicle.setSpeed(vehID=self.ID, speed=tgt_spd)
    
    def assignTargetAcceleration(self, tgt_acc, v_max):
        [_, v, _] = self.getVehicleStates()
        if v - v_max > 0:
            tgt_acc = np.min([tgt_acc, 0.0])
        
        self.a = self.a + 1.0 * (tgt_acc - self.a)
        
        traci.vehicle.setAcceleration(vehID=self.ID, acceleration=self.a, duration=0.1)
    
    def assignLaneChangeMode(self, mode):
        traci.vehicle.setLaneChangeMode(vehID=self.ID, laneChangeMode=mode)
    
    def _normalize_preview_array(self, preview_values, name):
        if preview_values is None:
            return None

        preview_array = np.asarray(preview_values, dtype=float).reshape(-1)
        if preview_array.size == 0:
            raise ValueError(name + " cannot be empty.")
        return preview_array

    def update_vehicle_future_states_preview(
        self,
        pv_s,
        pv_v,
        pv_a=None,
        sim_step=None,
        preview_dt=None,
        source=None,
        is_relative=False,
    ):
        preview_s = self._normalize_preview_array(pv_s, "Preview position array")
        preview_v = self._normalize_preview_array(pv_v, "Preview speed array")

        if preview_s.shape != preview_v.shape:
            raise ValueError("Preview position and speed arrays must have the same shape.")

        if pv_a is not None:
            preview_a = self._normalize_preview_array(pv_a, "Preview acceleration array")
            if preview_a.shape != preview_s.shape:
                raise ValueError(
                    "Preview acceleration array must match the preview position and speed arrays."
                )
        else:
            preview_a = None

        [_, current_v_t, current_s_t] = self.getVehicleStates()
        self.preview_ref_s = current_s_t
        self.preview_ref_v = current_v_t

        if is_relative:
            self.preview_s = preview_s.copy()
            self.preview_v = preview_v.copy()
        else:
            self.preview_s = preview_s.copy() - self.preview_ref_s
            self.preview_v = preview_v.copy() - self.preview_ref_v
        self.preview_a = None if preview_a is None else preview_a.copy()
        self.preview_dt = preview_dt
        self.preview_sim_step = sim_step
        self.preview_source = source
        self.preview_num_steps = self.preview_s.size
        self.pv_s_prev = self.preview_s
        self.pv_v_prev = self.preview_v

    def clear_vehicle_future_states_preview(self):
        self.preview_s = None
        self.preview_v = None
        self.preview_a = None
        self.preview_dt = None
        self.preview_sim_step = None
        self.preview_source = None
        self.preview_num_steps = 0
        self.preview_ref_s = None
        self.preview_ref_v = None
        self.pv_s_prev = None
        self.pv_v_prev = None

    def has_valid_future_states_preview(self, expected_steps=None, sim_step=None):
        if self.preview_s is None or self.preview_v is None:
            return False
        if expected_steps is not None and self.preview_num_steps != expected_steps:
            return False
        if sim_step is not None and self.preview_sim_step is not None:
            return self.preview_sim_step == sim_step
        return True

    def get_vehicle_future_states_preview(self, expected_steps=None, sim_step=None):
        if not self.has_valid_future_states_preview(
            expected_steps=expected_steps,
            sim_step=sim_step,
        ):
            return None, None
        return self.preview_s.copy(), self.preview_v.copy()

    def get_vehicle_future_states_preview_absolute(self, expected_steps=None, sim_step=None):
        preview_s, preview_v = self.get_vehicle_future_states_preview(
            expected_steps=expected_steps,
            sim_step=sim_step,
        )
        if preview_s is None or preview_v is None:
            return None, None
        if self.preview_ref_s is None or self.preview_ref_v is None:
            raise ValueError("Preview reference state is missing for absolute preview reconstruction.")
        return preview_s + self.preview_ref_s, preview_v + self.preview_ref_v

    def predict_constant_acceleration_preview(self, preview_steps, preview_dt, clip_min_speed=0.0):
        [veh_a_t, veh_v_t, veh_s_t] = self.getVehicleStates()

        if preview_steps <= 0:
            raise ValueError("Preview step count must be positive.")
        if preview_dt <= 0:
            raise ValueError("Preview timestep must be positive.")

        preview_a = np.full(preview_steps, veh_a_t, dtype=float)
        preview_v = np.empty(preview_steps, dtype=float)
        preview_s = np.empty(preview_steps, dtype=float)

        predicted_v = veh_v_t
        predicted_s = veh_s_t
        for preview_id in range(preview_steps):
            predicted_v = max(predicted_v + veh_a_t * preview_dt, clip_min_speed)
            predicted_s = predicted_s + predicted_v * preview_dt
            preview_v[preview_id] = predicted_v
            preview_s[preview_id] = predicted_s

        return preview_s, preview_v, preview_a

    def ensure_vehicle_future_states_preview(
        self,
        preview_steps,
        preview_dt,
        sim_step=None,
        source="constant_acceleration",
    ):
        if self.has_valid_future_states_preview(
            expected_steps=preview_steps,
            sim_step=sim_step,
        ):
            return self.preview_s.copy(), self.preview_v.copy()

        preview_s, preview_v, preview_a = self.predict_constant_acceleration_preview(
            preview_steps=preview_steps,
            preview_dt=preview_dt,
        )
        self.update_vehicle_future_states_preview(
            preview_s,
            preview_v,
            pv_a=preview_a,
            sim_step=sim_step,
            preview_dt=preview_dt,
            source=source,
        )
        return self.preview_s.copy(), self.preview_v.copy()

    def build_preview_features_from_preceding_vehicle(
        self,
        preceding_vehicle,
        preview_steps,
        preview_dt,
        sim_step=None,
        use_current_ego_state=True,
    ):
        if preceding_vehicle is None:
            raise ValueError("A preceding vehicle object is required to build preview features.")

        front_preview_s, front_preview_v = preceding_vehicle.get_vehicle_future_states_preview_absolute(
            expected_steps=preview_steps,
            sim_step=sim_step,
        )
        if front_preview_s is None or front_preview_v is None:
            preceding_vehicle.ensure_vehicle_future_states_preview(
                preview_steps=preview_steps,
                preview_dt=preview_dt,
                sim_step=sim_step,
            )
            front_preview_s, front_preview_v = preceding_vehicle.get_vehicle_future_states_preview_absolute(
                expected_steps=preview_steps,
                sim_step=sim_step,
            )

        if use_current_ego_state:
            [_, ego_v_t, ego_s_t] = self.getVehicleStates()
            ego_preview_s = np.full(preview_steps, ego_s_t, dtype=float)
            ego_preview_v = np.full(preview_steps, ego_v_t, dtype=float)
        else:
            ego_preview_s, ego_preview_v = self.get_vehicle_future_states_preview_absolute(
                expected_steps=preview_steps,
                sim_step=sim_step,
            )
            if ego_preview_s is None or ego_preview_v is None:
                self.ensure_vehicle_future_states_preview(
                    preview_steps=preview_steps,
                    preview_dt=preview_dt,
                    sim_step=sim_step,
                )
                ego_preview_s, ego_preview_v = self.get_vehicle_future_states_preview_absolute(
                    expected_steps=preview_steps,
                    sim_step=sim_step,
                )

        distance_headway_preview = front_preview_s - ego_preview_s
        speed_gap_preview = front_preview_v - ego_preview_v
        return distance_headway_preview, speed_gap_preview
    
    def get_electricity_power(self):
        electric_consumption = traci.vehicle.getElectricityConsumption(vehID=self.ID)
        return electric_consumption

def delta_yaw_correction(delta_yaw):
    if delta_yaw > math.pi:
        delta_yaw = delta_yaw - 2*math.pi
    elif delta_yaw < -math.pi:
        delta_yaw = delta_yaw + 2*math.pi
    else:
        delta_yaw = delta_yaw
    return delta_yaw

def vehicle_coordinate_transformation(goal_pose, vehicle_pose):
    dx = goal_pose[0] - vehicle_pose[0]
    dy = goal_pose[1] - vehicle_pose[1]
    v_yaw = delta_yaw_correction(goal_pose[2] - vehicle_pose[2])
    v_x = dx * math.cos(vehicle_pose[2]) + dy * math.sin(vehicle_pose[2])
    v_y = dy * math.cos(vehicle_pose[2]) - dx * math.sin(vehicle_pose[2])
    v_goal_pose = np.array([v_x, v_y, v_yaw])
    return v_goal_pose

def vehicle_coordinate_transformation_3D(goal_pose, vehicle_pose):
    dx = goal_pose[0] - vehicle_pose[0]
    dy = goal_pose[1] - vehicle_pose[1]
    v_x = dx * math.cos(vehicle_pose[3]) + dy * math.sin(vehicle_pose[3])
    v_y = dy * math.cos(vehicle_pose[3]) - dx * math.sin(vehicle_pose[3])
    
    dxdy = (dx**2 + dy**2)**0.5
    dz = goal_pose[2] - vehicle_pose[2]
    v_z = dz * math.cos(vehicle_pose[4]) - dxdy * math.sin(vehicle_pose[4])
    
    v_yaw = delta_yaw_correction(goal_pose[3] - vehicle_pose[3])
    v_pitch = goal_pose[4] - vehicle_pose[4]
    
    return np.array([v_x, v_y, v_z, v_yaw, v_pitch])

def global_path_reader(global_path_name):
    with open(global_path_name) as f:
        path_points = [tuple(line) for line in csv.reader(f)]
    path_points = [(float(point[0]), float(point[1]), float(
        point[2]), float(point[3])) for point in path_points]
    path_points_pos_x = [float(point[0]) for point in path_points]
    path_points_pos_y = [float(point[1]) for point in path_points]
    path_point_pos_yaw = [float(point[2]) for point in path_points]
    path_point_pos_s = [float(point[3]) for point in path_points]
    global_path = np.transpose(np.array(
        [path_points_pos_x, path_points_pos_y, path_point_pos_yaw, path_point_pos_s]))
    return global_path

def global_path_reader2(global_path_name):
    with open(global_path_name) as f:
        path_points = [tuple(line) for line in csv.reader(f)]
    path_points = [(float(point[0]), float(point[1]), \
                    float(point[2]), float(point[3]), float(point[4]), \
                        float(point[5]), float(point[6])) for point in path_points]
    path_points_pos_x = [float(point[1]) for point in path_points]
    path_points_pos_y = [float(point[2]) for point in path_points]
    path_point_pos_yaw = [float(point[4]) for point in path_points]
    path_point_pos_time = [float(point[0]) for point in path_points]
    path_point_pos_velocity = [float(point[5]) for point in path_points]
    
    path_s = []
    path_s.append(0)
    for i in range(1, len(path_points_pos_x)):
        s = path_s[i-1] + np.linalg.norm([(path_points_pos_x[i]-path_points_pos_x[i-1]) , (path_points_pos_y[i]-path_points_pos_y[i-1])])
        path_s.append(s)
        
    global_path = np.transpose(np.array(
        [path_point_pos_time, path_points_pos_x, path_points_pos_y, path_s, path_point_pos_yaw, path_point_pos_velocity]))
    return global_path

def global_path_reader_mixed_reality(global_path_name):
    with open(global_path_name) as f:
        path_points = [tuple(line) for line in csv.reader(f)]
    path_points = [(float(point[0]), float(point[1]), float(
        point[2]), float(point[3]), float(point[4]), float(point[5])) for point in path_points]
    path_points_pos_x = [float(point[0]) for point in path_points]
    path_points_pos_y = [float(point[1]) for point in path_points]
    path_point_pos_z = [float(point[2]) for point in path_points]
    path_point_pos_yaw = [float(point[3]) for point in path_points]
    path_point_pos_pitch = [float(point[4]) for point in path_points]
    path_point_pos_dist = [float(point[5]) for point in path_points]
    global_path = np.transpose(np.array(
        [path_points_pos_x, path_points_pos_y, path_point_pos_z, path_point_pos_yaw, path_point_pos_pitch, path_point_pos_dist]))
    return global_path

def global_spd_profile_reader(speed_profile_name):
    with open(speed_profile_name) as f:
        path_points = [tuple(line) for line in csv.reader(f)]
    path_points = [(float(point[0]), float(point[1])) for point in path_points]
    path_points_t = [float(point[0]) for point in path_points]
    path_points_spd = [float(point[1]) for point in path_points]
    spd_profile = np.transpose(np.array([path_points_t, path_points_spd]))
    return spd_profile

def driving_cycle_spd_profile_reader(driving_cycle_profile_name):
    with open(driving_cycle_profile_name) as f:
        data_pointes = [tuple(line) for line in csv.reader(f)]
    data_pointes = [(float(point[0]), float(point[1]), float(point[2]), float(point[3])) for point in data_pointes]
    data_point_t = [float(point[0]) for point in data_pointes]
    data_point_spd = [float(point[1]) for point in data_pointes]
    data_point_acc = [float(point[2]) for point in data_pointes]
    data_point_dist = [float(point[3]) for point in data_pointes]
    spd_profile = np.transpose(np.array([data_point_t, data_point_spd, data_point_acc, data_point_dist]))
    return spd_profile

def TTCi_estimate(ego_v, front_v, front_s):
    ttc_i = (ego_v - front_v) / front_s
    return ttc_i

def driving_cycle_state_preview_searching(sim_t, record_t, front_v_t, mpc_dt, front_s_init):
    # Find leading vehicle's driving cycle
    cycle_vs = np.empty(32)
    cycle_vs.fill(np.nan)
    
    for i in range(32):
        t_id = np.argmin(np.abs([record_t - (i * mpc_dt + sim_t)]))
        cycle_vs[i] = front_v_t[t_id]
    
    # Keep the position preview the same length as the velocity preview. The
    # generated Eco-MPC interface consumes one reference for every stage.
    cycle_ss = (scipy.integrate.cumulative_trapezoid(
        cycle_vs, dx=mpc_dt, initial=0.0) + front_s_init)
    
    return cycle_vs, cycle_ss

def CBF_acceleration_bound_check(pv_vt, s_vt, pv_st, s_st, tao, alpha, L):
    a_ego_max = (pv_vt - s_vt + alpha * (pv_st - s_st - L - tao * s_vt)) / tao
    return a_ego_max    

def traffic_online_MPC_control_step(veh_0_acc_t, veh_0_spd_t, veh_0_dist_t,
                                    veh_1_acc_t, veh_1_spd_t, veh_1_dist_t,
                                    sim_t, online_MPC_control, record_t, 
                                    front_v_t, mpc_dt, pv_object, ego_object, 
                                    leading_preview=False):
    
    if leading_preview:
        cycle_vs, cycle_ss = driving_cycle_state_preview_searching(sim_t=sim_t, record_t=record_t, front_v_t=front_v_t, mpc_dt=mpc_dt, front_s_init=veh_0_dist_t)
    else:
        cycle_vs = []#pv_object.pv_v_prev
        cycle_ss = []#pv_object.pv_s_prev
        
    veh_1_pred_s, veh_1_pred_v, a_MPC = online_MPC_control.svs.setCommand_SUMO(t = sim_t, ego_s=veh_1_dist_t, ego_v=veh_1_spd_t, ego_a=veh_1_acc_t,
                                                   pv_s=veh_0_dist_t, pv_v=veh_0_spd_t, pv_a=veh_0_acc_t, cycle_ss=cycle_ss, cycle_vs=cycle_vs, cycle_dt=mpc_dt)
    if leading_preview:
        ego_object.update_vehicle_future_states_preview(np.array(veh_1_pred_s) - 5.0, veh_1_pred_v)
    
    # Check if CBF safety constraint is violated
    a_ego_max = CBF_acceleration_bound_check(pv_vt=veh_0_spd_t, s_vt=veh_1_spd_t, pv_st=veh_0_dist_t, s_st=veh_1_dist_t, tao=1.8, alpha=2.0, L=7.0)
    if a_MPC > a_ego_max:
        print(f"CBF safety constraint is violated at time {sim_t}! Adjusting MPC control to ensure safety...")
        a_MPC = np.minimum(a_MPC, a_ego_max)
    
    return [a_MPC]

def engine_power_estimation(ego_v, ego_a):
    m = 2218 # Vehicle weights
    rho = 1.293 # Air density
    Cd = 0.28 # Drag coefficient
    A = 2.84 # Frontal area
    mu = 0.015 # Rolling resistance
    g = 9.8 # Gravity
    
    f_aero = 0.5 * A * Cd * rho * ego_v ** 2
    f_roll = mu * m * g
    f_inertia = ego_a * m
    
    P = (f_aero + f_roll + f_inertia) * ego_v
    P = np.max([P, 0])
    
    return P

def data_logger(sim_t, ego_a, ego_v, ego_s, pv_a, pv_v, pv_s, filename):
    with open(filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow([sim_t, ego_a, ego_v, ego_s, pv_a, pv_v, pv_s])
        
def traffic_density_measurement(lead_s, end_s, num_vehicles):
    rho = num_vehicles / ((lead_s - end_s) / 1000)
    return rho

def average_speed_measurement(veh_v_t):
    avg_veh_v_t = np.mean(np.array(veh_v_t))
    return avg_veh_v_t
