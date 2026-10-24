################################
### Tyler Ard                ###
### Argonne National Lab     ###
### Vehicle Mobility Systems ###
### tard(at)anl(dot)gov      ###
################################

from math import sin, cos, sqrt, tan, atan, atan2, fmod, pi, inf, nan

import os
import numpy as np
import pandas as pd
import scipy

from _constants import *
from _sensor import *
from _cppwrapper import cpp_api

class _vehicle:
    '''Virtual vehicle super class'''
    id = 1

    def __init__(self, s, v, a):
        # Vehicle states
        self.s = s # Rear axis longitudinal position from origin of road
        self.v = v # Forward velocity
        self.a = a # Forward acceleration

        # Vehicle characteristics
        self.len = VEHLENGTH # Length of vehicle from axle to axle
        self.wid = VEHWIDTH # Width of vehicle
        
        self.id = _vehicle.id # Unique vehicle ID
        _vehicle.id += 1

        self.type = VEHTYPE.NONE # Vehicle type

        # Control
        self.v_max = nan

        self.exitflag = 1

        # Sensor
        self.dgap = 2000.

class PCC(_vehicle):
    '''CAV driver for virtual automated vehicles'''
    def __init__(self, dirname, s=0., v=0., a=0., v_max = 10.):
        super().__init__(s, v, a)

        self.type = VEHTYPE.PCC

        # Parameters
        self.v_max = v_max # Max allowed velocity [m/s]

        # Controls
        self.ua = 0. # Acceleration command

        # Autogen Simulink Model Interface
        if os.name == 'nt':
            libraryname = 'pcc_so'
        elif os.name == 'posix':
            libraryname = 'libpcc_so'
        else:
            raise ValueError('PCC class cannot determine system type!')
            
        self.api = cpp_api(dirname + '/' + libraryname)

    def predAcc(self, t, pv_state, v_max):
        dt_pred = 0.1 # Time stepsize between prediction stages [s]
        t_pred = t # [s]
        
        k = 0 # First index is current PV states
        self.api.inputs_p.contents.acc_pred[k] = pv_state[2]
        self.api.inputs_p.contents.vel_pred[k] = pv_state[1]
        self.api.inputs_p.contents.pos_pred[k] = pv_state[0]
        self.api.inputs_p.contents.time_pred[k] = t_pred

        n_pred_steps = 32 # Match the 32-stage generated Eco-MPC interface.
        for k in range(1, n_pred_steps): # Future indices are predicted PV states -
            # Logic to prevent overspeeding and reversing
            if pv_state[1] > v_max:
                pv_state[2] = 0.

            elif pv_state[1] < 0:
                pv_state[2] = 0.
                pv_state[1] = 0.

            # Propagate kinematic constant acceleration prediction
            t_pred += dt_pred

            self.api.inputs_p.contents.acc_pred[k] = pv_state[2]
            self.api.inputs_p.contents.vel_pred[k] = max(pv_state[1] + dt_pred*pv_state[2], 0)
            self.api.inputs_p.contents.pos_pred[k] = pv_state[0] + dt_pred*(max(pv_state[1] + dt_pred*pv_state[2], 0) + dt_pred*pv_state[2] )
            self.api.inputs_p.contents.time_pred[k] = t_pred

            pv_state[0] = self.api.inputs_p.contents.pos_pred[k]
            pv_state[1] = self.api.inputs_p.contents.vel_pred[k]

    def setPred(self, t, pv_state, cycle_ss, cycle_vs, cycle_dt, n_pred_steps=50):
        dt_pred = cycle_dt # Time stepsize between prediction stages [s]
        t_pred = t # [s]

        k = 0 # First index is current PV states
        self.api.inputs_p.contents.acc_pred[k] = pv_state[2]
        self.api.inputs_p.contents.vel_pred[k] = pv_state[1]
        self.api.inputs_p.contents.pos_pred[k] = pv_state[0]
        self.api.inputs_p.contents.time_pred[k] = t_pred

        n_pred_steps = min(
            int(n_pred_steps), len(cycle_vs) + 1, len(cycle_ss) + 1, 32
        )
        for k in range(1, n_pred_steps): # Future indices are predicted PV states -
            # Include prediction from external module
            t_pred += dt_pred

            self.api.inputs_p.contents.acc_pred[k] = pv_state[2]
            self.api.inputs_p.contents.vel_pred[k] = cycle_vs[k-1]
            self.api.inputs_p.contents.pos_pred[k] = cycle_ss[k-1]
            
            self.api.inputs_p.contents.time_pred[k] = t_pred
    
    def setCommand_SUMO(self, t, ego_s, ego_v, ego_a, pv_s, pv_v, pv_a, cycle_ss, cycle_vs, cycle_dt, pv_ind=0):
        '''Set the control commands, for example desired acceleration and desired lane'''
        # Controller parameters
        s_max = 50000 # Max position [m]
        v_max = self.v_max # Max velocity [m/s]
        
        ### Assign inputs struct properties
        ### The goal here is to collect vehicle sensor signals and pack them into inputs_p
        self.api.inputs_p.contents.t = t # Dereference pointer with .contents method
        
        # Ego vehicle states
        self.api.inputs_p.contents.ego_state[0] = ego_s + 7.0 # self.s + self.len # MPC wants the Frenet front bumper position - the simulation was written so that .s is the back bumper position for each simulated vehicle so add vehicle len to get front bumper
        self.api.inputs_p.contents.ego_state[1] = ego_v # self.v # Frenet forward velocity
        self.api.inputs_p.contents.ego_state[2] = ego_a # self.a # Frenet forward acceleration - use previous Ua command if unknown/very inaccurate ego accel
        
        pv_state = [None]*3
        if pv_ind >= 0:
            pv_state[0] = pv_s # Expects Frenet back bumper position - Front Bumper to back bumper Gap from sensor + Ego S
            pv_state[1] = pv_v # Forward velocity
            pv_state[2] = pv_a # Forward acceleration

        else:
            pv_state[0] = pv_s # pv s
            pv_state[1] = 0 # pv v
            pv_state[2] = 0 # pv a

        self.dgap = pv_s - ego_s

        # Reset the prediction inputs - just in case it is needed
        c_array_len = 201 # The total len of the c array - match with the _cppwrapper.py and their equivalent definitions in the EXTU struct found in longitudinal_mpc.h
        for k in range(0, c_array_len):
            self.api.inputs_p.contents.acc_pred[k] = nan
            self.api.inputs_p.contents.vel_pred[k] = nan
            self.api.inputs_p.contents.pos_pred[k] = nan
            self.api.inputs_p.contents.time_pred[k] = nan

        # Predict PV motion and then write to inputs
        if USING_PREVIEW:
            self.setPred(t=t, pv_state=pv_state, cycle_ss=cycle_ss, cycle_vs=cycle_vs, cycle_dt=cycle_dt, n_pred_steps=50)
        else:
            self.predAcc(t=t, pv_state=pv_state, v_max=35)
        
        # Ego vehicle state constraints
        self.api.inputs_p.contents.pos_max = s_max
        self.api.inputs_p.contents.vel_max = v_max

        ### Run the autogen controller
        self.api.step_inputs() # Reads inputs_p and writes them to controller
        self.api.step_controller() # Steps the controller
        self.api.step_outputs() # Writes the outputs from controller to outputs_p

        ### Assign outputs struct properties
        acc_des = self.api.outputs_p.contents.acc_des

        state_trajectory = self.api.outputs_p.contents.state_trajectory
        # control_trajectory = self.api.outputs_p.contents.control_trajectory
        # time_trajectory = self.api.outputs_p.contents.time_trajectory
        # slacks = self.api.outputs_p.contents.slacks
        # reference = self.api.outputs_p.contents.reference
        # constraint = self.api.outputs_p.contents.constraint
        # cost = self.api.outputs_p.contents.cost
        self.exitflag = self.api.outputs_p.contents.exitflag

        ### Log the inputs to the MPC inputs_p.contents here
        ### Log the outputs from the MPC outputs_p.contents here

        ### Assign the control
        # Commanded immediate acceleration
        self.ua = acc_des # Desired acceleration [m/s2]
        
        n_states = 3 # Number of states in the MPC - fixed number

        pos_traj = state_trajectory[0::n_states] # Pos state starts at index 0
        vel_traj = state_trajectory[1::n_states] # Vel state starts at index 1
        acc_traj = state_trajectory[2::n_states] # Acc state starts at index 2
        
        return pos_traj, vel_traj, acc_des

    def setCommand(self, nvs, t):
        '''Set the control commands, for example desired acceleration and desired lane'''
        # Controller parameters
        s_max = 5000 # Max position [m]
        v_max = self.v_max # Max velocity [m/s]
        
        ### Assign inputs struct properties
        ### The goal here is to collect vehicle sensor signals and pack them into inputs_p
        self.api.inputs_p.contents.t = t # Dereference pointer with .contents method
        
        # Ego vehicle states
        self.api.inputs_p.contents.ego_state[0] = self.s + self.len # MPC wants the Frenet front bumper position - the simulation was written so that .s is the back bumper position for each simulated vehicle so add vehicle len to get front bumper
        self.api.inputs_p.contents.ego_state[1] = self.v # Frenet forward velocity
        self.api.inputs_p.contents.ego_state[2] = self.a # Frenet forward acceleration - use previous Ua command if unknown/very inaccurate ego accel
        
        # Get PV by sensing
        pv_ds, pv_v, pv_a, pv_ind = getFrontVeh(nvs, self)
        
        pv_state = [None]*3
        if pv_ind >= 0:
            pv_state[0] = pv_ds + self.s # Expects Frenet back bumper position - Front Bumper to back bumper Gap from sensor + Ego S
            pv_state[1] = pv_v # Forward velocity
            pv_state[2] = pv_a # Forward acceleration

        else:
            pv_state[0] = pv_ds + self.s # pv s
            pv_state[1] = 0 # pv v
            pv_state[2] = 0 # pv a

        self.dgap = pv_ds

        # Reset the prediction inputs - just in case it is needed
        c_array_len = 201 # The total len of the c array - match with the _cppwrapper.py and their equivalent definitions in the EXTU struct found in longitudinal_mpc.h
        for k in range(0, c_array_len):
            self.api.inputs_p.contents.acc_pred[k] = nan
            self.api.inputs_p.contents.vel_pred[k] = nan
            self.api.inputs_p.contents.pos_pred[k] = nan
            self.api.inputs_p.contents.time_pred[k] = nan

        # Predict PV motion and then write to inputs
        if USING_PRED:
            cycle_ss, cycle_vs = nvs[0].getPrediction(t)
            self.setPred(t, pv_state, cycle_ss, cycle_vs)

        elif USING_PREVIEW:
            cycle_ss, cycle_vs = nvs[0].getPreview(t)
            self.setPred(t, pv_state, cycle_ss, cycle_vs)

        else:
            self.predAcc(t, pv_state, v_max)
        
        # Ego vehicle state constraints
        self.api.inputs_p.contents.pos_max = s_max
        self.api.inputs_p.contents.vel_max = v_max

        ### Run the autogen controller
        self.api.step_inputs() # Reads inputs_p and writes them to controller
        self.api.step_controller() # Steps the controller
        self.api.step_outputs() # Writes the outputs from controller to outputs_p

        ### Assign outputs struct properties
        acc_des = self.api.outputs_p.contents.acc_des

        # state_trajectory = self.api.outputs_p.contents.state_trajectory
        # control_trajectory = self.api.outputs_p.contents.control_trajectory
        # time_trajectory = self.api.outputs_p.contents.time_trajectory
        # slacks = self.api.outputs_p.contents.slacks
        # reference = self.api.outputs_p.contents.reference
        # constraint = self.api.outputs_p.contents.constraint
        # cost = self.api.outputs_p.contents.cost
        self.exitflag = self.api.outputs_p.contents.exitflag

        ### Log the inputs to the MPC inputs_p.contents here
        ### Log the outputs from the MPC outputs_p.contents here

        ### Assign the control
        # Commanded immediate acceleration
        self.ua = acc_des # Desired acceleration [m/s2]

        # # Expected state trajectory from the MPC model
        # # state_trajectory is a 1D list containing the states [s v a] for stages 1..N - stage 0 would be the current ego vehicle states, stage 1 is immediate predicted states, stage 2 is two-step predicted states, etc
        # n_states = 3 # Number of states in the MPC - fixed number

        # pos_traj = state_trajectory[0::n_states] # Pos state starts at index 0
        # vel_traj = state_trajectory[1::n_states] # Vel state starts at index 1
        # acc_traj = state_trajectory[2::n_states] # Acc state starts at index 2

        # # Slack variables
        # # We can monitor the slack variables to see if the MPC feels safe in the current situation
        # # Slack is the margin that the solver reduced the constraint so that a feasible solution could be found
        # max_s_slack = slacks[0] # s + e \leq s_max
        # min_v_slack = slacks[1] # v_min \leq v + e
        # max_v_slack = slacks[2] # v + e \leq v_max
        # gap_viol_slack = slacks[3] # (s+vTh) + e \leq P(s_pv)(confidence) - d_min

        # # These slack variables will naturally have some non-zero value occasionally to help with solution smoothness
        # # Let's monitor and if the MPC feels it cannot stop the vehicle before reaching the PV, a warning to the safety drivers goes off like so:
        # SAFETY_MARGIN = 2 # "Allowable" margin before flashing a warning [m] We should discuss on what to set this value as
        # if gap_viol_slack > SAFETY_MARGIN:
        #     print("MPC DETECTING GAP VIOLATIONS ARE OCCURING")

        # # Exitflags
        # # Exitflag is the MPC optimizer exitflag that indicates the quality of the solution
        # # Their codes can be found in longitudinal_mpc_types.h:
        # # enum class flags
        # #     : int32_T {
        # #     SOLVED = 1,                          /* Default value */
        # #     MAX_ITER = 0,
        # #     PRIMAL_INFEASIBLE = -2,
        # #     DUAL_INFEASIBLE = -3,
        # #     NONCONVEX_DETECTED = -6,
        # #     MI_SOLVED = 11,
        # #     MI_PRIMAL_INFEASIBLE = 13,
        # #     MI_DUAL_INFEASIBLE = 15,
        # #     MI_MAX_ITER = 12,
        # #     MI_MAX_ITER_UNSOLVED = 14,
        # #     MI_INTEGER_INFEASIBLE = 17,
        # #     UNSOLVED = -1,
        # #     EMPTY = -1235,
        # #     UNKNOWN = -1234
        # #     };
        
        # # Typically we want to monitor that the solution is returning SOLVED
        # # Max iter means that the active set algorithm maximum number of iterations set before solution converging was hit
        # # Primal infeasible means that the control constraints are not "being met well"

class UIUCCycle(_vehicle):
    '''Cycle-based open-loop driver for virtual vehicles'''
    def __init__(self, s=0., v=0., a=0., filename = 'update_tgsim/update_tgsim/leader_65_AV'):
        super().__init__(s, v, a)

        self.type = VEHTYPE.CYCLE

        # Parameters
        self.filename = filename + '.csv' # The name of the cycle trajectory that the vehicle takes
        self.pred_filename = filename + '_speed_pred' + '.csv'

        # Controls
        self.ua = 0. # Acceleration command

        ### Read file contents
        self.cycle = pd.read_csv(self.filename)
        self.cycle_pred = pd.read_csv(self.pred_filename)

        # Process cycle
        self.cycle['time'] = self.cycle['time']-self.cycle['time'][0]
        self.cycle_pred['time'] = self.cycle_pred['time']-self.cycle_pred['time'][0]

        self.v = self.cycle['speed_kf'][0]

    def getStatus(self, t):
        '''Returns current velocity and commanded acceleration of the current cycle states'''
        # Get number of rows
        num_rows = self.cycle.index.size

        # Find the nearest row of the current time
        diff_cycle = self.cycle['time'] - t
        nearest_index = (diff_cycle).abs().idxmin()

        cycle = self.cycle.loc[nearest_index]

        # States
        cycle_v = cycle['speed_kf']

        if nearest_index+1 + 1 < num_rows:
            next_cycle = self.cycle.loc[nearest_index+1]

            cycle_a = (next_cycle['speed_kf'] - cycle_v) / (next_cycle['time']-cycle['time'])

        else: # Cycle is finished so slow down
            cycle_v = self.v
            cycle_a = -1.2

        # Error check
        assert (diff_cycle).abs().min() < 15.0, 'The cycle time seems to have ended but simulation time is still continuing.'

        # Return
        return cycle_v, cycle_a

    def getPreview(self, t, n_pred_steps=50):
        '''Returns a preview of the future upcoming cycle states'''
        # Prediction
        dt = 0.10
        t_pred = t

        cycle_vs = np.empty(( n_pred_steps )) # Initialize prediction array
        cycle_vs.fill(np.nan)

        for i in range(0, n_pred_steps):
            t_pred += dt
            cycle_vs[i], _ = self.getStatus(t_pred)

        cycle_ss = scipy.integrate.cumulative_trapezoid(cycle_vs, dx=dt) + self.s

        # Return
        return cycle_ss, cycle_vs

    def getPrediction(self, t, n_pred_steps=50):
        '''Returns a prediction of the upcoming cycle states'''
        # Find the nearest row of the current time for prediction
        diff_cycle_pred = self.cycle_pred['time'] - t
        nearest_index = (diff_cycle_pred).abs().idxmin()
        
        cycle_pred = self.cycle_pred.loc[nearest_index]

        # Prediction
        dt = 0.10

        cycle_vs = np.empty(( n_pred_steps )) # Initialize prediction array
        cycle_vs.fill(np.nan)

        for i in range(0, n_pred_steps):
            key = 't_{:d}'.format(i+1)
            cycle_vs[i] = cycle_pred[key]

        cycle_ss = scipy.integrate.cumulative_trapezoid(cycle_vs, dx=dt) + self.s

        # Error check
        assert (diff_cycle_pred).abs().min() < 15.0, 'The cycle prediction times seem to have ended but simulation time is still continuing.'

        # Return
        return cycle_ss, cycle_vs

    def setCommand(self, nvs, t):
        '''Set the control commands, for example desired acceleration and desired lane'''
        # Some logic to run control that follows a pre-defined cycle
        acc_des = 0.

        ### Get acceleration needed
        cycle_v, acc_des = self.getStatus(t)

        # Check that the current velocity matches the intended cycle velocity
        if abs(cycle_v - self.v) > 0.5:
            raise ValueError('Cycle agent deviating from pre-defined cycle?')
        
        # Assign the control
        self.ua = acc_des

class EPACycle(_vehicle):
    '''Cycle-based open-loop driver for virtual vehicles'''
    def __init__(self, s=0., v=0., a=0., filename = 'EPA/US06'):
        super().__init__(s, v, a)

        self.type = VEHTYPE.CYCLE

        # Parameters
        self.filename = filename + '.csv' # The name of the cycle trajectory that the vehicle takes

        # Controls
        self.ua = 0. # Acceleration command

        ### Read file contents
        self.cycle = pd.read_csv(self.filename)

        # Process cycle
        self.cycle['t'] = self.cycle['t']-self.cycle['t'][0]

        self.v = self.cycle['v'][0]

    def getStatus(self, t):
        '''Returns current velocity and commanded acceleration of the current cycle states'''
        # Get number of rows
        num_rows = self.cycle.index.size

        # Find the nearest row of the current time
        diff_cycle = self.cycle['t'] - t
        nearest_index = (diff_cycle).abs().idxmin()

        cycle = self.cycle.loc[nearest_index]

        # States
        cycle_v = cycle['v']

        if nearest_index+1 + 1 < num_rows:
            next_cycle = self.cycle.loc[nearest_index+1]

            cycle_a = (next_cycle['v'] - cycle_v) / (next_cycle['t']-cycle['time'])

        else: # Cycle is finished so slow down
            cycle_v = self.v
            cycle_a = -1.2

        # Error check
        assert (diff_cycle).abs().min() < 0.150 or t > 600., 'The cycle appears to have deviated.'

        # Return
        return cycle_v, cycle_a

    def getPreview(self, t, n_pred_steps=150):
        '''Returns a preview of the future upcoming cycle states'''
        # Prediction
        dt = 0.10
        t_pred = t

        cycle_vs = np.empty(( n_pred_steps )) # Initialize prediction array
        cycle_vs.fill(np.nan)

        for i in range(0, n_pred_steps):
            t_pred += dt
            cycle_vs[i], _ = self.getStatus(t_pred)

        cycle_ss = scipy.integrate.cumulative_trapezoid(cycle_vs, dx=dt) + self.s

        # Return
        return cycle_ss, cycle_vs

    def getPrediction(self, t, n_pred_steps=50):
        '''Returns a prediction of the upcoming cycle states'''
        # Find the nearest row of the current time for prediction
        diff_cycle_pred = self.cycle_pred['time'] - t
        nearest_index = (diff_cycle_pred).abs().idxmin()
        
        cycle_pred = self.cycle_pred.loc[nearest_index]

        # Prediction
        dt = 0.10

        cycle_vs = np.empty(( n_pred_steps )) # Initialize prediction array
        cycle_vs.fill(np.nan)

        for i in range(0, n_pred_steps):
            key = 't_{:d}'.format(i+1)
            cycle_vs[i] = cycle_pred[key]

        cycle_ss = scipy.integrate.cumulative_trapezoid(cycle_vs, dx=dt) + self.s

        # Error check
        assert (diff_cycle_pred).abs().min() < 15.0, 'The cycle prediction times seem to have ended but simulation time is still continuing.'

        # Return
        return cycle_ss, cycle_vs

    def setCommand(self, nvs, t):
        '''Set the control commands, for example desired acceleration and desired lane'''
        # Some logic to run control that follows a pre-defined cycle
        acc_des = 0.

        ### Get acceleration needed
        cycle_v, acc_des = self.getStatus(t)

        # Check that the current velocity matches the intended cycle velocity
        if abs(cycle_v - self.v) > 0.5:
            raise ValueError('Cycle agent deviating from pre-defined cycle?')
         
        # Assign the control
        self.ua = acc_des

class Dummy(_vehicle):
    '''Dummy open-loop driver for virtual vehicles'''
    def __init__(self, s=0., v=0., a=0., v_max = 10., case = ''):
        super().__init__(s, v, a)

        self.type = VEHTYPE.NONE

        # Parameters
        self.v_max = v_max # Max allowed velocity [m/s]

        self.case = case # The name of open-loop case that the trajectory dummy vehicle takes

        # Controls
        self.ua = 0. # Acceleration command

    def getStatus(self, t):
        '''Returns commanded acceleration of the current open-loop states'''
        if self.case == 'sawtooth' or self.case == '':
            # Sawtooth case
            if t < 10:
                acc_des = 1.
            elif t < 30:
                acc_des = -1.
            if t < 50:
                acc_des = 1.
            elif t < 70:
                acc_des = -1.

        elif self.case == 'sinusoid' or self.case == 'sinusoidal':
            # Sinusoidal case
            acc_des = 1.5 * sin(t * 0.25)

        elif self.case == 'static':
            # Error check
            assert self.v == 0., 'Static vehicle but non-zero velocity?'

            # Static obstacle case
            acc_des = 0.

        else:
            raise ValueError('Unknown case type in Dummy vehicle control')
         
        # Return
        return acc_des

    def getPreview(self, t, n_pred_steps=200):
        '''Returns a preview of the future upcoming open-loop states'''
        # Prediction
        dt = 0.10
        t_pred = t

        open_as = np.empty(( n_pred_steps )) # Initialize prediction array
        open_as.fill(np.nan)

        for i in range(0, n_pred_steps):
            t_pred += dt
            open_as[i] = self.getStatus(t_pred)

        open_vs = scipy.integrate.cumulative_trapezoid(open_as, dx=dt) + self.v
        open_ss = scipy.integrate.cumulative_trapezoid(open_vs, dx=dt) + self.s

        # Return
        return open_ss, open_vs

    def setCommand(self, nvs, t):
        '''Set the control commands, for example desired acceleration and desired lane'''
        s_max = 5000 # [m]
        v_max = self.v_max # [m/s]

        # Some basic logic to run control
        acc_des = 0.

        acc_des = self.getStatus(t)

        # Assign the control
        self.ua = acc_des
