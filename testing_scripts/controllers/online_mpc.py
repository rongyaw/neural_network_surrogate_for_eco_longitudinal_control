"""CasADi-based online MPC controller."""

import casadi
import numpy as np


class car_following_mpc():
    def __init__(self):
        super().__init__()
        self.mpc_acc = 0.0
        self.mpc_jerk = 0.0
        self.mpc_v_tgt = 0.0
        self.h = 20
        self.v_max = 40
        self.dt = 0.5
        self.s_ref = np.zeros((3, self.h), dtype=float)
        self.s_0 = np.zeros((3, 1), dtype=float)
        
        self.mpc_sol_id = 0
        self.mpc_sol_s = np.zeros(self.h, dtype=float)
        self.mpc_sol_v = np.zeros(self.h, dtype=float)
        self.mpc_sol_a = np.zeros(self.h, dtype=float)
        self.mpc_sol_u = np.zeros(self.h - 1, dtype=float)
        self.mpc_warm_start_valid = False
        
        self.control_cost = 4000.0
        self.distance_cost = 1.0
        self.d_gap = 7.0
        self._mpc_problem_key = None

    def set_mpc_warm_start(self, opti, s, u, e_0, e_1, e_2):
        if not self.mpc_warm_start_valid:
            return

        if (
            len(self.mpc_sol_s) != self.h
            or len(self.mpc_sol_v) != self.h
            or len(self.mpc_sol_a) != self.h
            or len(self.mpc_sol_u) != self.h - 1
        ):
            self.mpc_warm_start_valid = False
            return

        try:
            previous_state = np.vstack((self.mpc_sol_s, self.mpc_sol_v, self.mpc_sol_a))
            state_guess = np.zeros((3, self.h), dtype=float)
            state_guess[:, :-1] = previous_state[:, 1:]
            state_guess[:, -1] = previous_state[:, -1]
            state_guess[:, 0] = self.s_0[:, 0]

            control_guess = np.zeros(self.h - 1, dtype=float)
            if self.h > 2:
                control_guess[:-1] = self.mpc_sol_u[1:]
            control_guess[-1] = self.mpc_sol_u[-1]

            opti.set_initial(s, state_guess)
            opti.set_initial(u, control_guess)
            opti.set_initial(e_1, np.zeros(self.h))
            opti.set_initial(e_2, np.zeros(self.h))
            opti.set_initial(e_0, np.zeros(self.h))
        except (RuntimeError, TypeError, ValueError) as err:
            self.mpc_warm_start_valid = False
        
    def reference_update(self, s_ref, v_ref, a_ref, ego_s, ego_v, ego_a):
        # Check the length of the reference signals
        h = len(s_ref)
        # Assign the reference signals to the internal state
        self.s_ref[0,:h] = s_ref
        self.s_ref[1,:h] = v_ref
        self.s_ref[2,:h] = a_ref
        
        # Assign MPC initial state
        self.s_0[0, 0] = ego_s
        self.s_0[1, 0] = ego_v
        self.s_0[2, 0] = ego_a
        
    def _build_mpc_problem(self):
        """Construct the solver once for the current model and cost settings."""
        # Convert v_max from mph to m/s
        v_max = self.v_max * 0.44704
        
        # Create an optimization problem
        opti = casadi.Opti()
        
        # Define optimization variables
        s = opti.variable(3, self.h)  # [s, v, a]
        u = opti.variable(self.h - 1)  # Acceleration
        e_0 = opti.variable(self.h)
        e_1 = opti.variable(self.h)
        e_2 = opti.variable(self.h)
        s_ref = opti.parameter(3, self.h)
        s_0 = opti.parameter(3, 1)
        
        q_a = casadi.MX(self.control_cost)
        q_u = casadi.MX(1)
        q_g = casadi.MX(1)
        q_e0 = casadi.MX(1e4)
        q_e1 = casadi.MX(1e6)
        q_e2 = casadi.MX(1e6)
        
        A = casadi.MX(np.array([[1, self.dt, self.dt ** 2 / 2],
                                [0, 1,  self.dt],
                                [0, 0, 1]]))
        
        B = casadi.MX(np.array([[self.dt ** 3 / 6],
                                [self.dt ** 2 / 4],
                                [self.dt]]))
        
        opti.subject_to(s[:, 0] == s_0)
        
        cost = 0
        
        for t in range(self.h - 1):
            # Define cost function
            cost += q_u * u[t] ** 2 + q_g * (s_ref[0, t] - self.d_gap - s[0, t]) ** 2 + q_a * s[2, t] ** 2
            
            # Add system state update function
            opti.subject_to(s[:, t + 1] == A @ s[:, t] + B @ u[t])
            
            # Add constriants
            opti.subject_to(s[1, t] <= v_max + e_1[t])
            opti.subject_to(s[1, t] >= 0.0 - e_2[t])
            opti.subject_to(s[0, t] <= s_ref[0, t] - self.d_gap + e_0[t])
            cost += q_e0 * (e_0[t] ** 2) + q_e1 * (e_1[t] ** 2) + q_e2 * (e_2[t] ** 2)
            
        # Add final states' cost
        cost += q_g * (s_ref[0, self.h - 1] - self.d_gap - s[0, self.h - 1]) ** 2 + q_a * s[2, self.h - 1] ** 2
        cost += q_e0 * (e_0[self.h - 1] ** 2) + q_e1 * (e_1[self.h - 1] ** 2) + q_e2 * (e_2[self.h - 1] ** 2)
        
        # Add find states' constriants
        opti.subject_to(s[1, self.h - 1] <= v_max + e_1[self.h - 1])
        opti.subject_to(s[1, self.h - 1] >= 0.0 - e_2[self.h - 1])
        opti.subject_to(s[0, self.h - 1] <= s_ref[0, self.h - 1] - self.d_gap + e_0[self.h - 1])
        
        opti.minimize(cost)
        opti.solver('ipopt', {"expand": True, "print_time": 0}, {"print_level": 0})
        
        self._mpc_opti = opti
        self._mpc_variables = (s, u, e_0, e_1, e_2)
        self._mpc_parameters = (s_ref, s_0)

    def mpc_solver(self):
        # These settings are embedded in the graph; changing one requires a
        # rebuild. References and the measured state only update parameters.
        problem_key = (self.h, self.dt, self.v_max, self.control_cost, self.d_gap)
        if self._mpc_problem_key != problem_key:
            self._build_mpc_problem()
            self._mpc_problem_key = problem_key

        opti = self._mpc_opti
        s, u, e_0, e_1, e_2 = self._mpc_variables
        s_ref, s_0 = self._mpc_parameters
        opti.set_value(s_ref, self.s_ref)
        opti.set_value(s_0, self.s_0)
        # A reused Opti retains initial guesses. Reset them when a previous
        # failure or horizon change invalidates the trajectory warm start.
        opti.set_initial(opti.x, 0)
        self.set_mpc_warm_start(opti, s, u, e_0, e_1, e_2)
        
        try:
            sol = opti.solve()
        except RuntimeError:
            self.mpc_warm_start_valid = False
            # FATROP can fail from the shifted trajectory even when the same
            # problem solves from a zero initial guess. Retry once without the
            # warm start; propagate any second failure instead of using a
            # failed iterate as a control command.
            opti.set_initial(opti.x, 0)
            opti.set_initial(opti.lam_g, 0)
            sol = opti.solve()
        
        self.mpc_sol_s = sol.value(s[0, :])
        self.mpc_sol_v = sol.value(s[1, :])
        self.mpc_sol_a = sol.value(s[2, :])
        self.mpc_sol_u = sol.value(u)
        
        self.mpc_jerk = self.mpc_sol_u[0]
        self.mpc_v_tgt = self.mpc_sol_v[1]
        self.mpc_sol_acc = self.mpc_sol_a
        self.mpc_warm_start_valid = True
