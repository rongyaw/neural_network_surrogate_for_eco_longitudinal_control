"""Python adapter for the CasADi-generated Preview MPC library."""

import atexit
import ctypes
import os

import numpy as np


class CGeneratedPreviewMPC:
    def __init__(self, library_path=None):
        self.h = 20
        self.v_max = 40.0
        self.dt = 0.5
        self.control_cost = 4000.0
        self.distance_cost = 1.0
        self.d_gap = 7.0
        self.mpc_acc = 0.0
        self.mpc_jerk = 0.0
        self.mpc_v_tgt = 0.0
        self.mpc_sol_id = 0
        self.mpc_sol_s = np.zeros(self.h, dtype=float)
        self.mpc_sol_v = np.zeros(self.h, dtype=float)
        self.mpc_sol_a = np.zeros(self.h, dtype=float)
        self.mpc_sol_u = np.zeros(self.h - 1, dtype=float)
        self.mpc_sol_acc = self.mpc_sol_a
        self.mpc_warm_start_valid = False
        self.s_ref = np.zeros((3, self.h), dtype=float)
        self.s_0 = np.zeros((3, 1), dtype=float)

        if library_path is None:
            library_path = os.path.join(
                os.path.dirname(__file__), "libpreview_mpc.so"
            )
        if not os.path.isfile(library_path):
            build_script = os.path.join(os.path.dirname(__file__), "preview_mpc.py")
            raise FileNotFoundError(
                "Generated Preview MPC library not found: "
                + library_path
                + ". Build it with: python3 "
                + build_script
            )

        self._library = ctypes.CDLL(library_path)
        self._configure_library()
        self._allocate_buffers()
        self._closed = False
        atexit.register(self.close)

    def _configure_library(self):
        double_pointer = ctypes.POINTER(ctypes.c_double)
        self._library.preview_mpc_step.argtypes = [double_pointer] * 6
        self._library.preview_mpc_step.restype = ctypes.c_int
        self._library.preview_mpc_cleanup.argtypes = []
        self._library.preview_mpc_cleanup.restype = None

    def close(self):
        if not self._closed:
            self._library.preview_mpc_cleanup()
            self._closed = True

    def _allocate_buffers(self):
        self._reference_buffer = np.zeros((3, self.h), dtype=np.float64, order="F")
        self._initial_state_buffer = np.zeros((3, 1), dtype=np.float64, order="F")
        self._state_guess_buffer = np.zeros((3, self.h), dtype=np.float64, order="F")
        self._control_guess_buffer = np.zeros(self.h - 1, dtype=np.float64)
        self._state_solution_buffer = np.zeros((3, self.h), dtype=np.float64, order="F")
        self._control_solution_buffer = np.zeros(self.h - 1, dtype=np.float64)

    def reference_update(self, s_ref, v_ref, a_ref, ego_s, ego_v, ego_a):
        horizon = len(s_ref)
        self.s_ref[0, :horizon] = s_ref
        self.s_ref[1, :horizon] = v_ref
        self.s_ref[2, :horizon] = a_ref
        self.s_0[:, 0] = [ego_s, ego_v, ego_a]

    def _initial_guesses(self):
        if not self.mpc_warm_start_valid:
            return (
                np.zeros((3, self.h), dtype=float, order="F"),
                np.zeros(self.h - 1, dtype=float),
            )

        previous_state = np.vstack(
            (self.mpc_sol_s, self.mpc_sol_v, self.mpc_sol_a)
        )
        state_guess = np.zeros((3, self.h), dtype=float, order="F")
        state_guess[:, :-1] = previous_state[:, 1:]
        state_guess[:, -1] = previous_state[:, -1]
        state_guess[:, 0] = self.s_0[:, 0]

        control_guess = np.zeros(self.h - 1, dtype=float)
        control_guess[:-1] = self.mpc_sol_u[1:]
        control_guess[-1] = self.mpc_sol_u[-1]
        return state_guess, control_guess

    def _solve(self, state_guess, control_guess):
        np.copyto(self._reference_buffer, self.s_ref)
        np.copyto(self._initial_state_buffer, self.s_0)
        np.copyto(self._state_guess_buffer, state_guess)
        np.copyto(self._control_guess_buffer, control_guess)
        self._state_solution_buffer.fill(0.0)
        self._control_solution_buffer.fill(0.0)

        double_pointer = ctypes.POINTER(ctypes.c_double)
        status = self._library.preview_mpc_step(
            self._reference_buffer.ctypes.data_as(double_pointer),
            self._initial_state_buffer.ctypes.data_as(double_pointer),
            self._state_guess_buffer.ctypes.data_as(double_pointer),
            self._control_guess_buffer.ctypes.data_as(double_pointer),
            self._state_solution_buffer.ctypes.data_as(double_pointer),
            self._control_solution_buffer.ctypes.data_as(double_pointer),
        )

        if (
            status != 0
            or not np.all(np.isfinite(self._state_solution_buffer))
            or not np.all(np.isfinite(self._control_solution_buffer))
        ):
            return None
        return self._state_solution_buffer.copy(), self._control_solution_buffer.copy()

    def mpc_solver(self):
        if self._closed:
            raise RuntimeError("Generated Preview MPC controller has been closed")
        state_guess, control_guess = self._initial_guesses()
        result = self._solve(state_guess, control_guess)
        if result is None and self.mpc_warm_start_valid:
            self.mpc_warm_start_valid = False
            result = self._solve(
                np.zeros((3, self.h), dtype=float, order="F"),
                np.zeros(self.h - 1, dtype=float),
            )
        if result is None:
            raise RuntimeError("Generated Preview MPC QP solve failed")

        state_solution, self.mpc_sol_u = result
        self.mpc_sol_s = state_solution[0, :].copy()
        self.mpc_sol_v = state_solution[1, :].copy()
        self.mpc_sol_a = state_solution[2, :].copy()
        self.mpc_sol_acc = self.mpc_sol_a
        self.mpc_jerk = self.mpc_sol_u[0]
        self.mpc_v_tgt = self.mpc_sol_v[1]
        self.mpc_warm_start_valid = True
