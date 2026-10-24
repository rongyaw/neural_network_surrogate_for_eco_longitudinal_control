# /usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from scipy.interpolate import RegularGridInterpolator

import math
import csv
import os
import hashlib
import json
import tempfile
import numpy as np


class ExplicitMPCConfig:
    """Configuration for the reconstructed fixed-horizon explicit MPC."""

    def __init__(self, dt=0.1, horizon=20, vehicle_length=7.0,
                 desired_gap=0.0, minimum_gap=0.0, time_headway=0.0,
                 v_min=0.0, v_max=35.0, u_min=-3.0, u_max=3.0,
                 q_gap=1.0, q_acceleration=2000.0, q_command=2000.0,
                 q_slack=1.0e6, sample_count=400, random_seed=44,
                 enable_cbf=True, include_acceleration_speed_constraints=False,
                 cache_regions=True, region_cache_dir=None):
        if horizon <= 0:
            raise ValueError("Explicit MPC horizon must be positive.")
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.vehicle_length = float(vehicle_length)
        self.desired_gap = float(desired_gap)
        self.minimum_gap = float(minimum_gap)
        self.time_headway = float(time_headway)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.q_gap = float(q_gap)
        self.q_acceleration = float(q_acceleration)
        self.q_command = float(q_command)
        self.q_slack = float(q_slack)
        self.sample_count = int(sample_count)
        self.random_seed = int(random_seed)
        self.enable_cbf = bool(enable_cbf)
        self.include_acceleration_speed_constraints = bool(
            include_acceleration_speed_constraints
        )
        self.cache_regions = bool(cache_regions)
        self.region_cache_dir = region_cache_dir


class ExplicitMPCRegion:
    """One critical-region affine solution and its KKT validity inequalities."""

    def __init__(self, theta_lower, theta_upper, control_gain, control_offset,
                 active_set=()):
        self.theta_lower = np.asarray(theta_lower, dtype=float)
        self.theta_upper = np.asarray(theta_upper, dtype=float)
        self.control_gain = np.asarray(control_gain, dtype=float)
        self.control_offset = float(control_offset)
        self.active_set = tuple(active_set)

    def contains(self, theta, tolerance=1.0e-7):
        theta = np.asarray(theta, dtype=float).reshape(-1)
        return bool(np.all(theta >= self.theta_lower - tolerance) and
                    np.all(theta <= self.theta_upper + tolerance))

    def evaluate(self, theta):
        return float(self.control_gain.dot(theta) + self.control_offset)


class _ExplicitMPCBase:
    """Reduced multiparametric-QP explicit MPC implementation.

    The runtime law is represented by affine laws identified from the QP active
    sets. Region discovery is performed during controller initialization; an
    OSQP solve is used whenever a discovered region does not cover the input.
    """

    def __init__(self, config=None, generate_regions=True, print_level="info"):
        self.config = config if config is not None else ExplicitMPCConfig()
        self.print_level = print_level
        self.theta_dim = 6 if self.mode == "unconnected" else 6 + self.config.horizon
        self._build_prediction_model()
        self._build_qp()
        self.regions = []
        self.fallback_count = 0
        self.region_hits = 0
        self.last_diagnostics = {}
        self.region_cache_path = self._region_cache_path()
        if generate_regions:
            loaded_from_cache = self.config.cache_regions and self.load_regions()
            if not loaded_from_cache:
                self.generate_regions()
                if self.config.cache_regions:
                    self.save_regions()

    def _region_cache_path(self):
        """Return a configuration-specific cache path for the region set."""
        cache_settings = {
            "schema": 1,
            "mode": self.mode,
            "dt": self.config.dt,
            "horizon": self.config.horizon,
            "vehicle_length": self.config.vehicle_length,
            "desired_gap": self.config.desired_gap,
            "minimum_gap": self.config.minimum_gap,
            "time_headway": self.config.time_headway,
            "v_min": self.config.v_min,
            "v_max": self.config.v_max,
            "u_min": self.config.u_min,
            "u_max": self.config.u_max,
            "q_gap": self.config.q_gap,
            "q_acceleration": self.config.q_acceleration,
            "q_command": self.config.q_command,
            "q_slack": self.config.q_slack,
            "sample_count": self.config.sample_count,
            "random_seed": self.config.random_seed,
            "include_acceleration_speed_constraints": (
                self.config.include_acceleration_speed_constraints
            ),
            "theta_dim": self.theta_dim,
        }
        signature = hashlib.sha256(
            json.dumps(cache_settings, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        cache_dir = self.config.region_cache_dir
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     ".explicit_mpc_cache")
        filename = (
            f"{self.mode}_N{self.config.horizon}_samples"
            f"{self.config.sample_count}_{signature}.npz"
        )
        return os.path.join(cache_dir, filename)

    def load_regions(self):
        """Load a previously generated region set if it matches this QP."""
        if not os.path.exists(self.region_cache_path):
            return False
        try:
            with np.load(self.region_cache_path, allow_pickle=True) as data:
                region_count = int(data["region_count"])
                if int(data["theta_dim"]) != self.theta_dim:
                    return False
                if int(data["constraint_count"]) != len(self.constraint_names):
                    return False
                regions = []
                for index in range(region_count):
                    active_set = tuple(
                        np.asarray(data["active_sets"][index], dtype=int).tolist()
                    )
                    region = ExplicitMPCRegion(
                        theta_lower=np.full(self.theta_dim, -np.inf),
                        theta_upper=np.full(self.theta_dim, np.inf),
                        control_gain=data["control_gain"][index],
                        control_offset=data["control_offset"][index],
                        active_set=active_set,
                    )
                    region.primal_offset = data["primal_offset"][index]
                    region.primal_gain = data["primal_gain"][index]
                    region.dual_offset = data["dual_offset"][index]
                    region.dual_gain = data["dual_gain"][index]
                    region.sample_theta = data["sample_theta"][index]
                    regions.append(region)
                self.regions = regions
            if self.print_level == "debug":
                print("Loaded", len(self.regions), self.mode,
                      "explicit MPC regions from", self.region_cache_path)
            return True
        except (OSError, KeyError, ValueError, IndexError):
            return False

    def save_regions(self):
        """Save the generated region set atomically for later controller runs."""
        cache_dir = os.path.dirname(self.region_cache_path)
        os.makedirs(cache_dir, exist_ok=True)
        region_count = len(self.regions)
        active_sets = np.empty(region_count, dtype=object)
        control_gain = np.empty((region_count, self.theta_dim))
        control_offset = np.empty(region_count)
        primal_offset = np.empty((region_count, len(self.constraint_names)))
        primal_gain = np.empty((region_count, len(self.constraint_names), self.theta_dim))
        sample_theta = np.empty((region_count, self.theta_dim))
        max_active = max((len(region.active_set) for region in self.regions), default=0)
        dual_offset = np.zeros((region_count, max_active))
        dual_gain = np.zeros((region_count, max_active, self.theta_dim))
        active_lengths = np.zeros(region_count, dtype=int)

        for index, region in enumerate(self.regions):
            active_sets[index] = np.asarray(region.active_set, dtype=int)
            active_lengths[index] = len(region.active_set)
            control_gain[index] = region.control_gain
            control_offset[index] = region.control_offset
            primal_offset[index] = region.primal_offset
            primal_gain[index] = region.primal_gain
            sample_theta[index] = region.sample_theta
            if region.active_set:
                dual_offset[index, :len(region.active_set)] = region.dual_offset
                dual_gain[index, :len(region.active_set)] = region.dual_gain

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=cache_dir, suffix=".npz", delete=False
            ) as temporary_file:
                temporary_path = temporary_file.name
            np.savez_compressed(
                temporary_path,
                region_count=region_count,
                theta_dim=self.theta_dim,
                constraint_count=len(self.constraint_names),
                active_sets=active_sets,
                active_lengths=active_lengths,
                control_gain=control_gain,
                control_offset=control_offset,
                primal_offset=primal_offset,
                primal_gain=primal_gain,
                dual_offset=dual_offset,
                dual_gain=dual_gain,
                sample_theta=sample_theta,
            )
            os.replace(temporary_path, self.region_cache_path)
            if self.print_level == "debug":
                print("Saved", len(self.regions), self.mode,
                      "explicit MPC regions to", self.region_cache_path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _build_prediction_model(self):
        dt = self.config.dt
        self.A = np.array([[1.0, dt, 0.25 * dt * dt],
                           [0.0, 1.0, 0.5 * dt],
                           [0.0, 0.0, 0.0]])
        self.B = np.array([[0.25 * dt * dt], [0.5 * dt], [1.0]])
        self.C_gap = np.array([[1.0, self.config.time_headway, 0.0]])
        self._state_maps = []
        for stage in range(self.config.horizon + 1):
            state_theta = np.zeros((3, self.theta_dim))
            state_theta[:, :3] = np.linalg.matrix_power(self.A, stage)
            state_u = np.zeros((3, self.config.horizon))
            for j in range(min(stage, self.config.horizon)):
                state_u[:, j:j + 1] = np.linalg.matrix_power(self.A, stage - 1 - j).dot(self.B)
            self._state_maps.append((state_theta, state_u))

    def _pv_position_map(self, stage):
        """Return p_pv(stage) = offset + gain @ theta."""
        gain = np.zeros(self.theta_dim)
        if self.mode == "unconnected":
            tau = stage * self.config.dt
            gain[3] = 1.0
            gain[4] = tau
            gain[5] = 0.5 * tau * tau
        else:
            # Connected parameters carry the absolute PV position preview
            # after the current PV state [ego state (3) + PV state (3)].
            # This is the same reference trajectory supplied to setPred().
            if stage > 0:
                gain[6 + stage - 1] = 1.0
        return gain

    def _build_qp(self):
        """Build H, q(theta), G, and rhs(theta) for the condensed QP."""
        import scipy.sparse as sp
        import osqp

        n = self.config.horizon
        self._n_u = n
        self._n_slack = 3 * n
        self._n_z = self._n_u + self._n_slack
        H = np.eye(self._n_z) * 1.0e-8
        q0 = np.zeros(self._n_z)
        Qtheta = np.zeros((self._n_z, self.theta_dim))

        def u_col(j):
            return j

        def slack_col(kind, j):
            return self._n_u + kind * n + j

        for stage in range(1, n + 1):
            x_theta, x_u = self._state_maps[stage]
            pv_gain = self._pv_position_map(stage)
            gap_theta = self.C_gap.dot(x_theta).reshape(-1) - pv_gain
            gap_u = self.C_gap.dot(x_u).reshape(-1)
            gap_offset = self.config.desired_gap
            H += 2.0 * self.config.q_gap * np.outer(
                np.r_[gap_u, np.zeros(self._n_slack)],
                np.r_[gap_u, np.zeros(self._n_slack)])
            q0 += 2.0 * self.config.q_gap * gap_offset * np.r_[gap_u, np.zeros(self._n_slack)]
            Qtheta += 2.0 * self.config.q_gap * np.outer(
                np.r_[gap_u, np.zeros(self._n_slack)], gap_theta)

            acc_theta = x_theta[2, :]
            acc_u = x_u[2, :]
            H += 2.0 * self.config.q_acceleration * np.outer(
                np.r_[acc_u, np.zeros(self._n_slack)],
                np.r_[acc_u, np.zeros(self._n_slack)])
            q0 += 2.0 * self.config.q_acceleration * 0.0 * np.r_[acc_u, np.zeros(self._n_slack)]
            Qtheta += 2.0 * self.config.q_acceleration * np.outer(
                np.r_[acc_u, np.zeros(self._n_slack)], acc_theta)

            H[u_col(stage - 1), u_col(stage - 1)] += 2.0 * self.config.q_command
            q0[slack_col(0, stage - 1)] = self.config.q_slack
            q0[slack_col(1, stage - 1)] = self.config.q_slack
            q0[slack_col(2, stage - 1)] = self.config.q_slack

        rows = []
        w = []
        W = []
        names = []

        def add_row(g, rhs_offset, rhs_gain, name):
            rows.append(np.asarray(g, dtype=float))
            w.append(float(rhs_offset))
            W.append(np.asarray(rhs_gain, dtype=float))
            names.append(name)

        for stage in range(n):
            x_theta, x_u = self._state_maps[stage]
            v_theta = x_theta[1, :]
            v_u = x_u[1, :]
            if self.config.include_acceleration_speed_constraints:
                for slope, intercept, label in ((0.285, 2.0, "acc_upper_1"),
                                                 (-0.121, 4.83, "acc_upper_2")):
                    add_row(np.r_[-slope * v_u + np.eye(1, n, stage).reshape(-1), np.zeros(self._n_slack)],
                            intercept, slope * v_theta, label + str(stage))
            add_row(np.r_[-np.eye(1, n, stage).reshape(-1), np.zeros(self._n_slack)],
                    -self.config.u_min, np.zeros(self.theta_dim), "u_lower" + str(stage))
            add_row(np.r_[np.eye(1, n, stage).reshape(-1), np.zeros(self._n_slack)],
                    self.config.u_max, np.zeros(self.theta_dim), "u_upper" + str(stage))

        for stage in range(1, n + 1):
            x_theta, x_u = self._state_maps[stage]
            v_theta = x_theta[1, :]
            v_u = x_u[1, :]
            s_theta = x_theta[0, :]
            s_u = x_u[0, :]
            add_row(np.r_[v_u, np.eye(1, n, stage - 1).reshape(-1),
                          np.zeros(2 * n)], self.config.v_max,
                    -v_theta, "v_upper" + str(stage))
            add_row(np.r_[-v_u, np.zeros(n), np.eye(1, n, stage - 1).reshape(-1),
                          np.zeros(n)], -self.config.v_min,
                    v_theta, "v_lower" + str(stage))
            pv_gain = self._pv_position_map(stage)
            add_row(np.r_[s_u, np.zeros(n), np.zeros(n),
                          np.eye(1, n, stage - 1).reshape(-1)], -self.config.minimum_gap,
                    pv_gain - s_theta, "gap" + str(stage))

        for kind in range(3):
            for stage in range(n):
                g = np.zeros(self._n_z)
                g[slack_col(kind, stage)] = -1.0
                add_row(g, 0.0, np.zeros(self.theta_dim), "slack" + str(kind) + str(stage))

        self.H = 0.5 * (H + H.T)
        self.q0 = q0
        self.Qtheta = Qtheta
        self.G = np.asarray(rows)
        self.w = np.asarray(w)
        self.W = np.asarray(W)
        self.constraint_names = names
        self._osqp = osqp.OSQP()
        self._osqp.setup(P=sp.csc_matrix(self.H), q=self.q0,
                         A=sp.csc_matrix(self.G), l=-np.inf * np.ones(len(w)),
                         u=self.w, verbose=False, polish=True,
                         eps_abs=1.0e-6, eps_rel=1.0e-6, max_iter=4000)

    def _solve_qp(self, theta):
        theta = np.asarray(theta, dtype=float).reshape(-1)
        self._osqp.update(q=self.q0 + self.Qtheta.dot(theta),
                          u=self.w + self.W.dot(theta))
        result = self._osqp.solve()
        if result.x is None or result.info.status not in ("solved", "solved inaccurate"):
            return None, None
        residual = self.G.dot(result.x) - self.w - self.W.dot(theta)
        active = tuple(np.flatnonzero(residual >= -2.0e-5).tolist())
        return result.x, active

    def _region_from_active_set(self, active):
        active = tuple(active)
        if not active:
            return None
        GA = self.G[list(active), :]
        WA = self.W[list(active), :]
        n = self._n_z
        kkt = np.block([[self.H, GA.T], [GA, np.zeros((len(active), len(active)))]])
        if np.linalg.matrix_rank(kkt) < kkt.shape[0]:
            return None
        rhs0 = np.r_[-self.q0, self.w[list(active)]]
        rhs_theta = np.vstack((-self.Qtheta, WA))
        try:
            solution0 = np.linalg.solve(kkt, rhs0)
            solution_theta = np.linalg.solve(kkt, rhs_theta)
        except np.linalg.LinAlgError:
            return None
        z0 = solution0[:n]
        Z = solution_theta[:n, :]
        lam0 = solution0[n:]
        Lam = solution_theta[n:, :]
        upper = self.w - self.G.dot(z0)
        upper_gain = self.G.dot(Z) - self.W
        # KKT primal and dual inequalities define the exact polyhedral region.
        lower = lam0
        lower_gain = Lam
        # Store a conservative box extracted from the sampled region. This
        # keeps lookup cheap while retaining the exact affine law.
        return z0, Z, upper, upper_gain, lower, lower_gain

    def generate_regions(self):
        rng = np.random.default_rng(self.config.random_seed + (1 if self.mode == "connected" else 0))
        samples = []
        for _ in range(self.config.sample_count):
            ego_s = rng.uniform(0.0, 500.0)
            ego_v = rng.uniform(0.0, 30.0)
            ego_a = rng.uniform(-2.0, 2.0)
            pv_s = ego_s + rng.uniform(self.config.minimum_gap + 1.0, 100.0)
            pv_v = rng.uniform(0.0, 30.0)
            pv_a = rng.uniform(-2.0, 2.0)
            if self.mode == "unconnected":
                samples.append([ego_s, ego_v, ego_a, pv_s, pv_v, pv_a])
            else:
                preview_s = []
                predicted_s = pv_s
                predicted_v = pv_v
                for stage in range(self.config.horizon):
                    predicted_v = max(predicted_v + pv_a * self.config.dt, 0.0)
                    predicted_v = min(predicted_v, self.config.v_max)
                    predicted_s += predicted_v * self.config.dt
                    preview_s.append(predicted_s)
                samples.append([ego_s, ego_v, ego_a, pv_s, pv_v, pv_a] + preview_s)

        seen = set()
        for theta in samples:
            z, active = self._solve_qp(theta)
            if z is None or active in seen:
                continue
            seen.add(active)
            affine = self._region_from_active_set(active)
            if affine is None:
                continue
            z0, Z, upper, upper_gain, lower, lower_gain = affine
            # Derive a conservative axis-aligned box around the sampled point
            # using the exact KKT inequalities. Runtime verification still
            # checks the stored inequalities below.
            region = ExplicitMPCRegion(
                theta_lower=np.full(self.theta_dim, -np.inf),
                theta_upper=np.full(self.theta_dim, np.inf),
                control_gain=Z[0, :], control_offset=z0[0], active_set=active)
            region.primal_offset = upper
            region.primal_gain = upper_gain
            region.dual_offset = lower
            region.dual_gain = lower_gain
            region.sample_theta = np.asarray(theta, dtype=float)
            self.regions.append(region)
        if self.print_level == "debug":
            print("Generated", len(self.regions), self.mode, "explicit MPC active-set regions")

    def _region_is_valid(self, region, theta, tolerance=2.0e-5):
        return bool(np.all(region.primal_offset - region.primal_gain.dot(theta) >= -tolerance) and
                    np.all(region.dual_offset + region.dual_gain.dot(theta) >= -tolerance))

    def _cbf_limit(self, ego_state, pv_state):
        ego_s, ego_v, _ = ego_state
        pv_s, pv_v, _ = pv_state
        tau, alpha, length = 1.8, 2.0, 7.0
        return (pv_v - ego_v + alpha * (pv_s - ego_s - length - tau * ego_v)) / tau

    def _step(self, theta, ego_state, pv_state):
        theta = np.asarray(theta, dtype=float)
        selected = None
        for region in self.regions:
            if self._region_is_valid(region, theta):
                selected = region
                break
        fallback = selected is None
        if fallback:
            self.fallback_count += 1
            z, _ = self._solve_qp(theta)
            acceleration = self.config.u_min if z is None else z[0]
        else:
            self.region_hits += 1
            acceleration = selected.evaluate(theta)
        raw_acceleration = float(acceleration)
        if self.config.enable_cbf:
            safety_limit = self._cbf_limit(ego_state, pv_state)
            safety_override = raw_acceleration > safety_limit
            acceleration = min(raw_acceleration, safety_limit)
        else:
            safety_limit = float("inf")
            safety_override = False
        acceleration = float(np.clip(acceleration, self.config.u_min, self.config.u_max))
        self.last_diagnostics = {
            "mode": self.mode, "fallback": fallback,
            "region_active_set": None if selected is None else selected.active_set,
            "raw_acceleration": raw_acceleration,
            "safety_limit": float(safety_limit),
            "safety_override": safety_override,
            "cbf_enabled": self.config.enable_cbf,
            "region_hits": self.region_hits,
            "fallback_count": self.fallback_count,
        }
        return acceleration


class ExplicitMPCUnconnectedController:
    """Explicit MPC with a constant-acceleration preview.

    The estimated PV trajectory is passed through the same preview-parameter
    controller used by :class:`ExplicitMPCConnectedController`. This makes the
    connected/unconnected distinction depend only on the preview source:
    estimated PV motion versus communicated PV motion.
    """

    def __init__(self, config=None, generate_regions=True, print_level="info"):
        # The connected controller owns the single preview-parameterized QP and
        # its region set. Runtime inputs determine whether the preview is
        # estimated locally or received from another vehicle.
        self._preview_controller = ExplicitMPCConnectedController(
            config=config,
            generate_regions=generate_regions,
            print_level=print_level,
        )

    @property
    def config(self):
        return self._preview_controller.config

    @property
    def regions(self):
        return self._preview_controller.regions

    @property
    def fallback_count(self):
        return self._preview_controller.fallback_count

    @property
    def region_hits(self):
        return self._preview_controller.region_hits

    @property
    def last_diagnostics(self):
        return self._preview_controller.last_diagnostics

    def _constant_acceleration_preview(self, pv_state):
        pv_s, pv_v, pv_a = np.asarray(pv_state, dtype=float).reshape(3)
        preview_s = np.empty(self.config.horizon, dtype=float)
        predicted_s = pv_s
        predicted_v = pv_v
        for stage in range(self.config.horizon):
            predicted_v = np.clip(
                predicted_v + self.config.dt * pv_a,
                self.config.v_min,
                self.config.v_max,
            )
            # Match the project's constant-acceleration preview convention:
            # each preview position is advanced using the clipped next speed.
            predicted_s += self.config.dt * predicted_v
            preview_s[stage] = predicted_s
        return preview_s

    def step(self, ego_state, pv_state, return_diagnostics=False):
        pv_state = np.asarray(pv_state, dtype=float).reshape(3)
        preview_s = self._constant_acceleration_preview(pv_state)
        result = self._preview_controller.step(
            ego_state=ego_state,
            pv_state=pv_state,
            pv_position_preview=preview_s,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            acceleration, diagnostics = result
            diagnostics["mode"] = "unconnected_constant_acceleration_preview"
            return acceleration, diagnostics
        return result


class ExplicitMPCConnectedController(_ExplicitMPCBase):
    """Explicit MPC using communicated PV position preview."""
    mode = "connected"

    def step(self, ego_state, pv_state, pv_position_preview,
             pv_velocity_preview=None, return_diagnostics=False):
        ego_state = np.asarray(ego_state, dtype=float).reshape(3)
        pv_state = np.asarray(pv_state, dtype=float).reshape(3)
        preview_s = np.asarray(pv_position_preview, dtype=float).reshape(-1)
        if preview_s.size < self.config.horizon:
            raise ValueError("Connected preview must contain at least the configured horizon positions.")
        preview_s = preview_s[:self.config.horizon]
        # The velocity preview is accepted for interface parity and for
        # diagnostics, but the current Eco-MPC objective uses PV position
        # references. This matches the generated controller's setPred input.
        if pv_velocity_preview is not None:
            preview_v = np.asarray(pv_velocity_preview, dtype=float).reshape(-1)
            if preview_v.size < self.config.horizon:
                raise ValueError("Connected preview must contain at least the configured horizon velocities.")
        qp_ego_state = ego_state.copy()
        qp_ego_state[0] += self.config.vehicle_length
        theta = np.r_[qp_ego_state, pv_state, preview_s]
        acceleration = self._step(theta, ego_state, pv_state)
        return (acceleration, self.last_diagnostics.copy()) if return_diagnostics else acceleration
class IDM():
    def __init__(self, a, b, s0, v0, T):
        self.a = a
        self.b = b
        self.s0 = s0
        self.v0 = v0
        self.T = T

    def IDM_acceleration(self, front_v, ego_v, front_s, ego_s):
        s_safe = self.s0 + ego_v * self.T + front_v * (ego_v - front_v) / (2 * (self.a * self.b)**0.5)
        s_safe[s_safe < self.s0 + 3] = self.s0 + 3
        acc = self.a * (1 - (ego_v / self.v0) ** 4 -
                        (s_safe / (front_s - ego_s - self.s0)) ** 2)
        acc = np.clip(acc, -4, 4)
        return acc


# Import these after IDM is defined. utils.py imports IDM for the legacy
# controller path, so importing them at module top before IDM creates a
# circular-import failure when _controller.py is imported directly.
from utils import *
from _sensor import *
from _agents import *
    
class PCC_MPC_controller():
    def __init__(self, dirname):
        self.s = 0.0
        self.v = 0.0
        self.a = 0.0
        self.svs = PCC(dirname, self.s, self.v, self.a, v_max=35)

class Model(nn.Module):
    def __init__(self, in_features=3, h1=512, h2=512, h3=512, h4=512, h5=512, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        # self.fc3 = nn.Linear(h2, h3)
        # self.fc4 = nn.Linear(h3, h4)
        # self.fc5 = nn.Linear(h4, h5)
        self.out = nn.Linear(h2, out_features)
        
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        # x = F.sigmoid(self.fc3(x))
        # x = F.sigmoid(self.fc4(x))
        # x = F.sigmoid(self.fc5(x))
        x = self.out(x)
        return x

class Model_4_input(nn.Module):
    def __init__(self, in_features=4, h1=512, h2=512, h3=512, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = self.out(x)
        return x


class PreviewModel(nn.Module):
    def __init__(
        self,
        in_features=41,
        preview_steps=20,
        preview_channels=2,
        ego_features=1,
        conv_channels=16,
        h1=512,
        h2=512,
        trajectory_out_features=38,
        acceleration_h1=512,
        acceleration_h2=512,
        acceleration_out_features=1,
    ):
        super().__init__()
        if in_features != ego_features + preview_steps * preview_channels:
            raise ValueError("Input feature setup does not match preview sequence layout.")

        self.preview_steps = preview_steps
        self.preview_channels = preview_channels
        self.ego_features = ego_features
        self.trajectory_out_features = trajectory_out_features
        self.acceleration_out_features = acceleration_out_features

        self.conv1 = nn.Conv1d(
            preview_channels, conv_channels, kernel_size=9, padding=1
        )
        self.conv_activation = nn.ReLU()
        self.conv_pool = nn.AdaptiveAvgPool1d(8)

        conv_output_features = conv_channels * 8
        self.fc1 = nn.Linear(conv_output_features + ego_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.output = nn.Linear(
            h2, trajectory_out_features + acceleration_out_features
        )
        self.dp = nn.Identity()

    def split_input_features(self, model_input):
        if model_input.shape[1] != self.ego_features + self.preview_steps * self.preview_channels:
            raise ValueError(
                "Unexpected model input feature count: expected "
                + str(self.ego_features + self.preview_steps * self.preview_channels)
                + ", got "
                + str(model_input.shape[1])
            )

        ego_v = model_input[:, :self.ego_features]
        preview_sequence = model_input[:, self.ego_features:].reshape(
            -1, self.preview_steps, self.preview_channels
        )
        preview_sequence = preview_sequence.transpose(1, 2).contiguous()
        return ego_v, preview_sequence

    def forward(self, x):
        ego_v, preview_sequence = self.split_input_features(x)
        preview_features = self.conv_activation(self.conv1(preview_sequence))
        preview_features = self.conv_pool(preview_features)
        preview_features = torch.flatten(preview_features, start_dim=1)

        x = torch.cat((ego_v, preview_features), dim=1)
        x = torch.sigmoid(self.fc1(x))
        x = self.dp(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.dp(x)
        joint_prediction = self.output(x)
        trajectory_prediction = joint_prediction[:, : self.trajectory_out_features]
        acceleration_prediction = joint_prediction[:, self.trajectory_out_features :]
        return trajectory_prediction, acceleration_prediction


class PreviewNN_controller():
    def __init__(
        self,
        nn_pt_file,
        preview_steps=20,
        preview_channels=2,
        ego_features=1,
        conv_channels=32,
        h1=1024,
        h2=1024,
        trajectory_out_features=38,
        acceleration_h1=1024,
        acceleration_h2=1024,
        acceleration_out_features=1,
        safe_distance_headway=8.0,
        enable_cbf_safety=True,
        print_level="info",
    ):
        self.preview_steps = preview_steps
        self.preview_channels = preview_channels
        self.ego_features = ego_features
        self.safe_distance_headway = safe_distance_headway
        self.enable_cbf_safety = enable_cbf_safety
        self.print_level = print_level
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn_controller = PreviewModel(
            in_features=ego_features + preview_steps * preview_channels,
            preview_steps=preview_steps,
            preview_channels=preview_channels,
            ego_features=ego_features,
            conv_channels=conv_channels,
            h1=h1,
            h2=h2,
            trajectory_out_features=trajectory_out_features,
            acceleration_h1=acceleration_h1,
            acceleration_h2=acceleration_h2,
            acceleration_out_features=acceleration_out_features,
        )
        self.nn_controller.eval()
        self.nn_controller.load_state_dict(
            torch.load(nn_pt_file, map_location=self.device, weights_only=True)
        )
        self.nn_controller.to(self.device)

    def CBF_acceleration_bound_check(self, pv_vt, s_vt, pv_st, s_st, tao, alpha, L):
        a_ego_max = (pv_vt - s_vt + alpha * (pv_st - s_st - L - tao * s_vt)) / tao
        return a_ego_max

    def build_model_input(self, ego_vt, distance_headway_preview, speed_gap_preview):
        ego_vt = np.asarray(ego_vt, dtype=np.float32).reshape(-1)
        distance_headway_preview = np.asarray(
            distance_headway_preview, dtype=np.float32
        )
        speed_gap_preview = np.asarray(speed_gap_preview, dtype=np.float32)

        if distance_headway_preview.ndim == 1:
            distance_headway_preview = distance_headway_preview.reshape(1, -1)
        if speed_gap_preview.ndim == 1:
            speed_gap_preview = speed_gap_preview.reshape(1, -1)

        if distance_headway_preview.shape != speed_gap_preview.shape:
            raise ValueError("Distance-headway preview and speed-gap preview must have the same shape.")
        if distance_headway_preview.shape[1] != self.preview_steps:
            raise ValueError(
                "Preview input length mismatch: expected "
                + str(self.preview_steps)
                + ", got "
                + str(distance_headway_preview.shape[1])
            )
        if ego_vt.shape[0] != distance_headway_preview.shape[0]:
            raise ValueError("Ego speed batch size must match preview batch size.")

        distance_headway_preview = distance_headway_preview - self.safe_distance_headway

        model_input = np.zeros(
            (ego_vt.shape[0], self.ego_features + self.preview_steps * self.preview_channels),
            dtype=np.float32,
        )
        model_input[:, 0] = ego_vt
        for step in range(self.preview_steps):
            feature_start = self.ego_features + step * self.preview_channels
            model_input[:, feature_start] = distance_headway_preview[:, step]
            model_input[:, feature_start + 1] = speed_gap_preview[:, step]

        return torch.from_numpy(model_input).to(self.device)

    def step_forward(
        self,
        ego_vt,
        distance_headway_preview,
        speed_gap_preview,
        pv_vt=None,
        pv_st=None,
        s_st=None,
        sim_t=None,
        lambda_smooth=0.0,
        s_at=None,
        use_cbf_safety=None,
        return_trajectory=False,
    ):
        if use_cbf_safety is None:
            use_cbf_safety = self.enable_cbf_safety

        model_input = self.build_model_input(
            ego_vt=ego_vt,
            distance_headway_preview=distance_headway_preview,
            speed_gap_preview=speed_gap_preview,
        )

        with torch.no_grad():
            trajectory_prediction, acceleration_prediction = self.nn_controller(model_input)

        trajectory_prediction = trajectory_prediction.detach().cpu().numpy()
        acceleration_prediction = acceleration_prediction.detach().cpu().numpy().flatten()

        if s_at is not None and lambda_smooth > 0.0:
            s_at = np.asarray(s_at, dtype=float).reshape(-1)
            acceleration_prediction = (
                acceleration_prediction + lambda_smooth * s_at
            ) / (1.0 + lambda_smooth)

        if use_cbf_safety:
            if pv_vt is None or pv_st is None or s_st is None:
                raise ValueError(
                    "pv_vt, pv_st, and s_st are required when use_cbf_safety is enabled."
                )
            a_ego_max = self.CBF_acceleration_bound_check(
                pv_vt=np.asarray(pv_vt, dtype=float).reshape(-1),
                s_vt=np.asarray(ego_vt, dtype=float).reshape(-1),
                pv_st=np.asarray(pv_st, dtype=float).reshape(-1),
                s_st=np.asarray(s_st, dtype=float).reshape(-1),
                tao=0.8,
                alpha=2.0,
                L=5.0,
            )
            if np.any(acceleration_prediction > a_ego_max):
                if sim_t is not None and self.print_level == "debug":
                    print(
                        f"CBF safety constraint is violated at time {sim_t}! Adjusting preview NN control to ensure safety..."
                    )
                acceleration_prediction = np.minimum(acceleration_prediction, a_ego_max)

        if return_trajectory:
            return acceleration_prediction.tolist(), trajectory_prediction
        return acceleration_prediction.tolist()


class NN_controller():
    def __init__(self, nn_pt_file, input_num, print_level="info"):
        self.num_input = input_num
        self.print_level = print_level
        if input_num == 3:
            self.nn_controller = Model(h1=256, h2=256, h3=256, h4=256, h5=256)
        if input_num == 4:
            self.nn_controller = Model_4_input(h1=256, h2=256)
        self.nn_controller.eval()
        self.nn_controller.load_state_dict(torch.load(nn_pt_file, map_location='cpu'))
        self.nn_controller.to('cuda')
    
    def CBF_acceleration_bound_check(self, pv_vt, s_vt, pv_st, s_st, tao, alpha, L):
        a_ego_max = (pv_vt - s_vt + alpha * (pv_st - s_st - L - tao * s_vt)) / tao
        return a_ego_max
    
    def step_forward(self, s_vt, pv_vt, s_st, pv_st, s_at, pv_at, use_prediction_horizon, sim_t, lambda_smooth):
        # Calculate the prediction horizon length
        pv_s_end = np.zeros(pv_st.shape)
        pv_v_end = np.zeros(pv_vt.shape)
        a_input = np.array(pv_at)
        a = np.tile(a_input, (49, 1))
        v = pv_vt + np.cumsum(a * 0.5, axis=0)
        v = np.clip(v, 0, np.Inf)
        s = pv_st + np.cumsum(v * 0.5, axis=0)
        pv_s_end = np.clip(s[-1, :] - s_st, -10, 2000)
        pv_v_end = v[-1, :] - s_vt
        
        if self.num_input == 3:
            if use_prediction_horizon:
                nn_input_vec = np.array([s_vt, pv_v_end, pv_s_end])
            else:
                nn_input_vec = np.array([s_vt, pv_vt - s_vt, pv_st - s_st])
        if self.num_input == 4:
            nn_input_vec = np.array([s_vt, pv_v_end, pv_s_end, s_at])
        
        nn_input = torch.FloatTensor(nn_input_vec.T).cuda()
        
        # Compute the neural network control
        with torch.no_grad():
            ego_a_nn = self.nn_controller.forward(nn_input)
            s_a_nn = (ego_a_nn.flatten()).tolist()
        
        # Add damper to EcoNN control to prevent aggressive acceleration
        s_a_nn = (s_a_nn + lambda_smooth * s_at) / (1 + lambda_smooth)
        # Check if CBF safety constraint is violated
        a_ego_max = self.CBF_acceleration_bound_check(pv_vt=pv_vt, s_vt=s_vt, pv_st=pv_st, s_st=s_st, tao=1.5, alpha=2.0, L=7.0)
        if np.any(s_a_nn > a_ego_max):
            if self.print_level == "debug":
                print(f"CBF safety constraint is violated at time {sim_t}! Adjusting NN control to ensure safety...")
            ego_a_tgt = np.minimum(s_a_nn, a_ego_max)
        else:
            ego_a_tgt = s_a_nn

        return ego_a_tgt
