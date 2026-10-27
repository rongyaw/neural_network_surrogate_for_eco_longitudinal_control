"""Simple baseline controllers."""

import numpy as np


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
