# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple


# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

def torch_wrap_to_pi_minuspi(angles):
    angles = angles % (2 * np.pi)
    angles -= 2 * np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles

def euler_from_quat(quat_angle):
    """
    + Convert a quaternion into euler angles (roll, pitch, yaw)
    + roll is rotation around x in radians (counterclockwise)
    + pitch is rotation around y in radians (counterclockwise)
    + yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians

def quat_from_euler_xyz(roll, pitch, yaw):
    """
    Convert euler angles (roll, pitch, yaw) to quaternion
    + roll is rotation around x in radians (counterclockwise)
    + pitch is rotation around y in radians (counterclockwise)
    + yaw is rotation around z in radians (counterclockwise)
    """
    # Convert to half angles
    roll_half = roll * 0.5
    pitch_half = pitch * 0.5
    yaw_half = yaw * 0.5
    
    # Compute trigonometric functions
    cr = torch.cos(roll_half)
    sr = torch.sin(roll_half)
    cp = torch.cos(pitch_half)
    sp = torch.sin(pitch_half)
    cy = torch.cos(yaw_half)
    sy = torch.sin(yaw_half)
    
    # Compute quaternion components
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    # Stack into quaternion tensor
    quat = torch.stack([qx, qy, qz, qw], dim=-1)
    
    return quat

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0.0, -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.0) / 2.0
    return (upper - lower) * r + lower

class CubicSpline:
    def __init__(self, start, end):
        self.t0 = start['time']
        self.t1 = end['time']
        self.dt = end['time'] - start['time']

        dp = end['position'] - start['position']
        dv = end['velocity'] - start['velocity']

        self.dc0 = torch.tensor(0.0)
        self.dc1 = start['velocity']
        self.dc2 = -(3.0 * start['velocity'] + dv)
        self.dc3 = (2.0 * start['velocity'] + dv)

        self.c0 = self.dc0 * self.dt + start['position']
        self.c1 = self.dc1 * self.dt
        self.c2 = self.dc2 * self.dt + 3.0 * dp
        self.c3 = self.dc3 * self.dt - 2.0 * dp

    def position(self, time):
        tn = self.normalized_time(time)
        return self.c3 * tn ** 3 + self.c2 * tn ** 2 + self.c1 * tn + self.c0

    def velocity(self, time):
        tn = self.normalized_time(time)
        return (3.0 * self.c3 * tn ** 2 + 2.0 * self.c2 * tn + self.c1) / self.dt

    def acceleration(self, time):
        tn = self.normalized_time(time)
        return (6.0 * self.c3 * tn + 2.0 * self.c2) / (self.dt ** 2)

    def start_time_derivative(self, t):
        tn = self.normalized_time(t)
        dCoff = -(self.dc3 * tn ** 3 + self.dc2 * tn ** 2 + self.dc1 * tn + self.dc0)
        dTn = -(self.t1 - t) / (self.dt ** 2)
        return self.velocity(t) * self.dt * dTn + dCoff

    def final_time_derivative(self, t):
        tn = self.normalized_time(t)
        dCoff = (self.dc3 * tn ** 3 + self.dc2 * tn ** 2 + self.dc1 * tn + self.dc0)
        dTn = -(t - self.t0) / (self.dt ** 2)
        return self.velocity(t) * self.dt * dTn + dCoff

    def normalized_time(self, t):
        return (t - self.t0) / self.dt