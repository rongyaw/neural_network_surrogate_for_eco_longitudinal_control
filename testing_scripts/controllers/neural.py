"""Neural-network car-following controllers."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class NN_controller():
    def __init__(self, nn_pt_file, input_num, print_level="info"):
        self.num_input = input_num
        self.print_level = print_level
        self.nn_controller = Model(h1=512, h2=512, h3=512, h4=512, h5=512)
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
