"""Preview neural-network controller."""

import numpy as np
import torch
import torch.nn as nn


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
