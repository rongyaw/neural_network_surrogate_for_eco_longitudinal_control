# /usr/bin/env python3

import csv
import os

import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_LAYER_WIDTH = 1024
INPUT_FEATURES = 41
TRAJECTORY_OUTPUT_FEATURES = 38
ACCELERATION_OUTPUT_FEATURES = 1
PREVIEW_STEPS = 20
PREVIEW_CHANNELS = 2
EGO_FEATURES = 1
RAW_DISTANCE_HEADWAY_START = 0
RAW_EGO_V_INDEX = PREVIEW_STEPS
RAW_SPEED_GAP_START = RAW_EGO_V_INDEX + 1
CONV_CHANNEL_WIDTH = 32
ACCELERATION_HIDDEN_LAYER_WIDTH_1 = 1024
ACCELERATION_HIDDEN_LAYER_WIDTH_2 = 1024
TRAJECTORY_LOSS_WEIGHT = 1.0
ACCELERATION_LOSS_WEIGHT = 1.0
MODEL_FILENAME = "preview_traffic_following_control_conv.pt"
BEST_MODEL_FILENAME = "preview_traffic_following_control_conv_best.pt"


def reorder_input_features(row):
    if len(row) != INPUT_FEATURES:
        raise ValueError(
            "Unexpected input feature count: expected "
            + str(INPUT_FEATURES)
            + ", got "
            + str(len(row))
        )

    reordered_row = [row[RAW_EGO_V_INDEX]]
    for step in range(PREVIEW_STEPS):
        reordered_row.append(row[RAW_DISTANCE_HEADWAY_START + step])
        reordered_row.append(row[RAW_SPEED_GAP_START + step])

    return reordered_row


def split_input_features(model_input):
    if model_input.shape[1] != INPUT_FEATURES:
        raise ValueError(
            "Unexpected model input feature count: expected "
            + str(INPUT_FEATURES)
            + ", got "
            + str(model_input.shape[1])
        )

    ego_v = model_input[:, :EGO_FEATURES]
    preview_sequence = model_input[:, EGO_FEATURES:].reshape(
        -1, PREVIEW_STEPS, PREVIEW_CHANNELS
    )
    preview_sequence = preview_sequence.transpose(1, 2).contiguous()
    return ego_v, preview_sequence


class Model(nn.Module):
    def __init__(
        self,
        in_features=INPUT_FEATURES,
        preview_steps=PREVIEW_STEPS,
        preview_channels=PREVIEW_CHANNELS,
        ego_features=EGO_FEATURES,
        conv_channels=CONV_CHANNEL_WIDTH,
        h1=1024,
        h2=1024,
        trajectory_out_features=TRAJECTORY_OUTPUT_FEATURES,
        acceleration_h1=ACCELERATION_HIDDEN_LAYER_WIDTH_1,
        acceleration_h2=ACCELERATION_HIDDEN_LAYER_WIDTH_2,
        acceleration_out_features=ACCELERATION_OUTPUT_FEATURES,
    ):
        super().__init__()
        if in_features != ego_features + preview_steps * preview_channels:
            raise ValueError("Input feature setup does not match preview sequence layout.")

        self.preview_steps = preview_steps
        self.preview_channels = preview_channels
        self.ego_features = ego_features

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
        self.dp = nn.Dropout(0.3)
        self.loss = nn.MSELoss()

    def forward(self, x):
        ego_v, preview_sequence = split_input_features(x)
        preview_features = self.conv_activation(self.conv1(preview_sequence))
        preview_features = self.conv_pool(preview_features)
        preview_features = torch.flatten(preview_features, start_dim=1)

        x = torch.cat((ego_v, preview_features), dim=1)
        x = torch.sigmoid(self.fc1(x))
        x = self.dp(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.dp(x)
        joint_prediction = self.output(x)
        trajectory_prediction = joint_prediction[:, :TRAJECTORY_OUTPUT_FEATURES]
        acceleration_prediction = joint_prediction[:, TRAJECTORY_OUTPUT_FEATURES:]
        return trajectory_prediction, acceleration_prediction

    def compute_losses(self, trajectory_gt, acceleration_gt, model_input):
        trajectory_prediction, acceleration_prediction = self.forward(model_input)
        trajectory_loss = self.loss(trajectory_prediction, trajectory_gt)
        acceleration_loss = self.loss(acceleration_prediction, acceleration_gt)
        total_loss = (
            TRAJECTORY_LOSS_WEIGHT * trajectory_loss
            + ACCELERATION_LOSS_WEIGHT * acceleration_loss
        )
        return total_loss, trajectory_loss, acceleration_loss


def load_csv_matrix(filename, reorder_input=False):
    matrix = []
    with open(filename, newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        if not header:
            raise ValueError(f"Empty CSV header in {filename}")

        for row in reader:
            if not row:
                continue
            sample = [float(value) for value in row[1:]]
            if reorder_input:
                sample = reorder_input_features(sample)
            matrix.append(sample)

    if not matrix:
        raise ValueError(f"No samples found in {filename}")

    return matrix


def load_dataset(input_filename, trajectory_filename, acceleration_filename):
    input_matrix = load_csv_matrix(input_filename, reorder_input=True)
    trajectory_matrix = load_csv_matrix(trajectory_filename)
    acceleration_matrix = load_csv_matrix(acceleration_filename)

    if len(input_matrix) != len(trajectory_matrix):
        raise ValueError(
            f"Input/target row mismatch: {len(input_matrix)} inputs vs "
            f"{len(trajectory_matrix)} trajectory targets"
        )
    if len(input_matrix) != len(acceleration_matrix):
        raise ValueError(
            f"Input/output row mismatch: {len(input_matrix)} inputs vs "
            f"{len(acceleration_matrix)} acceleration targets"
        )

    return input_matrix, trajectory_matrix, acceleration_matrix


if __name__ == "__main__":
    print(torch.cuda.is_available())

    dirname = os.path.dirname(__file__)
    train_input_filename = os.path.join(dirname, "nn_training_inputs_train.csv")
    train_trajectory_filename = os.path.join(dirname, "nn_training_targets_train.csv")
    train_acceleration_filename = os.path.join(dirname, "nn_training_output_train.csv")
    test_input_filename = os.path.join(dirname, "nn_training_inputs_test.csv")
    test_trajectory_filename = os.path.join(dirname, "nn_training_targets_test.csv")
    test_acceleration_filename = os.path.join(dirname, "nn_training_output_test.csv")
    valid_input_filename = os.path.join(dirname, "nn_training_inputs_validation.csv")
    valid_trajectory_filename = os.path.join(dirname, "nn_training_targets_validation.csv")
    valid_acceleration_filename = os.path.join(dirname, "nn_training_output_validation.csv")

    train_x, train_trajectory_y, train_acceleration_y = load_dataset(
        train_input_filename,
        train_trajectory_filename,
        train_acceleration_filename,
    )
    test_x, test_trajectory_y, test_acceleration_y = load_dataset(
        test_input_filename,
        test_trajectory_filename,
        test_acceleration_filename,
    )
    valid_x, valid_trajectory_y, valid_acceleration_y = load_dataset(
        valid_input_filename,
        valid_trajectory_filename,
        valid_acceleration_filename,
    )

    train_x = torch.FloatTensor(train_x).to(device)
    train_trajectory_y = torch.FloatTensor(train_trajectory_y).to(device)
    train_acceleration_y = torch.FloatTensor(train_acceleration_y).to(device)
    test_x = torch.FloatTensor(test_x).to(device)
    test_trajectory_y = torch.FloatTensor(test_trajectory_y).to(device)
    test_acceleration_y = torch.FloatTensor(test_acceleration_y).to(device)
    valid_x = torch.FloatTensor(valid_x).to(device)
    valid_trajectory_y = torch.FloatTensor(valid_trajectory_y).to(device)
    valid_acceleration_y = torch.FloatTensor(valid_acceleration_y).to(device)

    if train_x.shape[1] != INPUT_FEATURES:
        raise ValueError(
            "Unexpected training input feature count: expected "
            + str(INPUT_FEATURES)
            + ", got "
            + str(train_x.shape[1])
        )
    if valid_x.shape[1] != INPUT_FEATURES or test_x.shape[1] != INPUT_FEATURES:
        raise ValueError("Validation/test input feature count does not match training setup.")
    if train_trajectory_y.shape[1] != TRAJECTORY_OUTPUT_FEATURES:
        raise ValueError(
            "Unexpected training trajectory target feature count: expected "
            + str(TRAJECTORY_OUTPUT_FEATURES)
            + ", got "
            + str(train_trajectory_y.shape[1])
        )
    if (
        valid_trajectory_y.shape[1] != TRAJECTORY_OUTPUT_FEATURES
        or test_trajectory_y.shape[1] != TRAJECTORY_OUTPUT_FEATURES
    ):
        raise ValueError(
            "Validation/test trajectory target feature count does not match training setup."
        )
    if train_acceleration_y.shape[1] != ACCELERATION_OUTPUT_FEATURES:
        raise ValueError(
            "Unexpected training acceleration target feature count: expected "
            + str(ACCELERATION_OUTPUT_FEATURES)
            + ", got "
            + str(train_acceleration_y.shape[1])
        )
    if (
        valid_acceleration_y.shape[1] != ACCELERATION_OUTPUT_FEATURES
        or test_acceleration_y.shape[1] != ACCELERATION_OUTPUT_FEATURES
    ):
        raise ValueError(
            "Validation/test acceleration target feature count does not match training setup."
        )

    model = Model(
        in_features=INPUT_FEATURES,
        preview_steps=PREVIEW_STEPS,
        preview_channels=PREVIEW_CHANNELS,
        ego_features=EGO_FEATURES,
        conv_channels=CONV_CHANNEL_WIDTH,
        h1=HIDDEN_LAYER_WIDTH,
        h2=HIDDEN_LAYER_WIDTH,
        trajectory_out_features=TRAJECTORY_OUTPUT_FEATURES,
        acceleration_h1=ACCELERATION_HIDDEN_LAYER_WIDTH_1,
        acceleration_h2=ACCELERATION_HIDDEN_LAYER_WIDTH_2,
        acceleration_out_features=ACCELERATION_OUTPUT_FEATURES,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.000001)
    epochs = 500
    batch_size = 512
    loss_record = []
    trajectory_loss_record = []
    acceleration_loss_record = []
    validation_loss_record = []
    validation_trajectory_loss_record = []
    validation_acceleration_loss_record = []
    best_valid_loss = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_x.size(0), device=device)
        epoch_loss = 0.0
        epoch_trajectory_loss = 0.0
        epoch_acceleration_loss = 0.0

        for batch_start in range(0, train_x.size(0), batch_size):
            batch_indices = permutation[batch_start : batch_start + batch_size]
            batch_x = train_x[batch_indices]
            batch_trajectory_y = train_trajectory_y[batch_indices]
            batch_acceleration_y = train_acceleration_y[batch_indices]

            optimizer.zero_grad()
            total_loss, trajectory_loss, acceleration_loss = model.compute_losses(
                batch_trajectory_y,
                batch_acceleration_y,
                batch_x,
            )

            if not torch.isnan(total_loss):
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item() * batch_x.size(0)
                epoch_trajectory_loss += trajectory_loss.item() * batch_x.size(0)
                epoch_acceleration_loss += acceleration_loss.item() * batch_x.size(0)

        epoch_loss /= train_x.size(0)
        epoch_trajectory_loss /= train_x.size(0)
        epoch_acceleration_loss /= train_x.size(0)
        loss_record.append(epoch_loss)
        trajectory_loss_record.append(epoch_trajectory_loss)
        acceleration_loss_record.append(epoch_acceleration_loss)

        model.eval()
        with torch.no_grad():
            (
                valid_loss_tensor,
                valid_trajectory_loss_tensor,
                valid_acceleration_loss_tensor,
            ) = model.compute_losses(
                valid_trajectory_y,
                valid_acceleration_y,
                valid_x,
            )
            valid_loss = valid_loss_tensor.item()
            valid_trajectory_loss = valid_trajectory_loss_tensor.item()
            valid_acceleration_loss = valid_acceleration_loss_tensor.item()
        validation_loss_record.append(valid_loss)
        validation_trajectory_loss_record.append(valid_trajectory_loss)
        validation_acceleration_loss_record.append(valid_acceleration_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            best_model_filename = os.path.join(dirname, BEST_MODEL_FILENAME)
            torch.save(model.state_dict(), best_model_filename)

        print(
            "Epoch "
            + str(epoch + 1)
            + "/"
            + str(epochs)
            + " train_total="
            + str(round(epoch_loss, 4))
            + " train_traj="
            + str(round(epoch_trajectory_loss, 4))
            + " train_acc="
            + str(round(epoch_acceleration_loss, 4))
            + " valid_total="
            + str(round(valid_loss, 4))
            + " valid_traj="
            + str(round(valid_trajectory_loss, 4))
            + " valid_acc="
            + str(round(valid_acceleration_loss, 4)),
            end="\r",
        )

    print("\nTraining Finished.")

    if plt is not None:
        plt.style.use("bmh")
        plt.figure(1)
        plt.plot(range(len(loss_record)), loss_record, label="train_total")
        plt.plot(range(len(validation_loss_record)), validation_loss_record, label="validation_total")
        plt.plot(range(len(trajectory_loss_record)), trajectory_loss_record, label="train_trajectory")
        plt.plot(
            range(len(validation_trajectory_loss_record)),
            validation_trajectory_loss_record,
            label="validation_trajectory",
        )
        plt.plot(range(len(acceleration_loss_record)), acceleration_loss_record, label="train_acceleration")
        plt.plot(
            range(len(validation_acceleration_loss_record)),
            validation_acceleration_loss_record,
            label="validation_acceleration",
        )
        plt.xlabel("Epoch")
        plt.ylabel("MSE LOSS")
        plt.legend()
        plt.show()
    else:
        print("matplotlib is not installed, skipping loss plot display.")

    model.load_state_dict(torch.load(os.path.join(dirname, BEST_MODEL_FILENAME), map_location=device))
    model.eval()
    with torch.no_grad():
        test_total_loss, test_trajectory_loss, test_acceleration_loss = model.compute_losses(
            test_trajectory_y,
            test_acceleration_y,
            test_x,
        )
    print("Test total loss: " + str(round(test_total_loss.item(), 4)))
    print("Test trajectory loss: " + str(round(test_trajectory_loss.item(), 4)))
    print("Test acceleration loss: " + str(round(test_acceleration_loss.item(), 4)))
    print(
        "Best validation loss: "
        + str(round(best_valid_loss, 4))
        + " at epoch "
        + str(best_epoch)
    )
    
    model_filename = os.path.join(dirname, MODEL_FILENAME)
    torch.save(model.state_dict(), model_filename)
    print("Saved model to " + model_filename)
    print("Saved best validation model to " + os.path.join(dirname, BEST_MODEL_FILENAME))
