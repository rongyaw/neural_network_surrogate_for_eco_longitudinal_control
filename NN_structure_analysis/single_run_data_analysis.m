%%  Section 1: Choose data
traffic_info_data = uigetfile(".csv");
file_split = split(traffic_info_data, ["_", "."]);
mpc_file = "Ftp_Online_MPC.csv";

%%  Section 2: Load data
data = load(traffic_info_data);
num_neurons = file_split{4};
num_layers = file_split{6};
runtime_data_filename = strcat(['Runtime_traffic_following_control_', num2str(num_neurons), '_x_', num2str(num_layers), '.pt.csv']);
runtime_data = load(runtime_data_filename);
mpc_data = load(mpc_file);

%%  Section 2: Check the car following data
sim_t = data(:, 1);
ego_a = data(:, 2);
ego_v = data(:, 3);
ego_s = data(:, 4);
pv_a = data(:, 5);
pv_v = data(:, 6);
pv_s = data(:, 7);

ego_v_ref = mpc_data(:, 3);
ego_s_ref = mpc_data(:, 4);

% Sanity check

figure(1)
plot(sim_t, pv_v);hold on
plot(sim_t, ego_v);
plot(sim_t, ego_v_ref);hold off

% Compute total difference
ego_v_error = sum(abs(ego_v - ego_v_ref));
avg_runtime = runtime_data(2);