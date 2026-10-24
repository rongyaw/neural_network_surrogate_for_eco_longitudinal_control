clear
clc
close all
dbstop if error

%%  Do the analysis
error_record = [];
num_neurons_record = [];
num_layers_record = [];
runtime_record = [];

while true
    next_run = input("Do you need to record another runs? [0 or 1]");
    if next_run
        single_run_data_analysis;
        disp(['NN with ', num2str(num_layers), ' layers and ', num2str(num_neurons), ' hidden neurons']);
        error_record = [error_record, ego_v_error];
        num_layers_record = [num_layers_record, str2num(num_layers)];
        num_neurons_record = [num_neurons_record, str2num(num_neurons)];
        runtime_record = [runtime_record, avg_runtime];
        clearvars -except num_neurons_record num_layers_record error_record runtime_record
    else
        break
    end
end

%%  Check the error
num_neurons_mesh = linspace(64,1024,1000);
num_layers_mesh = linspace(2,4,50);
[num_neurons_meshgrid, num_layers_meshgrid] = meshgrid(num_neurons_mesh, num_layers_mesh);
F_error = scatteredInterpolant(num_layers_record.', num_neurons_record.', error_record.'/14000, 'linear');
F_runtime = scatteredInterpolant(num_layers_record.', num_neurons_record.', runtime_record.', 'linear');
error_interp = F_error(num_layers_meshgrid, num_neurons_meshgrid);
runtime_interp = F_runtime(num_layers_meshgrid, num_neurons_meshgrid);

figure(1)
scatter3(num_layers_record, num_neurons_record, error_record/14000, 10, 'filled');hold on
surf(num_layers_meshgrid, num_neurons_meshgrid, error_interp, 'EdgeColor', 'none');
xlabel('Number of hidden layers');ylabel('Number of neurons in each hidden layer');zlabel('Average speed error [m/s]')

figure(2)
scatter3(num_layers_record, num_neurons_record, runtime_record, 10, 'filled');hold on
surf(num_layers_meshgrid, num_neurons_meshgrid, runtime_interp, 'EdgeColor', 'none');
zlim([0.0,1.0])
xlabel('Number of hidden layers');ylabel('Number of neurons in each hidden layer');zlabel('Runtime [ms]')

%%  Save the data
data_to_save = [num_layers_record; num_neurons_record; error_record; runtime_record];
writematrix(data_to_save, 'nn_analysis_data.csv');