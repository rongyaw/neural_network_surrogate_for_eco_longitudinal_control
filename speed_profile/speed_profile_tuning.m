%%  This script is intended to fine tune speed profile for front vehicles
% The original driving cycle text files can be found in this website:https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules
clear
clc
close all
dbstop if error
save_data = 1;

%%  Load the data
spd_data_raw = readtable('im240col.txt');
t_raw = spd_data_raw.Var1;
spd_raw = spd_data_raw.Var2 * 0.447;

%%  Sanity check
figure(1)
subplot(2,1,1)
plot(t_raw, spd_raw, 'LineWidth', 2);hold on
xlabel('Time [s]');ylabel('Speed [m/s]')

%%  Decrease time interval
dt = 0.1;
t = t_raw(1) : dt : t_raw(end);
spd = interp1(t_raw, spd_raw, t, "makima") / 1.5;

% Estimate the acceleration and distance travelled
acc = [diff(spd) / dt, 0];
dist = cumsum(spd * dt);
figure(1)
subplot(2,1,1)
plot(t, spd, '--', 'LineWidth', 3);

subplot(2,1,2)
plot(t, acc, '-k', 'LineWidth', 2);
xlabel('Time [s]');ylabel('Acceleration [m/s^{2}]')

%%  Save data
filename = "I85_im240.csv";
data_to_save = [t.', spd.', acc.', dist.'];
if save_data
    writematrix(data_to_save,filename);
end