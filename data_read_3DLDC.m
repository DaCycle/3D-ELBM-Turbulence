clc; clear; close all;

%% Collect data
snapread = 100000;
info = dir(sprintf('snapshots/Density_%06d.dat', snapread));
N = int64((info.bytes/8)^(1/3));
N_x = N;
N_y = N;
N_z = N;

%% Initialize Constants
c_s = 1/sqrt(3); % Speed of Sound for D2Q9
U_lid = 0.05 * c_s;
Re = 100;   % Reynolds number
Beta = 1 / (6*U_lid*double(N)/Re + 1); % Relaxation time (must be >0.5)
% U_lid = Re*(0.5/Beta-0.5)*c_s^2/double(N);

%% Regular Plots
fid_Rho = fopen(sprintf('snapshots/Density_%06d.dat', snapread));
fid_XVel = fopen(sprintf('snapshots/X_Velocity_%06d.dat', snapread));
fid_YVel = fopen(sprintf('snapshots/Y_Velocity_%06d.dat', snapread));
fid_ZVel = fopen(sprintf('snapshots/Z_Velocity_%06d.dat', snapread));
% fid_Visc = fopen(sprintf('snapshots/Viscousity_%06d.dat',snapread));
% fid_pdf = fopen(sprintf('snapshots/pdf_%06d.dat',snapread));
raw_Rho = fread(fid_Rho, 'double');
raw_U = fread(fid_XVel, 'double');
raw_V = fread(fid_YVel, 'double');
raw_W = fread(fid_ZVel, 'double');
% raw_Visc = fread(fid_Visc, 'double');
% raw_pdf = fread(fid_pdf, 'double');
fclose(fid_Rho);
fclose(fid_XVel);
fclose(fid_YVel);
fclose(fid_ZVel);
% fclose(fid_Visc);
% fclose(fid_pdf);
Rho = reshape(raw_Rho, N, N, N);
U = reshape(raw_U, N, N, N);
V = reshape(raw_V, N, N, N);
W = reshape(raw_W, N, N, N);
% Visc = reshape(raw_Visc, N, N, N)';
% pdf = reshape(raw_pdf, N, N, N, 27);
Rho = permute(Rho, [3 2 1]);
U = permute(U, [3 2 1]);
V = permute(V, [3 2 1]);
W = permute(W, [3 2 1]);


%% Plot data
y_plot = [1 round(N/4) round(N/2) round(3*N/4) N];

for i = 1:length(y_plot)
    % Extract and plot the density data for the current slice
    figure
    contourf(squeeze(Rho(:,y_plot(i),:)), 30)
    title(sprintf('Density at y = %d', y_plot(i)));
    xlim([-0.2*N N*1.2])
    ylim([-0.2*N N*1.2])
    axis equal tight
    colorbar;
end

for i = 1:length(y_plot)
    % Extract and plot the velocity data for the current slice
    figure
    quiver(squeeze(U(:,y_plot(i),:)), squeeze(W(:,y_plot(i),:)), 10)
    title(sprintf('Velocity Field at y = %d', y_plot(i)));
    xlim([-0.2*N N*1.2])
    ylim([-0.2*N N*1.2])
    axis equal tight
end

% [X, Y, Z] = meshgrid(1:double(N), 1:double(N), 1:double(N));
% figure; hold on
% skip = 15;
% for j = 1:skip:N
%     quiver3( ...
%         X(j,:,:), ...
%         Y(j,:,:), ...
%         Z(j,:,:), ...
%         U(j,:,:), ...
%         V(j,:,:), ...
%         W(j,:,:), ...
%         'Color','b');
% end
% axis equal
% view(3)