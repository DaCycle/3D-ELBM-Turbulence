clc; clear; close all;

%% Collect data
snapread = 74000;
info = dir(sprintf('snapshots/Density_%06d.dat', snapread));
N = int64((info.bytes/8)^(1/3));
N_x = N;
N_y = N;
N_z = N;

%% Initialize Constants
c_s = 1/sqrt(3); % Speed of Sound for D2Q9
U_lid = 0.05 * c_s;
Re = 5000;   % Reynolds number
Beta = 1 / (6*U_lid*double(N)/Re + 1); % Relaxation time (must be >0.5)
% U_lid = Re*(0.5/Beta-0.5)*c_s^2/double(N);

%% Save Data to Matlab
max_iterations = snapread;
FW_frequency = 1000;
CaseName = sprintf('LDC_Resolution%d^3_Iterations%d.%d.mat', N, max_iterations, FW_frequency);
Rho = zeros(N,N,N,max_iterations/FW_frequency);
U = zeros(N,N,N,max_iterations/FW_frequency);
V = zeros(N,N,N,max_iterations/FW_frequency);
W = zeros(N,N,N,max_iterations/FW_frequency);
for i = 1:floor(max_iterations/FW_frequency)
    fid_Rho = fopen(sprintf('snapshots/Density_%06d.dat', i*FW_frequency));
    fid_U = fopen(sprintf('snapshots/X_Velocity_%06d.dat', i*FW_frequency));
    fid_V = fopen(sprintf('snapshots/Y_Velocity_%06d.dat', i*FW_frequency));
    fid_W = fopen(sprintf('snapshots/Z_Velocity_%06d.dat', i*FW_frequency));
    raw_Rho = fread(fid_Rho, 'double');
    raw_U = fread(fid_U, 'double');
    raw_V = fread(fid_V, 'double');
    raw_W = fread(fid_W, 'double');
    fclose(fid_Rho);
    fclose(fid_U);
    fclose(fid_V);
    fclose(fid_W);
    
    Rho_save = reshape(raw_Rho, N, N, N);
    U_save = reshape(raw_U, N, N, N);
    V_save = reshape(raw_V, N, N, N);
    W_save = reshape(raw_W, N, N, N);
    Rho_save = permute(Rho_save, [3 2 1]);
    U_save = permute(U_save, [3 2 1]);
    V_save = permute(V_save, [3 2 1]);
    W_save = permute(W_save, [3 2 1]);
    
    Rho(:,:,:,i) = Rho_save;
    U(:,:,:,i) = U_save;
    V(:,:,:,i) = V_save;
    W(:,:,:,i) = W_save;
end
save(CaseName, "Rho", "U", "V", "W", "Beta", "Re", "U_lid",'-v7.3')

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

% Density Contour
for i = 1:length(y_plot)
    figure
    contourf(squeeze(Rho(:,y_plot(i),:)), 30)
    title(sprintf('Density at y = %d', y_plot(i)));
    xlim([-0.2*N N*1.2])
    ylim([-0.2*N N*1.2])
    axis equal tight
    colorbar;
end

% Velocity Quiver
for i = 1:length(y_plot)
    figure
    quiver(squeeze(U(:,y_plot(i),:)), squeeze(W(:,y_plot(i),:)), 10)
    title(sprintf('Velocity Field at y = %d', y_plot(i)));
    xlim([-0.2*N N*1.2])
    ylim([-0.2*N N*1.2])
    axis equal tight
end

% Velocity Profile
figure
img = imread('Plot_Backgrounds/Re5000_UW.png');
ax = axes;
imagesc(ax, [-1 1], [0 1], img)
set(ax, 'YDir', 'normal')   % important for correct orientation
hold on

U_f2d = squeeze(U(:,round(N/2),:));
u_sim = U_f2d(:,ceil(N/2));
z_sim = linspace(0,1,N);
plot(rot90(u_sim/U_lid,1), z_sim,'blue')
hold on

W_f2d = squeeze(W(:,round(N/2),:));
w_sim = W_f2d(ceil(N/2), :) ./ 2;
x_sim = linspace(-1,1,N);
plot(x_sim, rot90(w_sim/U_lid,3)+0.5,'green')

axis tight

% % Streamlines
% N = double(N);
% % Grid (must match matrix ordering!)
% [x,y,z] = meshgrid(1:N, 1:N, 1:N);
% 
% Nseed = 500;   % number of random seeds
% 
% startx = 1 + (N-2)*rand(Nseed,1);
% starty = 1 + (N-2)*rand(Nseed,1);
% startz = 1 + (N-2)*rand(Nseed,1);
% 
% % Plot streamlines
% figure
% streamline(x,y,z,U,V,W,startx,starty,startz)
% 
% axis equal
% view(3)
% xlabel('x'); ylabel('y'); zlabel('z');
% box on
