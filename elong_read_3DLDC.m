clc; clear; close all;
%% Collect data
snapread = 100000;
FW_freq = 1000;
N = 64;
EL1 = load("LDC_Resolution64^2x64_Iterations500000.10000.mat");
EL2 = load("LDC_Resolution64^2x128_Iterations100000.1000.mat");
EL3 = load("LDC_Resolution64^2x192_Iterations100000.1000.mat");

%%
U_lid = EL1.U_lid;

U1 = EL1.U;
U2 = EL2.U;
U3 = EL3.U;

W1 = EL1.W;
W2 = EL2.W;
W3 = EL3.W;

Ghia = load("Ghia_Re100.mat");
Gu = Ghia.u_Ghia;
Gy = Ghia.y_Ghia;
Gv = Ghia.v_Ghia;
Gx = Ghia.x_Ghia;

%% Plot data
figure
plot(Gx*2-1,Gv+0.5,'X','color','black')
hold on
plot(Gu,Gy,'^','color','black')
hold on

U_f2d = squeeze(U1(:,round(N*1/2),:,10));
u_sim = U_f2d(:,ceil(N/2));
z_sim = linspace(0,1,N);
plot(rot90(u_sim/U_lid,1), z_sim,'red')
hold on
W_f2d = squeeze(W1(:,round(N*1/2),:,10));
w_sim = W_f2d(ceil(N/2), :) ./ 2;
x_sim = linspace(-1,1,N);
plot(x_sim, rot90(w_sim/U_lid,3)+0.5, 'Color', 'red','Linestyle', '--')
hold on

U_f2d = squeeze(U2(:,round(N*2/2),:,snapread/FW_freq));
u_sim = U_f2d(:,ceil(N/2));
z_sim = linspace(0,1,N);
plot(rot90(u_sim/U_lid,1), z_sim,'blue')
hold on
W_f2d = squeeze(W2(:,round(N*2/2),:,snapread/FW_freq));
w_sim = W_f2d(ceil(N/2), :) ./ 2;
x_sim = linspace(-1,1,N);
plot(x_sim, rot90(w_sim/U_lid,3)+0.5, 'Color', 'blue','Linestyle', '--')
hold on

U_f2d = squeeze(U3(:,round(N*3/2),:,snapread/FW_freq));
u_sim = U_f2d(:,ceil(N/2));
z_sim = linspace(0,1,N);
plot(rot90(u_sim/U_lid,1), z_sim,'green')
hold on
W_f2d = squeeze(W3(:,round(N*3/2),:,snapread/FW_freq));
w_sim = W_f2d(ceil(N/2), :) ./ 2;
x_sim = linspace(-1,1,N);
plot(x_sim, rot90(w_sim/U_lid,3)+0.5, 'Color', 'green','Linestyle', '--')
hold on

legend('','','1X Elongation', '', '2X Elongation', '','3X Elongation')
axis tight