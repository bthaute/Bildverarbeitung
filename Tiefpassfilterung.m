%% TIEFPASS-FILTERUNG 

%clear memory
clear all;

%% Data

%load seq
Data = load('C:\Users\Bill\Documents\Universität\Studienarbeit\GET\Simulation\video\compKSM01.mat');
Seq = Data.seq;
% m = 640x480 Pixel, n = 2048 Frames
[m,n] = size(Seq);
tp_max = max(reshape(Seq,m*n,1));
tp_min = min(reshape(Seq,m*n,1));

%Skalieren der Werte von -1 bis 1 
% -1 = tp_max, +1 = tp_min
u_max = 1;
u_min = -1;
offset = u_max+tp_min*(u_max-u_min)/(tp_max-tp_min);
slope = -(u_max-u_min)/(tp_max-tp_min); 
Seq = slope*Seq+offset;

%reshape
Seq = reshape(Seq,480,640,2048);

% cast wirklich notwendig?
% Seq = double(Seq); %Speicher läuft voll und Matlab stürtzt ab
k = 200;

Seq = Seq(:,:,1:k);
%Anfangswerte
X0 = Seq(:,:,1);
%% Template
b_00 = 0.05;
a_00 = 0.95;
A = [ 0 0 0; 0 a_00 0; 0 0 0];
B = [ 0 0 0; 0 b_00 0; 0 0 0];
z = 0;

template = {A,B,z};

%% Settings
h = 1;
N = 1;
bc = 'dirichlet';
func = 'limit';
K = 1;
val = 0;
fsr = 0;
fixp = 0;
dw = 12;
round_mode = 'round';

settings = {h,N,bc,func,K,val,fsr,fixp,dw,round_mode};
method = 'euler';

%% Filterung ueber alle Bilder der Sequenz
for i=1:k    
    % nächstes Bild in der Sequenz
    P = Seq(:,:,i);
    data = {X0,P};    
    % Systemantwort berechnen
    X(:,:,i) = cnn_operation(data, template, settings, method);
end

%% Plot 
pix_i = 300;
pix_j = 300;
Seq_ij = squeeze(Seq(pix_i,pix_j,:));
X_ij = squeeze(X(pix_i,pix_j,:));

%ylabel('u_{ij}[k], y_{ij}[k]');
plot(1:k,Seq_ij,'b',1:k,X_ij,'r'); 
title(['Zellkoordinaten: i = ' int2str(pix_i) ', j = ' int2str(pix_j)]);
legend('u_{ij}[k]', 'y_{ij}[k]');
xlabel('k');
grid on;


%% Simulinkmodell 
% zur Verifikation der Ergebnisse des CNN-Simulators

% IN = double([ [1:k]' Seq_ij ]);
% %Anfangsbedingung
% x_ij0 = Seq_ij(1); 
% sim('Dynamik_Zelle_ij');



