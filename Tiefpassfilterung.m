%% TIEFPASS-FILTERUNG 

%clear memory
clear all;

%% 

pix_i = 219;
pix_j = 435;

% Anzahl der Bilder, die ausgewertet werden
k = 512;

% figure-Variable
nr_fig = 1;

%% Daten holen

%load seq
Data = load('/home/bill/Uni/Studienarbeit/GET/Simulation/video/compKSM01.mat');
% Temperaturwerte jedes Pixels
U_tp = Data.seq;

%% Daten anpassen

% m = 640x480 Pixel, n = 2048 Frames
[m,n] = size(U_tp);
tp_max = max(reshape(U_tp,m*n,1));
tp_min = min(reshape(U_tp,m*n,1));

%Skalieren der Werte von -1 bis 1 
% -1 = tp_max, +1 = tp_min
u_max = 1;
u_min = -1;
offset = u_max+tp_min*(u_max-u_min)/(tp_max-tp_min);
slope = -(u_max-u_min)/(tp_max-tp_min); 
% auf CNN angepasster Wertebereich
U = slope*U_tp+offset;

%reshape
U = reshape(U,480,640,2048);

%% FFT
% fs = 50; 
% ts = 1/fs; 
% length = 2048;
% t = (0:length-1)*ts;
% NFFT = 2^nextpow2(length);
% Y = fft(U(pix_i,pix_j,:),NFFT)/length;
% Y = squeeze(Y);
% f =fs/2*linspace(0,1,NFFT/2+1);
% figure(nr_fig);
% nr_fig = nr_fig + 1;
% semilogx(f,2*abs(Y(1:NFFT/2+1))); grid on;

%% CNN-Input und Template

%prinzipiell wären noch die virtuellen Zellen zu ergänzen
U = U(:,:,1:k);
%Anfangswerte
X0 = U(:,:,1);

%Grenzfrequenz des Tiefpass 1.Ordnung, TODO: höhere Ordnung
fg = 0.005; % [fg] = Hz
a_00 = 1-2*pi*fg;
b_00 = 1-a_00;
A = [ 0 0 0; 0 a_00 0; 0 0 0];
B = [ 0 0 0; 0 b_00 0; 0 0 0];
z = 0;

template = {A,B,z};

%% CNN-Settings
%h = ts;
h = 1;
N = 10;
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
    P = U(:,:,i);
    data = {X0,P};    
    % Systemantwort berechnen
    X(:,:,i) = cnn_operation(data, template, settings, method);
    % neuer Anfangswert des Integrators
    X0 = X(:,:,i);
end

%% Auswertung eines Pixels, TODO: Spektren sind wahrscheinlich aussagekräftiger
U_ij = squeeze(U(pix_i,pix_j,:));
X_ij = squeeze(X(pix_i,pix_j,:));

%ylabel('u_{ij}[k], y_{ij}[k]');
figure(nr_fig);
nr_fig = nr_fig+1;
plot(1:k,U_ij,'b',1:k,X_ij,'r'); 
title(['Zellkoordinaten: i = ' int2str(pix_i) ', j = ' int2str(pix_j)]);
legend('u_{ij}[k]', 'y_{ij}[k]');
xlabel('k');
grid on;

%% RMSE, Maß für alle Pixel

meas_U = evaluateSeries(U);
meas_X = evaluateSeries(X);

figure(nr_fig);
nr_fig = nr_fig + 1;
title('RMSE');
plot(1:k, meas_U.RMSEplot,'b',1:k, meas_X.RMSEplot,'r');
legend('U','X');
xlabel('k-tes Bild');
ylabel('Abweichung zu erstem Bild');
grid on;

%% FFT vom RMSE

fs = 50; 
ts = 1/fs; 
length = k;
t = (0:length-1)*ts;
NFFT = 2^nextpow2(length);
Y = fft(meas_U.RMSEplot,NFFT)/length;
Y = squeeze(Y);
f =fs/2*linspace(0,1,NFFT/2+1);
figure(nr_fig);
nr_fig = nr_fig + 1;
SpecRMSE.amp = 2*abs(Y(1:NFFT/2+1));
SpecRMSE.f = f;
plot(SpecRMSE.f,SpecRMSE.f); 
grid on;



%% Simulinkmodell 
% zur Verifikation der Ergebnisse des CNN-Simulators

% IN = double([ [1:k]' U_ij ]);
% %Anfangsbedingung
% x_ij0 = U_ij(1); 
% sim('Dynamik_Zelle_ij');



