%% TIEFPASS-FILTERUNG 

%clear memory
clear all;

%% 

% Sampling
fs = 50;
ts = 1/fs;

% Pixel
pix_i = 219;
pix_j = 435;

% Anzahl der Bilder, die ausgewertet werden
k = 256;

% figure-Variable
nr_fig = 1;

% Tiepass-Parameter
fg = 0.02; % Grenzfrequenz
Order = 1; % Anzahl der in Reihe geschalteten Zellen

%% Daten holen

%load seq
Data = load('/home/bill/Uni/Studienarbeit/GET/Simulation/video/compKSM01.mat');
% Temperaturwerte jedes Pixels
U_tp = Data.seq;
%wird nicht länger gebraucht
clear Data;

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
clear U_tp;
%reshape
U = reshape(U,480,640,2048);

%% CNN-Input und Template

%TODO: prinzipiell wären noch die virtuellen Zellen zu ergänzen

%Einlesen der von k-Bilder der Sequenz, da Programm bei 2048 abstürtzt
U = U(:,:,1:k);
%Anfangswerte
X0 = U(:,:,1);

% %debug
% U = zeros(480,640,k)+0.1;
% X0 = zeros(480,640);

%Berechnung der Koeffizienten, so dass bei fg -3dB Dämpfung, 
%im Fall von Order-In-Reihe geschalteten Zellen. Fall ist hier 
%allerdings nicht implementiert
a_00 = 1-2*pi*fg/sqrt(2^(1/Order)-1);
b_00 = 1-a_00;
A = [ 0 0 0; 0 a_00 0; 0 0 0];
B = [ 0 0 0; 0 b_00 0; 0 0 0];
z = 0;

template = {A,B,z};

%% CNN-Settings
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
    data = {X0,U(:,:,i)};    
    % Systemantwort berechnen
    X(:,:,i) = cnn_operation(data, template, settings, method);
    % neuer Anfangswert des Integrators
    X0 = X(:,:,i);
end

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

%% Auswertung eines Pixels
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

%% DFT der Zeitreihe des Pixels

length = k;
t = (0:length-1)*ts;
NFFT = 2^nextpow2(length);
Uf_ij = squeeze(fft(U_ij,NFFT)/length);
Xf_ij = squeeze(fft(X_ij,NFFT)/length);
Gf_ij = Xf_ij./Uf_ij;
f =fs/2*linspace(0,1,NFFT/2+1);
figure(nr_fig);
nr_fig = nr_fig + 1;
Xfabs = 2*abs(Xf_ij(1:NFFT/2+1));
Ufabs = 2*abs(Uf_ij(1:NFFT/2+1));

%semilogy(f,Xfabs./Ufabs); 
loglog(f,Xfabs./Ufabs)
xlabel('f/Hz');
%xlim([0 5]);
ylabel('|X_{ij}(f)/U_{ij}(f)|');
grid on;

%% Summe der Amplitudenbetragsquadrate des Pixels in den Bändern: 
% % [0.01-0.1]Hz, 0.125-0.5Hz, 0.8-1.4Hz, 1.8-2.5Hz
% figure(nr_fig);
% % welche an welchen Frequenzen wird Amplitudenspektrum berechnet? 
% f = meas_X.settings.powerArray(:,1); 
% % remove singleton dimension
% Xf = squeeze(meas_X.IFS(pix_i,pix_j,:));
% Uf = squeeze(meas_U.IFS(pix_i,pix_j,:));
% 
% Gf = Xf./Uf;
% 
% plot(f,Gf);
% xlabel('f/Hz');
% ylabel('|X_{ij}(f)/U_{ij}(f)|');
% grid on;

%% FFT vom RMSE

% fs = 50; 
% ts = 1/fs; 
% length = k;
% t = (0:length-1)*ts;
% NFFT = 2^nextpow2(length);
% Y = fft(meas_U.RMSEplot,NFFT)/length;
% Y = squeeze(Y);
% f =fs/2*linspace(0,1,NFFT/2+1);
% figure(nr_fig);
% nr_fig = nr_fig + 1;
% SpecRMSE.amp = 2*abs(Y(1:NFFT/2+1));
% SpecRMSE.f = f;
% plot(SpecRMSE.f,SpecRMSE.amp); 
% grid on;
