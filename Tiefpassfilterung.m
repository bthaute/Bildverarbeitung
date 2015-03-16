%% TIEFPASS-FILTERUNG

% wo soll die laufen? auf Co-Prozessor? 

%% Input

%clear memory
clear;

%load seq
load('/home/bill/Uni/Studienarbeit/GET/Simulation/video/compKSM01.mat');
[m,n] = size(seq);
max_val = max(reshape(seq,m*n,1));
min_val = min(reshape(seq,m*n,1));

%Skalieren der Werte von -1 bis 1
ymax = 1;
ymin = -1;

o = 1-(ymax-ymin)*max_val/(max_val-min_val);
s = (ymax-ymin)/(max_val-min_val);
seq = s*seq+o;



%% Template
b_00 = 1;
a_00 = 0.95;
A = [ 0 0 0; 0 a_00 0; 0 0 0];
B = [ 0 0 0; 0 b_00 0; 0 0 0];
z = 0;


template = {A,B,z};

X0 = zeros(480,640);


%% Eigentliche Filterung


%% Evaluation der Sequenz