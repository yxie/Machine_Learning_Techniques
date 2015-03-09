clc
clear

D = dlmread('hw4_nnet_train.dat');
X = D(:,1:2);
y = D(:,3);
d = 2;
N = size(y, 1);

TD = dlmread('hw4_nnet_test.dat');
TX = TD(:,1:2);
Ty = TD(:,3);
TN = size(Ty, 1);

M = [1 6 11 16 21];
T = 50000;
eta = 10;
r = 0.1;


m=3;
Eout_avg = 0;
for iteration = 1:500 
    %randomly initialize all weights w_ij
    w1 = 2*r * rand(d+1,m) - r; %w1 = [w11, w12, ..., w1M], -0.1 - 0.1
    w2 = 2*r * rand(m+1,1) - r;
    delta2 = 0;
    delta1 = zeros(m, 1);
    for t = 1:T
        n = randi(N);
        %forward propagation
        x0 = [1; X(n, :)']; % (d+1)x1
        s1 = w1' * x0;  %s1 = [s11; s12; ...;s1m], mx1
        x1 = [1; tanh(s1)]; % (m+1)x1
        s2 = w2' * x1; %scalar 
        x2 = tanh(s2); %scalar
        
        %backward propagation
        delta2 = -2 * (y(n) - x2) * d_tanh(s2);
        delta1 = delta2 * w2(2:(m+1), :) .* d_tanh(s1); %ignore w2(0)
        %gradient descent
        w1 = w1 - eta * x0 * delta1';
        w2 = w2 - eta * x1 * delta2;
    end
%{    
    Ein = 0;
    for i = 1:N
       x0 = [1; X(i, :)'];
       s1 = w1' * x0;
       x1 = [1; tanh(s1)];
       s2 = w2' * x1;
       x2 = tanh(s2);
       Ein = Ein + (x2 - y(i))^2;
    end
    display(Ein);
%}    

    
    Eout = 0;
    for i = 1:TN
       x0 = [1; TX(i, :)'];
       s1 = w1' * x0;
       x1 = [1; tanh(s1)];
       s2 = w2' * x1;
       x2 = tanh(s2);
       Eout = Eout + (x2 - Ty(i))^2;
    end
    %display(Eout);
    Eout_avg = Eout_avg + Eout;
    display(iteration);
end
Eout_avg = Eout_avg / 500;
display(Eout_avg);

%exp 1
%m=1, Eout = 187.6643
%m=6, Eout = 38.0693
%m=11, Eout = 63.3993
%m=16, Eout = 76.1524
%m=21, Eout = 95.9530

%exp 2
%r=0, 276.3090
%r=0.001, 31.9490 
%r=0.1, 32.1847 
%r=10, 319.0877 
%r=1000, 498.1088

%exp 3
%eta=0.001,  119.6907
%eta=0.01, 27.1590
%eta=0.1, 32.1847 
%eta=1,  476.6305
%eta=10, 495.8560

