clc
clear

D = dlmread('hw4_kmeans_train.dat');
X = D(:,1:9);
N = size(X, 1);

k=10;
[idx, C, sumd] = kmeans(X,k);
display(sumd);
display(sum(sumd));
