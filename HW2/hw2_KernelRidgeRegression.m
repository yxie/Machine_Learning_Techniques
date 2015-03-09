clc
clear

D = dlmread('hw2_lssvm_all.dat');
X = D(1:400, 1:10);
y = D(1:400, 11);
TX = D(401:500, 1:10);
Ty = D(401:500, 11);
Ein = zeros(9 ,1);
Eout = zeros(9 ,1);
count = 0;
for gamma = [32, 2, 0.125]
    K = zeros(400,400);
    KT = zeros(400,100);
    for i = 1:400
        for j = 1:400
            K(i,j) = exp( -gamma * (X(i,:)-X(j,:))*(X(i,:)-X(j,:))' );
        end
    end
    for i = 1:400
        for j = 1:100
            KT(i,j) = exp( -gamma * (X(i,:)-TX(j,:))*(X(i,:)-TX(j,:))' );
        end
    end
    
    for lambda = [0.001, 1, 1000]
        beta = (lambda * eye(400) + K)' * y;
        %Ein prediction
        g =  sign(beta' * K);
        count = count + 1;
        Ein(count) = ones(400,1)' * (g' ~= y);
        %Eout %prediction
        g =  sign(beta' * KT);
        Eout(count) = ones(100,1)' * (g' ~= Ty);
    end
end

