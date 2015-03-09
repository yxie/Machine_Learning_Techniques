clc
clear

D = dlmread('hw4_knn_train.dat');
X = D(:,1:9);
y = D(:,10); 
N = size(y, 1);

TD = dlmread('hw4_knn_test.dat');
TX = TD(:,1:9);
Ty = TD(:,10); 
TN = size(Ty, 1); 

%k=1 nn
IDX = knnsearch(X, X);
Ein = 0;
for i = 1:N
    if y(i) ~= y(IDX(i,1))
        Ein = Ein + 1;
    end
end
display('k=1');
display(Ein);

TIDX = knnsearch(X, TX);
Eout = 0;
for i = 1:TN
    if Ty(i) ~= y(TIDX(i,1))
        Eout = Eout + 1;
    end
end
display('k=1');
display(Eout);

%k=5 nn
IDX = knnsearch(X, X, 'K', 5);
Ein = 0;
for i = 1:N
    g = sign(sum( y(IDX(i, :)) ));
    if g ~= y(i)
        Ein = Ein + 1;
    end
end
display('k=5');
display(Ein);

TIDX = knnsearch(X, TX, 'K', 5);
Eout = 0;
for i = 1:TN
    g = sign(sum( y(TIDX(i, :)) ));
    if g ~= Ty(i)
        Eout = Eout + 1;
    end
end
display('k=5');
display(Eout);
