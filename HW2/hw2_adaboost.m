clc
clear

D = dlmread('hw2_adaboost_train.dat');
T = dlmread('hw2_adaboost_test.dat');
%D = dlmread('hw2_temp.txt');
X = D(:,1:2);
y = D(:,3);
TX = T(:,1:2);
Ty = T(:,3);
NT = 1000;

N = 100; %traing set size
u = 1/N * ones(N, 1);

T = 1; %iteration num
s_opt     = zeros(T,1);
theta_opt = zeros(T,1);
i_opt 	= zeros(T,1);
h_opt = zeros(N,T);
alpha = zeros(T,1);
diamond = zeros(T,1);
min_epsilon = 10000;
for t = 1:T
   %Decision stump 
   %sort x1
   %i_opt = 0;
   %s_opt = 0;
   %theta_opt = 0.0;
   Ein_u_opt = 1000;
   for i = 1 : 2
       xi = sort(D(:,i));
       %theta as all the midpoints
       for n = 1:N-1
           for s = [-1 1]
               theta = (xi(n) + xi(n+1)) / 2;
               h = s * sign(X(:,i) - theta);
               Ein_u = u' * (y ~= h);
               if Ein_u < Ein_u_opt
                  Ein_u_opt = Ein_u;
                  i_opt(t) = i;
                  s_opt(t) = s;
                  theta_opt(t) = theta;
               end
           end
       end
       %theta as inf
       theta = -Inf;
       for s = [-1 1]
           h = s * sign(X(:,i) - theta);
           Ein_u = u' * (y ~= h);
           if Ein_u < Ein_u_opt
              Ein_u_opt = Ein_u;
              i_opt(t) = i;
              s_opt(t) = s;
              theta_opt(t) = theta;
           end
       end
   end
   
   %update u
   epsilon = Ein_u_opt / sum(u);
   if epsilon < min_epsilon
       min_epsilon = epsilon;
   end
   diamond(t) = sqrt( (1-epsilon)/epsilon );
   h_opt(:,t) = s_opt(t) * sign(X(:,i_opt(t)) - theta_opt(t));
   u(y ~= h_opt(:,t)) = u(y ~= h_opt(:,t)) * diamond(t);
   u(y == h_opt(:,t)) = u(y == h_opt(:,t)) / diamond(t);
   alpha(t) = log(diamond(t));
end
sum(u)
G = sign( h_opt * alpha );
Ein_G = ones(N,1)' * (y ~= G) / N;

g_sum = zeros(NT,1);
for t = 1:T
    g_t = s_opt(t) * sign(TX(:,i_opt(t)) - theta_opt(t));
    g_sum = g_sum + g_t * alpha(t);
end
G_out = sign(g_sum);
Eout_G = ones(NT,1)' * (Ty ~= G_out) / NT;
