clc
clear

%addpath('')

D = dlmread('features.train.txt');
%y = double(D(:,1)==0);
y = -1 * ones(size(D,1),1);
for i=1:size(D,1)
   if(D(i,1)==0)
       y(i)=1;
   end
end
x1 = D(:,2);
x2 = D(:,3);
X=double([x1 x2]);

%scatter(x1(find(y==1)), x2(find(y==1)));
%hold on;
%scatter(x1(find(y==-1)), x2(find(y==-1)), [], 'r');

%Q15 linear kernel, C=0.01, 
display('**************  Q15  ************');
model = svmtrain(y, X, '-s 0 -c 0.01 -t 0');

w = model.SVs' * model.sv_coef;
display(norm(w));
display('press to continue');
pause();

%Q16 Quadratic Kernel, Q=2, C=0.01
display('**************  Q16  ************');
num = [0 2 4 6 8];
Ein = zeros(5,1);
sum_a = zeros(5,1);
for j = 1:5
    y = -1 * ones(size(D,1),1);
    for i=1:size(D,1)
       if(D(i,1)==num(j))
           y(i)=1;
       end
    end

    model = svmtrain(y, X, '-s 0 -c 0.01 -t 1 -g 1 -r 1 -d 2');
    w = model.SVs' * model.sv_coef;

    %calculate Ein
    for i = 1:size(D,1)
        if(y(i) ~= sign(X(i,:)*w - model.rho))
            Ein(j) = Ein(j) + 1;
        end
    end
    
    %Q17
    sum_a(j) = sum( abs(model.sv_coef) );
end
display('press to continue');
pause();
%Ein
%0: 1438
%2: 731
%4: 652
%6: 664
%8: 542 lowest


%Q18 Gaussian kernel, gamma=100,  C within {0.001,0.01,0.1,1,10}
%Q19 C=0.1, gamma within {1, 10, 1000, 10000, 100}
display('**************  Q18  ************');
y = -1 * ones(size(D,1),1);
for i=1:size(D,1)
   if(D(i,1)==0)
       y(i)=1;
   end
end
model = svmtrain(y, X, '-s 0 -c 0.1 -t 2 -g 1000');
%objective value        #SV
%0.001: -2.380633       2398
%0.01: -23.144993       2487
%0.1: -179.198592       2280
%1: -1401.258805        1773
%10: -13027.302689      1685
    
%validation
T = dlmread('features.test.txt');
test_label = -1 * ones(size(T,1),1);
test_instance = T(:, 2:3);
for i = 1:size(T,1)
   if(T(i,1)==8)
       test_label(i) = 1;
   end
end

[predicted_lable, accuracy, d] = svmpredict(test_label, test_instance, model);
display('press to continue');
pause();
%accuracy for Q18 with different c
%0.001: 91.7289%
%0.01: 91.7289%
%0.1: 81.7638%
%1: 79.6213%
%10: 79.422%

%accuracy for Q19 with different g, (may be wrong)
%1: 83.2586%
%10: 81.4649%
%100: 81.7638%
%1000: 91.7289%  
%10000: 91.7289%



%Q20 find best gamma among {1,10,100,1000,10000} 
clc
clear

D = dlmread('features.train.txt');

samplenum = size(D,1);
accuracy_avg = 0;
for iteration = 1:100

    r = randsample(samplenum, 1000);
    index = zeros(samplenum, 1);
    for i = 1:1000
        index(r(i)) = 1;
    end

    y = D(find(index==0), 1);
    X = D(find(index==0), 2:3);

    test_label = D(find(index==1), 1);
    test_instance = D(find(index==1), 2:3);

    for i=1:size(y)
       if y(i)==0 
           y(i)=1;
       else
           y(i)=-1;
       end
    end
    
    for i=1:size(test_label)
       if test_label(i)==0 
           test_label(i)=1;
       else
           test_label(i)=-1;
       end
    end

    model = svmtrain(y, X, '-s 0 -c 0.1 -t 2 -g 10000');
    [predicted_lable, accuracy, d] = svmpredict(test_label, test_instance, model);
    accuracy_avg = accuracy_avg + accuracy;
end

accuracy_avg = accuracy_avg / 100;

%gamma:
%1: 89.4290
%10: 90.1010 highest
%100: 89.855
%1000: 84.6
%10000: 83.675








