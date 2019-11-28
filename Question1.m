%Question 1

%Clearing memory
clear
clc
close all

%Setting seed for reproducibility
rng(1881)

%Generating data
%Generate samples from a 4-component GMM
%Generating all samples at once (test,validation, etc.)
N1 = 100; N2 = 1000; N3 = 10000; N4 = 20000;
N = [N1 N2 N3];
%Ratios of mixtures
p = [0.17,0.22,0.28,0.33];
%Matrix of means
m = [7 3 16 14;0 6 21 8;-6 4 2 8];
%Setting variance-covariance matrices
Sigma(:,:,1) = [25 1 4;1 20 6;18 13 7];
Sigma(:,:,2) = [27 4 0;4 5 7;3 1 20];
Sigma(:,:,3) = [15 9 6;9 15 8;3 4 6];
Sigma(:,:,4) = [4 1 2;1 22 3;5 8 13];
%Labeling data
[x1, label1] = randGMM(N1,p,m,Sigma);
[x2, label2] = randGMM(N2,p,m,Sigma);
[x3, label3] = randGMM(N3,p,m,Sigma);
[x4, label4] = randGMM(N4,p,m,Sigma);
%Generating a matrix of labels for the classification problem
label1n = zeros(4,N1);
label2n = zeros(4,N2);
label3n = zeros(4,N3);
label4n = zeros(4,N4);
for i = 1:N1
    if label1(i) == 1
        label1n(1,i) = 1;
    elseif label1(i) == 2
        label1n(2,i) = 1;
    elseif label1(i) == 3
        label1n(3,i) = 1;
    else
        label1n(4,i) = 1;
    end
end
for i = 1:N2
    if label2(i) == 1
        label2n(1,i) = 1;
    elseif label2(i) == 2
        label2n(2,i) = 1;
    elseif label2(i) == 3
        label2n(3,i) = 1;
    else
        label2n(4,i) = 1;
    end
end
for i = 1:N3
    if label3(i) == 1
        label3n(1,i) = 1;
    elseif label3(i) == 2
        label3n(2,i) = 1;
    elseif label3(i) == 3
        label3n(3,i) = 1;
    else
        label3n(4,i) = 1;
    end
end
for i = 1:N4
    if label4(i) == 1
        label4n(1,i) = 1;
    elseif label4(i) == 2
        label4n(2,i) = 1;
    elseif label4(i) == 3
        label4n(3,i) = 1;
    else
        label4n(4,i) = 1;
    end
end

%Plotting the data with N=1000 with actual labels
figure(1)
scatter3(x2(1,label2==1),x2(2,label2==1),x2(3,label2==1),'ob'); hold on
scatter3(x2(1,label2==2),x2(2,label2==2),x2(3,label2==2),'or')
scatter3(x2(1,label2==3),x2(2,label2==3),x2(3,label2==3),'ok')
scatter3(x2(1,label2==4),x2(2,label2==4),x2(3,label2==4),'og')
title("Data with True Labels")
xlabel("x1") 
ylabel("x2")
zlabel("x3")
legend("Label 1","Label 2", "Label 3", "Label 4")

%MAP Classifier
%Loss Function
lambda = [0 1 1 1;1 0 1 1;1 1 0 1;1 1 1 0]; 
%Threshold/decision determination
g1 = lambda(1,1)*evalGaussian(x4,m(:,1),Sigma(:,:,1))*p(1) + lambda(1,2)*evalGaussian(x4,m(:,2),Sigma(:,:,2))*p(2) + lambda(1,3)*evalGaussian(x4,m(:,3),Sigma(:,:,3))*p(3) + lambda(1,4)*evalGaussian(x4,m(:,4),Sigma(:,:,4))*p(4);
g2 = lambda(2,1)*evalGaussian(x4,m(:,1),Sigma(:,:,1))*p(1) + lambda(2,2)*evalGaussian(x4,m(:,2),Sigma(:,:,2))*p(2) + lambda(2,3)*evalGaussian(x4,m(:,3),Sigma(:,:,3))*p(3) + lambda(2,4)*evalGaussian(x4,m(:,4),Sigma(:,:,4))*p(4);
g3 = lambda(3,1)*evalGaussian(x4,m(:,1),Sigma(:,:,1))*p(1) + lambda(3,2)*evalGaussian(x4,m(:,2),Sigma(:,:,2))*p(2) + lambda(3,3)*evalGaussian(x4,m(:,3),Sigma(:,:,3))*p(3) + lambda(3,4)*evalGaussian(x4,m(:,4),Sigma(:,:,4))*p(4);
g4 = lambda(4,1)*evalGaussian(x4,m(:,1),Sigma(:,:,1))*p(1) + lambda(4,2)*evalGaussian(x4,m(:,2),Sigma(:,:,2))*p(2) + lambda(4,3)*evalGaussian(x4,m(:,3),Sigma(:,:,3))*p(3) + lambda(4,4)*evalGaussian(x4,m(:,4),Sigma(:,:,4))*p(4);
decision = zeros(1,N4); 
%Deciding on labels based on loss functions
for i = 1:N4
    if g1(i)<g2(i) && g1(i)<g3(i) && g1(i)<g4(i)
        decision(i) = 1;
    elseif g2(i)<g1(i) && g2(i)<g3(i) && g2(i)<g4(i)
        decision(i) = 2;
    elseif g3(i)<g1(i) && g3(i)<g2(i) && g3(i)<g4(i)
        decision(i) = 3;
    elseif g4(i)<g1(i) && g4(i)<g2(i) && g4(i)<g3(i)
        decision(i) = 4;
    end
end

%Plotting the decisions versus the actual labels
figure(2)
scatter3(x4(1,decision==1&label4==1),x4(2,decision==1&label4==1),x4(3,decision==1&label4==1),'ob'); hold on
scatter3(x4(1,decision==2&label4==1),x4(2,decision==2&label4==1),x4(3,decision==2&label4==1),'pr');
scatter3(x4(1,decision==3&label4==1),x4(2,decision==3&label4==1),x4(3,decision==3&label4==1),'pk');
scatter3(x4(1,decision==4&label4==1),x4(2,decision==4&label4==1),x4(3,decision==4&label4==1),'pg');
scatter3(x4(1,decision==1&label4==2),x4(2,decision==1&label4==2),x4(3,decision==1&label4==2),'pb');
scatter3(x4(1,decision==2&label4==2),x4(2,decision==2&label4==2),x4(3,decision==2&label4==2),'or');
scatter3(x4(1,decision==3&label4==2),x4(2,decision==3&label4==2),x4(3,decision==3&label4==2),'pk');
scatter3(x4(1,decision==4&label4==2),x4(2,decision==4&label4==2),x4(3,decision==4&label4==2),'pg');
scatter3(x4(1,decision==1&label4==3),x4(2,decision==1&label4==3),x4(3,decision==1&label4==3),'pb');
scatter3(x4(1,decision==2&label4==3),x4(2,decision==2&label4==3),x4(3,decision==2&label4==3),'pr');
scatter3(x4(1,decision==3&label4==3),x4(2,decision==3&label4==3),x4(3,decision==3&label4==3),'ok');
scatter3(x4(1,decision==4&label4==3),x4(2,decision==4&label4==3),x4(3,decision==4&label4==3),'pg');
scatter3(x4(1,decision==1&label4==4),x4(2,decision==1&label4==4),x4(3,decision==1&label4==4),'pb');
scatter3(x4(1,decision==2&label4==4),x4(2,decision==2&label4==4),x4(3,decision==2&label4==4),'pr');
scatter3(x4(1,decision==3&label4==4),x4(2,decision==3&label4==4),x4(3,decision==3&label4==4),'pk');
scatter3(x4(1,decision==4&label4==4),x4(2,decision==4&label4==4),x4(3,decision==4&label4==4),'og');
title("Data with MAP Decisions")
xlabel("x1") 
ylabel("x2")
zlabel("x3")

%Counting all errors
error = 0;
error1 = 0;
error2 = 0;
error3 = 0;
error4 = 0;

for d = 1:N4
    if decision(d) ~= label4(d)
        if label4(d) == 1
            error1 = error1 + 1;
        elseif label4(d) == 2
            error2 = error2 + 1;
        elseif label4(d) == 3
            error3 = error3 + 1;
        else
            error4 = error4 + 1;
        end
        error = error+1;
    end
end

%Neural Network
%Defining model specifications
%Maximum number of perceptrons
perceptrons = 15; 
%Number of folds
fold = 10;
%Type of neural network
MLP = patternnet(1);
%Using the whole sample to do the learning (no test or validation)
MLP.divideMode = 'none';
MLP.divideFcn = 'dividetrain';
%Defining the loss function (minimizing the number of errors)
MLP.performFcn = 'crossentropy';
%Using a sigmoid transfer function
MLP.layers{1}.transferFcn = 'logsig';
MLP.layers{2}.transferFcn = 'softmax';
%Performance measures
bestPerf = zeros(1,length(N));
perf = zeros(fold,1);
perfMean = zeros(perceptrons,length(N));
%Using k-fold for all sample sizes to determine the best number of
%perceptrons for each case
for j = 1:length(N)
    if j == 1
        x = x1;
        label = label1n;
    elseif j == 2
        x = x2;
        label = label2n;
    else
        x = x3;
        label = label3n;
    end
    %Partitioning data
    dummy = ceil(linspace(0,N(j),fold+1));
    for k = 1:fold
        indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
    end
    %Testing for every perceptron number
    for i = 1:perceptrons
            %K-Fold Cross-Validation
            for k = 1:fold
                indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
                %Using fold k as validation set
                x1Validate = x(:,indValidate); 
                x2Validate = label(:,indValidate);
                if k == 1
                    indTrain = [indPartitionLimits(k,2)+1:N(j)];
                elseif k == fold
                    indTrain = [1:indPartitionLimits(k,1)-1];
                else
                    indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,2):N(j)];
                end

                %Using all other folds as training set
                x1Train = x(:,indTrain); 
                x2Train = label(:,indTrain);
                Ntrain1 = length(indTrain); Nvalidate = length(indValidate);
                %Running the 1 layer neural network
                MLP.layers{1}.size = i;
                %Find the best model
                MLP = configure(MLP,x1Train,x2Train);
                MLP.trainParam.showWindow = true;
                MLP = train(MLP,x1Train,x2Train);
                %Find model predictions
                yhat = MLP(x1Validate);
                [v4dat, v4label] = max(yhat);
                [v5dat, v5label] = max(x2Validate);
                %Test the predictions against real labels for every fold
                perf(k) = 0;
                for h = 1:length(x1Validate)
                    if v4label(h) ~= v5label(h)
                        perf(k) = perf(k) + 1;
                    else
                        perf(k) = perf(k);
                    end
                end
                perf(k) = perf(k)/length(x1Validate);
            end
            %Get the mean performance over all folds
            perfMean(i,j) = mean(perf);
            %Store the best model
            if i == 1
                bestMLP = MLP;
                bestPerf = perfMean(1,j);
            else
                if perfMean(i,j) < bestPerf
                    bestMLP = MLP;
                    bestPerf = perfMean(i,j);
                end
            end       

    end
    %Train the whole dataset with best model for each sample size
    if j == 1        
        bestMLP1 = configure(bestMLP,x1,label);
        bestMLP1.trainParam.showWindow = true;
        bestMLP1 = train(bestMLP,x1,label);
        yhattest1 = bestMLP1(x4);
        perftest1 = perform(bestMLP1,label4n,yhattest1);
    elseif j == 2
        bestMLP2 = configure(bestMLP,x2,label);
        bestMLP2.trainParam.showWindow = true;
        bestMLP2 = train(bestMLP,x2,label);
        yhattest2 = bestMLP2(x4);
        perftest2 = perform(bestMLP2,label4n,yhattest2);
    else
        bestMLP3 = configure(bestMLP,x3,label);
        bestMLP3.trainParam.showWindow = true;
        bestMLP3 = train(bestMLP,x3,label);
        yhattest3 = bestMLP3(x4);
        perftest3 = perform(bestMLP3,label4n,yhattest3);
    end
end

%Average performance for N=100 
figure(3)
stem(perfMean(:,1))
title("Average Performance of the MLP Estimator Across Folds")
xlabel("Number of Perceptrons") 
ylabel("Performance Metric (Average Probability of Error)")
%Average performance for N=1000
figure(4)
stem(perfMean(:,2))
title("Average Performance of the MLP Estimator Across Folds")
xlabel("Number of Perceptrons") 
ylabel("Performance Metric (Average Probability of Error)")
%Average performance for N=10000
figure(5)
stem(perfMean(:,3))
title("Average Performance of the MLP Estimator Across Folds")
xlabel("Number of Perceptrons") 
ylabel("Performance Metric (Average Probability of Error)")

%Calculating the errors for the best model for every sample size on the
%test (N=20000) dataset
[vdat1, vlabel1] = max(yhattest1);
[vdat2, vlabel2] = max(yhattest2);
[vdat3, vlabel3] = max(yhattest3);

v1error = 0;
v1error1 = 0;
v1error2 = 0;
v1error3 = 0;
v1error4 = 0;

for d = 1:N4
    if vlabel1(d) ~= label4(d)
        if label4(d) == 1
            v1error1 = v1error1 + 1;
        elseif label4(d) == 2
            v1error2 = v1error2 + 1;
        elseif label4(d) == 3
            v1error3 = v1error3 + 1;
        else
            v1error4 = v1error4 + 1;
        end
        v1error = v1error+1;
    end
end

v2error = 0;
v2error1 = 0;
v2error2 = 0;
v2error3 = 0;
v2error4 = 0;

for d = 1:N4
    if vlabel2(d) ~= label4(d)
        if label4(d) == 1
            v2error1 = v2error1 + 1;
        elseif label4(d) == 2
            v2error2 = v2error2 + 1;
        elseif label4(d) == 3
            v2error3 = v2error3 + 1;
        else
            v2error4 = v2error4 + 1;
        end
        v2error = v2error+1;
    end
end

v3error = 0;
v3error1 = 0;
v3error2 = 0;
v3error3 = 0;
v3error4 = 0;
for d = 1:N4
    if vlabel3(d) ~= label4(d)
        if label4(d) == 1
            v3error1 = v3error1 + 1;
        elseif label4(d) == 2
            v3error2 = v3error2 + 1;
        elseif label4(d) == 3
            v3error3 = v3error3 + 1;
        else
            v3error4 = v3error4 + 1;
        end
        v3error = v3error+1;
    end
end

%Performance of the best model from N=100 on the test set
figure(6)
scatter3(x4(1,vlabel1==1&label4==1),x4(2,vlabel1==1&label4==1),x4(3,vlabel1==1&label4==1),'ob'); hold on
scatter3(x4(1,vlabel1==2&label4==1),x4(2,vlabel1==2&label4==1),x4(3,vlabel1==2&label4==1),'pr');
scatter3(x4(1,vlabel1==3&label4==1),x4(2,vlabel1==3&label4==1),x4(3,vlabel1==3&label4==1),'pk');
scatter3(x4(1,vlabel1==4&label4==1),x4(2,vlabel1==4&label4==1),x4(3,vlabel1==4&label4==1),'pg');
scatter3(x4(1,vlabel1==1&label4==2),x4(2,vlabel1==1&label4==2),x4(3,vlabel1==1&label4==2),'pb');
scatter3(x4(1,vlabel1==2&label4==2),x4(2,vlabel1==2&label4==2),x4(3,vlabel1==2&label4==2),'or');
scatter3(x4(1,vlabel1==3&label4==2),x4(2,vlabel1==3&label4==2),x4(3,vlabel1==3&label4==2),'pk');
scatter3(x4(1,vlabel1==4&label4==2),x4(2,vlabel1==4&label4==2),x4(3,vlabel1==4&label4==2),'pg');
scatter3(x4(1,vlabel1==1&label4==3),x4(2,vlabel1==1&label4==3),x4(3,vlabel1==1&label4==3),'pb');
scatter3(x4(1,vlabel1==2&label4==3),x4(2,vlabel1==2&label4==3),x4(3,vlabel1==2&label4==3),'pr');
scatter3(x4(1,vlabel1==3&label4==3),x4(2,vlabel1==3&label4==3),x4(3,vlabel1==3&label4==3),'ok');
scatter3(x4(1,vlabel1==4&label4==3),x4(2,vlabel1==4&label4==3),x4(3,vlabel1==4&label4==3),'pg');
scatter3(x4(1,vlabel1==1&label4==4),x4(2,vlabel1==1&label4==4),x4(3,vlabel1==1&label4==4),'pb');
scatter3(x4(1,vlabel1==2&label4==4),x4(2,vlabel1==2&label4==4),x4(3,vlabel1==2&label4==4),'pr');
scatter3(x4(1,vlabel1==3&label4==4),x4(2,vlabel1==3&label4==4),x4(3,vlabel1==3&label4==4),'pk');
scatter3(x4(1,vlabel1==4&label4==4),x4(2,vlabel1==4&label4==4),x4(3,vlabel1==4&label4==4),'og');
title("Data with N=100 MLP Decisions")
xlabel("x1") 
ylabel("x2")
zlabel("x3")

%Performance of the best model from N=1000 on the test set
figure(7)
scatter3(x4(1,vlabel2==1&label4==1),x4(2,vlabel2==1&label4==1),x4(3,vlabel2==1&label4==1),'ob'); hold on
scatter3(x4(1,vlabel2==2&label4==1),x4(2,vlabel2==2&label4==1),x4(3,vlabel2==2&label4==1),'pr');
scatter3(x4(1,vlabel2==3&label4==1),x4(2,vlabel2==3&label4==1),x4(3,vlabel2==3&label4==1),'pk');
scatter3(x4(1,vlabel2==4&label4==1),x4(2,vlabel2==4&label4==1),x4(3,vlabel2==4&label4==1),'pg');
scatter3(x4(1,vlabel2==1&label4==2),x4(2,vlabel2==1&label4==2),x4(3,vlabel2==1&label4==2),'pb');
scatter3(x4(1,vlabel2==2&label4==2),x4(2,vlabel2==2&label4==2),x4(3,vlabel2==2&label4==2),'or');
scatter3(x4(1,vlabel2==3&label4==2),x4(2,vlabel2==3&label4==2),x4(3,vlabel2==3&label4==2),'pk');
scatter3(x4(1,vlabel2==4&label4==2),x4(2,vlabel2==4&label4==2),x4(3,vlabel2==4&label4==2),'pg');
scatter3(x4(1,vlabel2==1&label4==3),x4(2,vlabel2==1&label4==3),x4(3,vlabel2==1&label4==3),'pb');
scatter3(x4(1,vlabel2==2&label4==3),x4(2,vlabel2==2&label4==3),x4(3,vlabel2==2&label4==3),'pr');
scatter3(x4(1,vlabel2==3&label4==3),x4(2,vlabel2==3&label4==3),x4(3,vlabel2==3&label4==3),'ok');
scatter3(x4(1,vlabel2==4&label4==3),x4(2,vlabel2==4&label4==3),x4(3,vlabel2==4&label4==3),'pg');
scatter3(x4(1,vlabel2==1&label4==4),x4(2,vlabel2==1&label4==4),x4(3,vlabel2==1&label4==4),'pb');
scatter3(x4(1,vlabel2==2&label4==4),x4(2,vlabel2==2&label4==4),x4(3,vlabel2==2&label4==4),'pr');
scatter3(x4(1,vlabel2==3&label4==4),x4(2,vlabel2==3&label4==4),x4(3,vlabel2==3&label4==4),'pk');
scatter3(x4(1,vlabel2==4&label4==4),x4(2,vlabel2==4&label4==4),x4(3,vlabel2==4&label4==4),'og');
title("Data with N=1000 MLP Decisions")
xlabel("x1") 
ylabel("x2")
zlabel("x3")

%Performance of the best model from N=10000 on the test set
figure(8)
scatter3(x4(1,vlabel3==1&label4==1),x4(2,vlabel3==1&label4==1),x4(3,vlabel3==1&label4==1),'ob'); hold on
scatter3(x4(1,vlabel3==2&label4==1),x4(2,vlabel3==2&label4==1),x4(3,vlabel3==2&label4==1),'pr');
scatter3(x4(1,vlabel3==3&label4==1),x4(2,vlabel3==3&label4==1),x4(3,vlabel3==3&label4==1),'pk');
scatter3(x4(1,vlabel3==4&label4==1),x4(2,vlabel3==4&label4==1),x4(3,vlabel3==4&label4==1),'pg');
scatter3(x4(1,vlabel3==1&label4==2),x4(2,vlabel3==1&label4==2),x4(3,vlabel3==1&label4==2),'pb');
scatter3(x4(1,vlabel3==2&label4==2),x4(2,vlabel3==2&label4==2),x4(3,vlabel3==2&label4==2),'or');
scatter3(x4(1,vlabel3==3&label4==2),x4(2,vlabel3==3&label4==2),x4(3,vlabel3==3&label4==2),'pk');
scatter3(x4(1,vlabel3==4&label4==2),x4(2,vlabel3==4&label4==2),x4(3,vlabel3==4&label4==2),'pg');
scatter3(x4(1,vlabel3==1&label4==3),x4(2,vlabel3==1&label4==3),x4(3,vlabel3==1&label4==3),'pb');
scatter3(x4(1,vlabel3==2&label4==3),x4(2,vlabel3==2&label4==3),x4(3,vlabel3==2&label4==3),'pr');
scatter3(x4(1,vlabel3==3&label4==3),x4(2,vlabel3==3&label4==3),x4(3,vlabel3==3&label4==3),'ok');
scatter3(x4(1,vlabel3==4&label4==3),x4(2,vlabel3==4&label4==3),x4(3,vlabel3==4&label4==3),'pg');
scatter3(x4(1,vlabel3==1&label4==4),x4(2,vlabel3==1&label4==4),x4(3,vlabel3==1&label4==4),'pb');
scatter3(x4(1,vlabel3==2&label4==4),x4(2,vlabel3==2&label4==4),x4(3,vlabel3==2&label4==4),'pr');
scatter3(x4(1,vlabel3==3&label4==4),x4(2,vlabel3==3&label4==4),x4(3,vlabel3==3&label4==4),'pk');
scatter3(x4(1,vlabel3==4&label4==4),x4(2,vlabel3==4&label4==4),x4(3,vlabel3==4&label4==4),'og');
title("Data with N=10000 MLP Decisions")
xlabel("x1") 
ylabel("x2")
zlabel("x3")

%Functions
function [x, labels] = randGMM(N,alpha,mu,Sigma)
%Generates N samples from a given GMM
d = size(mu,1); 
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
    labels(ind) = m;
end
end

function x = randGaussian(N,mu,Sigma)
%Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

function gmm = evalGMM(x,alpha,mu,Sigma)
%Evaluates the GMM on the grid
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) 
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end

function g = evalGaussian(x,mu,Sigma)
%Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end