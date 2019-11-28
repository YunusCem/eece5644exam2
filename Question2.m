%Question 2

%Clearing data
clear
clc
close all

%Setting a seed to ensure reproducibility
rng(5000)

%Generating the model
plotData = 1;
%Number of dimensions and number of samples
n = 2; Ntrain = 1000; Ntest = 10000; 
%GMM ratios
alpha = [0.33,0.34,0.33];
%Mean vectors
meanVectors = [-18 0 18;-8 0 8];
%Variance-covariance matrix
covEvalues = [3.2^2 0;0 0.6^2];
covEvectors(:,:,1) = [1 -1;1 1]/sqrt(2);
covEvectors(:,:,2) = [1 0;0 1];
covEvectors(:,:,3) = [1 -1;1 1]/sqrt(2);

%Generating the training data
t = rand(1,Ntrain);
ind1 = find(0 <= t & t <= alpha(1));
ind2 = find(alpha(1) < t & t <= alpha(1)+alpha(2));
ind3 = find(alpha(1)+alpha(2) <= t & t <= 1);
Xtrain = zeros(n,Ntrain);
Xtrain(:,ind1) = covEvectors(:,:,1)*covEvalues^(1/2)*randn(n,length(ind1))+meanVectors(:,1);
Xtrain(:,ind2) = covEvectors(:,:,2)*covEvalues^(1/2)*randn(n,length(ind2))+meanVectors(:,2);
Xtrain(:,ind3) = covEvectors(:,:,3)*covEvalues^(1/2)*randn(n,length(ind3))+meanVectors(:,3);

%Generating the test data
t = rand(1,Ntest);
ind1 = find(0 <= t & t <= alpha(1));
ind2 = find(alpha(1) < t & t <= alpha(1)+alpha(2));
ind3 = find(alpha(1)+alpha(2) <= t & t <= 1);
Xtest = zeros(n,Ntrain);
Xtest(:,ind1) = covEvectors(:,:,1)*covEvalues^(1/2)*randn(n,length(ind1))+meanVectors(:,1);
Xtest(:,ind2) = covEvectors(:,:,2)*covEvalues^(1/2)*randn(n,length(ind2))+meanVectors(:,2);
Xtest(:,ind3) = covEvectors(:,:,3)*covEvalues^(1/2)*randn(n,length(ind3))+meanVectors(:,3);

%Plotting the test and training data
if plotData == 1
    figure(1), subplot(1,2,1),
    plot(Xtrain(1,:),Xtrain(2,:),'.')
    title('Training Data'), axis equal,
    subplot(1,2,2),
    plot(Xtest(1,:),Xtest(2,:),'.')
    title('Testing Data'), axis equal,
end

%Choosing the maximum number of perceptrons
perceptrons = 10;
%Choosing the number of folds
fold = 10;
%Defining the model
MLP = feedforwardnet(1);
%No validation within the model, all of it is used for training
MLP.divideMode = 'none';
MLP.divideFcn = 'dividetrain';
%Defining performance metrics
perf = zeros(fold,1);
perfMean = zeros(perceptrons,2);
dummy = ceil(linspace(0,Ntrain,fold+1));
for k = 1:fold
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
end
for i = 1:perceptrons
    for j = 1:2
        %K-Fold Cross-Validation
        for k = 1:fold
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            %Using fold k as validation set
            x1Validate = Xtrain(1,indValidate); 
            x2Validate = Xtrain(2,indValidate);
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:Ntrain];
            elseif k == fold
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,2):Ntrain];
            end
            
            %Using all other folds as training set
            x1Train = Xtrain(1,indTrain); 
            x2Train = Xtrain(2,indTrain);
            Xtrain1 = [x1Train; x2Train];
            xValidate = [x1Validate; x2Validate];
            Ntrain1 = length(indTrain); Nvalidate = length(indValidate);
            %MLP
            if j==2
                MLP.layers{1}.transferFcn = 'softplus';
            else
                MLP.layers{1}.transferFcn = 'logsig';
            end
            %Running the model            
            MLP.layers{1}.size = i;
            MLP = configure(MLP,Xtrain1(1,:),Xtrain1(2,:));
            MLP.trainParam.showWindow = true;
            MLP = train(MLP,Xtrain1(1,:),Xtrain1(2,:));
            %Predicting outcome with the validation data
            yhat = MLP(xValidate(1,:));
            %Evaluating performance and storing it
            perf(k) = perform(MLP,xValidate(2,:),yhat);
        end
        perfMean(i,j) = mean(perf);
        %Finding and storing the best model dynamically
        if i == 1 && j == 1
            bestMLP = MLP;
            bestPerf = perfMean(1,1);
        else
            if perfMean(i,j) < bestPerf
                bestMLP = MLP;
                bestPerf = perfMean(i,j);
            end
        end       
    end
end

%Train the whole dataset with best model
bestMLP = configure(bestMLP,Xtrain(1,:),Xtrain(2,:));
bestMLP.trainParam.showWindow = true;
bestMLP = train(bestMLP,Xtrain(1,:),Xtrain(2,:));

%Try Against Ntest
yhattest = bestMLP(Xtest(1,:));
perftest = perform(bestMLP,Xtest(2,:),yhattest);

%Plots
figure(2)
hold on
stem(1+log(perfMean(:,1)))
stem(1+log(perfMean(:,2)))
title("Average Performance of the MLP Estimator Across Folds")
xlabel("Number of Perceptrons") 
ylabel("Performance Metric (1+log of MSE)")
legend('Sigmoid','Softplus')
hold off

figure(3)
hold on
scatter(Xtest(1,:),Xtest(2,:),'ob');
scatter(Xtest(1,:),yhattest,'xk');
hold off
title("Performance of the Best Model on Test Data")
xlabel("X1") 
ylabel("X2")
legend('True X2 Values','Predicted X2 Values');