clear ; close all; clc
 
% Load Data  (from Andrew Ng Machine Learning online MOOC)
%  The first two columns contains the X values and the third column
%  contains the label (y).
 
data = load('ex2data2.txt');  %data is 118x3
X = data(:, [1, 2]); y = data(:, 3);
 
index0 = find(y == 0);
index1 = find(y == 1);
 
hold off; plot(X(index0,1),X(index0,2),'ro'); hold on
plot(X(index1,1),X(index1,2),'g+'); 
 
% Labels and Legend
xlabel('Microchip Test 1','fontsize',12)
ylabel('Microchip Test 2','fontsize',12)
legend('y = 0', 'y = 1')

% The data points that are not
%  linearly separable. However, you would still like to use logistic 
%  regression to classify the data points. 
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
degree=6;  %degree of polynomial allowed
Xdata = mapFeature(X(:,1), X(:,2),degree);
 
% Initialize fitting parameters
initial_theta = zeros(size(Xdata, 2), 1);
 
% Set regularization parameter lambda to 1
lambda = 1;
 
% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionLogisticRegression_slow(initial_theta, Xdata, y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);  %should be about 0.693
 
[cost, grad] = costFunctionLogisticRegression(initial_theta, Xdata, y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);  %should be about 0.693

% the matlab functions you want to use are crossvalind.m and confusionmat.m_
% Xdata- A vector of feature, nxD, one set of attributes for each dataset sample
% y- A vector of ground truth labels, nx1 (each class has a unique integer value), one label for each dataset sample
% numberOfFolds- the number of folds for k-fold cross validation
numberOfFolds=5;
rng(2000);  %random number generator seed
CVindex = crossvalind('Kfold',y, numberOfFolds);
 
method='LogisticRegression'
 
lambda=1 
    for i = 1:numberOfFolds
        TestIndex = find(CVindex == i);
        TrainIndex = find(CVindex ~= i);
        
        TrainDataCV = Xdata(TrainIndex,:);
        TrainDataGT =y(TrainIndex);
        
        TestDataCV = Xdata(TestIndex,:);
        TestDataGT = y(TestIndex);
        
        %
        %build the model using TrainDataCV and TrainDataGT
        %test the built model using TestDataCV
        %
        switch method
            case 'LogisticRegression'
                % for Logistic Regression, we need to solve for theta
                % Insert code here to solve for theta... same as Q3
                % Initialize fitting parameters
                initial_theta = zeros(size(Xdata, 2), 1);
                 
                % Set regularization parameter lambda to 1 (you should vary this)
                lambda = 100;
                 
                % Set Options
                options = optimset('GradObj', 'on', 'MaxIter', 400);
                 
                % Optimize
                % Specifying function with the @(t) allows fminunc to call our costFunction
                % The t is an input argument, in this case initial_theta
                [theta, J, exit_flag] = ...
                    fminunc(@(t)(costFunctionLogisticRegression(t, Xdata, y, lambda)), initial_theta, options);                
                % Using TestDataCV, compute testing set prediction using
                % the model created
                % for Logistic Regression, the model is theta
                % Insert code here to see how well theta works...
                TestDataPred = sigmoid(TestDataCV * theta) >= 0.5;

            case 'KNN'
                disp('KNN not implemented yet')
            otherwise
                error('Unknown classification method')
        end
        
        predictionLabels(TestIndex,:) =double(TestDataPred);
    end
    
    confusionMatrix = confusionmat(y,predictionLabels);
    accuracy = sum(diag(confusionMatrix))/sum(sum(confusionMatrix));
    
    fprintf(sprintf('%s: Lambda = %d, Accuracy = %6.2f%%%% \n',method, lambda,accuracy*100));
    fprintf('Confusion Matrix:\n');
    [r c] = size(confusionMatrix);
    for i=1:r
        for j=1:r
            fprintf('%6d ',confusionMatrix(i,j));
        end
        fprintf('\n');
    end
