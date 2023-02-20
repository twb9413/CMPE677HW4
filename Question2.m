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
