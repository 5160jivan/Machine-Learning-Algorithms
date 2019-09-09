function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
predictionM = X * theta;
logisticPrediction = sigmoid(predictionM);
logOflogisticPrediction = log(logisticPrediction);
first = y' * logOflogisticPrediction;
second = (1- y)' * log(1 - logisticPrediction);
regularizationPara = 0;
%use for loop to  sum the square of thetas from 1 to n 
newTheta = theta; 
newTheta(1) = 0;
regularizationPara = newTheta' * newTheta;
J = (-1.0 / m) * (first + second);
J = J + ((lambda/ (2.0 * m))  * regularizationPara);
grad = ((1.0/m) *(X' * (logisticPrediction - y)) )+ (lambda/m) * newTheta;
% =============================================================

end
