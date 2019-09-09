function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%forward propagation 
% Add ones to the X data matrix
X = [ones(m, 1) X];

a1= X;
z2 = Theta1 * X'; % (25 * 401) * ( 401 * 5000) 
a2 = sigmoid(z2); 
a2 = [ones(1,m); a2];
z3 = Theta2 * a2; % (10 * 26) * ( 26 * 5000) ..
a3 = sigmoid(z3);
%[mx, p] = max(a3, [], 2);

%finding the y_k as a vector so that we could reference the 
y_k = zeros(num_labels,m);
for i = 1:m
  index = y(i);
  a = zeros(num_labels,1);
  a(index) = 1;
  y_k(:,i) = a;
endfor
%values required by index
sumF = 0;
% using for loop to iterate over all vector columns and am adding the sum 
for i = 1:m 
  termVector = log(a3(:,i));
  firstVector = (y_k(:,i))' * termVector;
  secondVector = (1- y_k(:,i))' * (log(1- a3(:, i)));
  sumF = sumF + firstVector + secondVector;
endfor
J = (-1.0/m) * sumF;
%J = (1/m) * sum ( sum ( (-y_k) .* log(a3) - (1-y_k) .* log(1-a3) )); 

% now with regularization 
RTheta1 = Theta1(:, 2:end);
RTheta2 = Theta2(:, 2: end);
reg = sum(sum(RTheta1.^2)) + sum(sum(RTheta2.^2));
J = J + ((lambda* reg)/ (2* m));

  

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
% Finding the error on the last layer 
error = zeros(num_labels, m);
for i = 1: m
  %Step 1
  a11 = X(i,:)'; % (1*401) as X already has a bias unit 
  z21 = Theta1 * a11; % (25*401)*(401*1)
  a21 = sigmoid(z21);
  a21 = [1; a21];
  z31 = Theta2 * a21; %(10*26)*(26*1)
  a31 = sigmoid(z31);

  % Step 2 
  d_3 = a31 - y_k(:,i);
  
  % Step 3 
  newz21 = [1; z21];
  d_2 = ((Theta2)' * d_3 ) .* (sigmoidGradient(newz21));
  
  % Step 4
  d_2 = d_2(2:end); % skips sigma2(0) (25*1)
  Theta2_grad = Theta2_grad + d_3 * a21'; % (10*1)*(1*26)
	Theta1_grad = Theta1_grad + d_2 * a11'; % (25*1)*(1*401)
endfor

% Step 5
Theta2_grad = (1/m) * Theta2_grad; % (10*26)
Theta1_grad = (1/m) * Theta1_grad; % (25*401)

% backpropagating to find error on previous layers

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
Theta1_grad(:, 2: end ) = Theta1_grad(:, 2: end ) + (lambda/ m) * Theta1(:, 2: end );
Theta2_grad(:, 2: end ) = Theta2_grad(:, 2: end ) + (lambda/ m) * Theta2(:, 2: end );





















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
