function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
dimension = size(z);
a = dimension(1);
b = dimension(2);
c = a * b;
for i = 1 : c
  g(i)= 1.0 /( 1 + e^-z(i));
endfor
% =============================================================

end
