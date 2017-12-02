data=csvread('C:\Users\akshaybahadur21\Desktop\BreastCancer_Neural\cancer_data.csv');
X=data(:,3:32);
y=data(:,2);
fprintf('Program paused. Press enter to continue.\n');
pause;
%No of output categories
input_layer_size  = 30;  
hidden_layer_size = 30;   
num_labels = 2; 
fprintf('Program paused. Press enter to continue.\n');
pause;
m = size(X, 1);

%theta1 should be 18*86
%theta2 should be 10*19

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('Calculated theta1 and theta2. Program paused. Press enter to continue.\n');
pause;
% Unroll parameters
nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('Unrolled Params. Program paused. Press enter to continue.\n');
pause;

lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n'], J);
     
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
         
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
            
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
