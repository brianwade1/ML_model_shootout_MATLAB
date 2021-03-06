%Copyright (C) 2020  Brian M. Wade

%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.

%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.

%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <https://www.gnu.org/licenses/>.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Grid Search for hyperpameters on multiple %%%
%%%  supervised learning models to include     %%%
%%%  neural networks, random forests, and      %%%
%%%  tree ensemble models                      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Written by: Brian Wade

%% Setup
clear
clc
close all
start_time = tic;
%seed = rng(123);

global want_parellel


%% Inputs

%input_file = 'abalone.csv';
%feature_cols = 1:8;
%ignore_col = 18;
%target_col = 9;

input_file = 'BostonHousePriceDataset.csv';
feature_cols = 1:13;
%ignore_col = 18;
target_col = 14;


want_parellel = false;

val_perc = 0.15;
test_perc = 0.15;

want_all_display =true;
want_final_display = true;
want_plot = true;


%% Get data

table = readtable(input_file, 'PreserveVariableNames', 1);
VarNames = table.Properties.VariableNames';
data_raw = table2array(table);

%%remove ignore cols
%data_raw(:,ignore_col)=[];

%divide data into input and target matricies
x = data_raw(:,feature_cols);
target = data_raw(:,target_col);

%% Prepare the data

%Divide into train and test set
[x_train, x_val, x_test, ind_set] = divide_data(x, val_perc, test_perc);
[t_train, t_val, t_test] = apply_divide_data(target, ind_set);

%% Setup parellel workers

if want_parellel == true
    %open pool of workers
    p=gcp('nocreate');
    if isempty(p)==1
        parpool('local',num_workers);
        p=gcp();
    end
end


%% Train Models
disp('Let the training begin!!')

% Neural Net Training 
disp(' ')
disp('Starting the training for Neural Nets')
[stats_train_NN, stats_val_NN, y_train_NN, y_val_NN, px, py, net_set,...
  lr_NN_best, num_NN, nodes_best] = NN_builder_func(x_train, x_val,...
  t_train, t_val, want_all_display);



%Transpose outputs of NN 
y_train_NN = y_train_NN';
y_val_NN = y_val_NN';

RMSE_NN = stats_val_NN(2);


% Random Forest Training
disp(' ')
disp('Starting the training for Random Forests')

[stats_train_RF, stats_val_RF, y_train_RF, y_val_RF,...
   forest_best, num_trees_best, max_splits_best, min_leaf_size_best] =...
   RF_builder_func(x_train, x_val, t_train, t_val, want_all_display);

RMSE_RF = stats_val_RF(2);

% Ensemble Forest Training
disp(' ')
disp('Starting the training for Ensemble Forest')

[stats_train_EF, stats_val_EF, y_train_EF, y_val_EF,...
    forest_best_EF, num_trees_best_EF, max_splits_best_EF, lr_best_EF] =...
    Ensemble_builder_func(x_train, x_val, t_train,t_val, want_all_display);

RMSE_EF = stats_val_EF(2);


%% Show final results
disp('*********************************')
disp('*********************************')
disp('... and the winner is ...')

RMSE_all = [RMSE_NN, RMSE_RF, RMSE_EF];
[best_RMSE, best_model] = min(RMSE_all);

% train the best model on the training and validation sets and test with
% the test set
x_set = [x_train; x_val];
t_set = [t_train; t_val];

%Show the predictions of the winning model on the full training and test
%sets.
if best_model == 1
    
    disp(['A neural network with ',num2str(nodes_best),...
        ' nodes and a learning rate of ',num2str(lr_NN_best),...
        ' is the best model'])
    
    x_set_scaled = mapminmax('apply', x_set', px);
    x_test_scaled = mapminmax('apply', x_test', px);
    
    %Predict outputs
    y_train_total = 0;
    y_test_total = 0;
    for i = 1:num_NN
        net_i = net_set{i};
        y_train_i=net_i(x_set_scaled);
        y_train_total = y_train_total+y_train_i;
        y_test_i=net_i(x_test_scaled);
        y_test_total = y_test_total+y_test_i;
    end

    y_train_scaled = y_train_total ./ num_NN;
    y_test_scaled = y_test_total ./ num_NN;
    
    %Reverse Scaling
    y_train_best = mapminmax('reverse',y_train_scaled,py);
    y_test_best = mapminmax('reverse',y_test_scaled,py);
    
    %Transpose back to vertical
    y_train_best = y_train_best';
    y_test_best = y_test_best';
    
elseif best_model == 2
    
    disp(['A random forest with ',num2str(num_trees_best),' trees, ',...
        num2str(max_splits_best), ...
        ' max splits and a minimume leaf size of ',...
        num2str(min_leaf_size_best), ' is the best model'])
    
    %use the best RF to predict the results
    y_train_best = predict(forest_best,x_set);
    y_test_best = predict(forest_best,x_test);
    
else
    
    disp(['An ensemble forest with ',num2str(num_trees_best_EF),...
        ' trees, ', num2str(max_splits_best_EF), ...
        ' max splits, and a minimume leaf size of ',...
        num2str(min_leaf_size_best_EF), ' is the best model'])
    
    %use the best RF to predict the results
    y_train_best = predict(forest_best_EF,x_set);
    y_test_best = predict(forest_best_EF,x_test);
    
end

disp('*********************************')

%Display the final stats and plot if desired.
[stats_train, stats_val] = get_model_results(y_train_best, y_test_best,...
    [t_train; t_val], t_test, want_final_display, want_plot);

final_time = toc(start_time);
hours = floor(final_time / 3600);
min = floor((final_time - hours*3600)/60);
sec = final_time - hours*3600 - min*60;

disp('*********************************')
disp('*********************************')

disp(['Total time to complete: ',num2str(hours),' hours, ',num2str(min),...
    ' minutes, and ',num2str(sec),' seconds'])


%% Supporting functions

function [x_train, x_val, x_test, ind_set] = divide_data(x, val_perc, test_perc)
    Q=size(x,1);
    [ind_train,ind_val,ind_test]= dividerand(Q,1-val_perc-test_perc,val_perc,test_perc);

    x_train=x(ind_train,:);
    x_val=x(ind_val,:);
    x_test = x(ind_test,:);

    ind_set = {ind_train,ind_val,ind_test};

end

function [x_train, x_val, x_test] = apply_divide_data(x, ind_set)
    ind_train = ind_set{1};
    ind_val = ind_set{2};
    ind_test = ind_set{3};
    
    x_train=x(ind_train,:);
    x_val=x(ind_val,:);
    x_test = x(ind_test,:);

end

