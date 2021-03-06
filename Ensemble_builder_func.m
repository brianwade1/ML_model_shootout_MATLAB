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


function [stats_train_EF, stats_val_EF, y_train_EF, y_val_EF,...
    forest_best_EF, num_trees_best_EF, max_splits_best_EF,lr_best_EF] =...
    Ensemble_builder_func(x_train, x_val, t_train, t_val, want_display)

% References:
% Part of this code was developed with the help of:
% https://www.mathworks.com/help/stats/fitensemble.html#d118e390831

global want_parellel

max_levels_inTree = floor(log2(size(x_train,1)-1));
max_splits = 2.^(0:max_levels_inTree);

if length(max_splits)>10
    max_splits = floor(linspace(1,max_splits(end),10)); 
end

lr = [0.1 0.25 0.5 1];

num_trees = 500;
KFold = 5;

surrogate = 'on';


%Prepare for loop
forest_set = cell(length(max_splits),length(lr));

for i = 1:length(max_splits)
    for k = 1:length(lr)
        
        if want_display == true
            disp('********************************')
            disp(['Starting to train ensemble with ',...
                num2str(max_splits(i)),...
                ' max splits and a learning rate of ',num2str(lr(k))])
        end
        
        if want_parellel == true
            temp = templateTree('MaxNumSplits',max_splits(i),...
                'Surrogate',surrogate);
            
            forest = fitrensemble(x_train,t_train,...
                'NumLearningCycles',num_trees, 'Learners',temp,...
                'KFold',KFold,'LearnRate',lr(k),'UseParallel','true');
            
        else
            temp = templateTree('MaxNumSplits',max_splits(i),...
                'Surrogate',surrogate);
            
            forest = fitrensemble(x_train,t_train,...
                'NumLearningCycles',num_trees, 'Learners',temp,...
                'KFold',KFold,'LearnRate',lr(k));
        end
        
        forest_set{i,k} = forest;
        
    end
end

kflAll = @(x)kfoldLoss(x,'Mode','cumulative');
MSE_set_2 = cellfun(kflAll,forest_set,'Uniform',false);
MSE_set = reshape(cell2mat(MSE_set_2),...
    [num_trees numel(max_splits) numel(lr)]);

[MSE_best, minErrIdxLin] = min(MSE_set(:));
[idx_Num_trees, idx_max_splits, idx_lr] = ...
    ind2sub(size(MSE_set),minErrIdxLin);


num_trees_best_EF = idx_Num_trees;
max_splits_best_EF = max_splits(idx_max_splits);
lr_best_EF = lr(idx_lr);


temp = templateTree('MaxNumSplits',max_splits_best_EF,...
                    'Surrogate',surrogate);

forest_best_EF = fitrensemble(x_train,t_train,...
                    'NumLearningCycles',num_trees_best_EF, 'Learners',...
                    temp,'LearnRate',lr_best_EF);

y_train_EF = predict(forest_best_EF,x_train);
y_val_EF = predict(forest_best_EF,x_val);

if want_display == true
    disp('********************************')
    disp('***         EF Results       ***')
    disp('********************************')
end

[stats_train_EF, stats_val_EF] = get_model_results(y_train_EF, y_val_EF,...
                t_train, t_val, want_display, false);

end