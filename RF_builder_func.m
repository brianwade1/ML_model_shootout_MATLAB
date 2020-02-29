function [stats_train_RF, stats_val_RF, y_train_RF, y_val_RF,...
    forest_best, num_trees_best, max_splits_best, min_leaf_size_best]=...
    RF_builder_func(x_train, x_val, t_train, t_val, want_display)

global want_parellel
min_leaf_size = [1, 2, 3, 5, 10, 15];

max_levels_inTree = floor(log2(size(x_train,1)-1));
max_splits = 2.^(0:max_levels_inTree);

if length(max_splits)>10
    max_splits = floor(linspace(2,size(x_train,1)-1,10)); 
end

num_trees = 500;

surrogate = 'on';


%Prepare for loop
oobError_set = cell(length(num_trees),length(min_leaf_size));
RMSE_set = zeros(length(num_trees),length(min_leaf_size));
forest_set = cell(length(num_trees),length(min_leaf_size));

for i = 1:length(max_splits)
    for j = 1:length(min_leaf_size)
        
        if want_display == true
            disp('********************************')
            disp(['Starting to train RF with ',num2str(max_splits(i)),...
                ' max splits and ',num2str(min_leaf_size(j)),...
                ' min leaf size'])
        end
        
        if want_parellel == true
            forest = TreeBagger(num_trees,x_train,t_train,...
                'MinLeafSize',min_leaf_size(j),...
                'MaxNumSplits',max_splits(i),...
                'Method','regression','Surrogate',surrogate,...
                'OOBPrediction','On','UseParallel','true');
        else
            forest = TreeBagger(num_trees,x_train,t_train,...
                'MinLeafSize',min_leaf_size(j),...
                'MaxNumSplits',max_splits(i),...
                'Method','regression','Surrogate',surrogate,...
                'OOBPrediction','On');
        end
        
        oobError_set{i,j}= oobError(forest,'Mode','cumulative');
        forest_set{i,j} = forest;
        
        %y_train = predict(forest,x_train);
        %y_val = predict(forest,x_val);
        %[~, stats_val_RF] = get_model_results(y_train, y_val,...
        %    t_train, t_val, false, false);
        
        %RMSE_set(i,j) = stats_val_RF(2)^2;

    end
end


MSE_set = reshape(cell2mat(oobError_set),...
    [num_trees numel(max_splits) numel(min_leaf_size)]);

[MSE_best, minErrIdxLin] = min(MSE_set(:));
[idx_Num_trees, idx_max_splits, idx_min_leaf_size] = ...
    ind2sub(size(MSE_set),minErrIdxLin);

num_trees_best = idx_Num_trees;
max_splits_best = max_splits(idx_max_splits);
min_leaf_size_best = min_leaf_size(idx_min_leaf_size);


forest_best = TreeBagger(num_trees_best,x_train,t_train,...
                'MinLeafSize',min_leaf_size_best,...
                'MaxNumSplits',max_splits_best,...
                'Method','regression','Surrogate',surrogate,...
                'OOBPrediction','On');


y_train_RF = predict(forest_best,x_train);
y_val_RF = predict(forest_best,x_val);

if want_display == true
    disp('********************************')
    disp('***         RF Results       ***')
    disp('********************************')
end

[stats_train_RF, stats_val_RF] = get_model_results(y_train_RF, y_val_RF,...
                t_train, t_val, want_display, false);

end