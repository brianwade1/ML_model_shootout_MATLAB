function [stats_train_best, stats_val_best, y_train_best, y_val_best,...
    px, py, nets_best, lr_best, num_NN, nodes_best] = ...
    NN_builder_func(x_train_raw, x_val_raw, t_train, t_val, want_display)


node_set = {5, 10, 15, 20, [5 3], [10, 5], [15, 5], [20, 10]};
lr_set = [0.001 0.05 0.01 0.05 0.1];

trainfunc='trainbr';  %trainlm %trainbr %trainscg
max_fail=6;
num_NN = 5;

%% Prepare the data

% Transpose and Rescale between 0-1 (MATLAB NN program likes features as
% rows and data as cols)

[x_train, px] = mapminmax(x_train_raw');
x_val = mapminmax('apply', x_val_raw', px);
[t_train_scaled, py] = mapminmax(t_train');
%t_val_scaled = mapminmax('apply', t_val', py);
t_train = t_train';
t_val = t_val';


%% Train Models

RMSE_best = inf;

for i = 1:length(node_set)
    for j = 1:length(lr_set)
        
        nodes = node_set{i};
        lr = lr_set(j);
        
        if want_display == true
            disp('********************************')
            disp(['Starting to train NN with ',num2str(nodes),...
                ' nodes, and a learning rate of ', num2str(lr)])
        end
        
        %train NN with settings
        [y_train_scaled, y_val_scaled, net_set] = NN_set_trainer(nodes,...
            trainfunc, lr, max_fail, num_NN, x_train, x_val,...
            t_train_scaled);
        
        %Reverse Scaling
        y_train = mapminmax('reverse',y_train_scaled,py);
        y_val = mapminmax('reverse',y_val_scaled,py);
        
        %Check goodness of fit
        [stats_train, stats_val] = get_model_results(y_train, y_val, ...
            t_train, t_val, false, false);
        
        RMSE_val = stats_val(2);
        
        if RMSE_val < RMSE_best
            RMSE_best = RMSE_val;
            nodes_best = nodes;
            lr_best = lr;
        end
       
        
    end
end

%train NN with best settings
[y_train_scaled, y_val_scaled, nets_best] = NN_set_trainer(nodes_best,...
    trainfunc, lr_best, max_fail, num_NN, x_train, x_val,...
    t_train_scaled);
        
%Reverse Scaling
y_train_best = mapminmax('reverse',y_train_scaled,py);
y_val_best = mapminmax('reverse',y_val_scaled,py);

if want_display == true
    disp('********************************')
    disp('***         NN Results       ***')
    disp('********************************')
end

%Check goodness of fit
[stats_train_best, stats_val_best] = get_model_results(y_train_best,...
    y_val_best, t_train, t_val, want_display, false);

end

%% NN Trainer Function

function [y_train_scaled, y_val_scaled, net_set] = NN_set_trainer(nodes,...
    trainfunc, lr, max_fail, num_NN, x_train, x_val, t_train)
    
    global want_parellel

    %Initialize net
    net=fitnet(nodes,trainfunc);

    %Set parameters
    net.trainParam.lr = lr;
    net.trainParam.max_fail=max_fail;
    net.trainParam.showCommandLine=0;
    net.trainParam.showWindow=0;
    net.layers{size(nodes,2)+1}.transferFcn ='purelin';

    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 0.85;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0;

    %train net
    net_set=cell(num_NN);
    if want_parellel == true
        parfor i = 1:num_NN
            net_set{i}=train(net,x_train,t_train);
        end
    else
        for i = 1:num_NN
            net_set{i}=train(net,x_train,t_train);
        end    
    end

    %Predict outputs
    y_train_total = 0;
    y_val_total = 0;
    for i = 1:num_NN
        net_i = net_set{i};
        y_train_i=net_i(x_train);
        y_train_total = y_train_total+y_train_i;
        y_val_i=net_i(x_val);
        y_val_total = y_val_total+y_val_i;
    end

    y_train_scaled = y_train_total ./ num_NN;
    y_val_scaled = y_val_total ./ num_NN;
    
end


