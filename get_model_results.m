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

function [stats_train, stats_val] = get_model_results(y_train, y_val, t_train,...
    t_val, want_display, want_plot)
    
    R2_train = 1 - sum((y_train - t_train).^2)/sum((t_train-mean(t_train)).^2);
    R2_val = 1 - sum((y_val - t_val).^2)/sum((t_val-mean(t_val)).^2);

    res_train = y_train - t_train;
    res_val = y_val - t_val;

    RMSE_train = sqrt(mean((y_train - t_train).^2));  % Root Mean Squared Error
    RMSE_val = sqrt(mean((y_val - t_val).^2));  % Root Mean Squared Error

    MAE_train = mean(abs(y_train - t_train));  % Mean Absolute Error
    MAE_val = mean(abs(y_val - t_val));  % Mean Absolute Error

    max_error_train = max(abs(res_train));
    max_error_val = max(abs(res_val));
    
    stats_train = [R2_train, RMSE_train, MAE_train, max_error_train];
    stats_val = [R2_val, RMSE_val, MAE_val, max_error_val];
    
    if want_display == true
        display(['R2 of the training set: ',num2str(R2_train)])
        display(['R2 of the validation set: ',num2str(R2_val)])
        display(['RMSE of the training set: ',num2str(RMSE_train)])
        display(['RMSE of the validation set: ',num2str(RMSE_val)])
        display(['MAE of the training set: ',num2str(MAE_train)])
        display(['MAE of the validation set: ',num2str(MAE_val)])
        display(['Max observed error of the training set: ',num2str(max_error_train)])
        display(['Max observed error of the validation set: ',num2str(max_error_val)])
    end
    
    if want_plot == true
        R2_train = stats_train(1);
        RMSE_train = stats_train(2);
        MAE_train = stats_train(3);
        max_error_train = stats_train(4);

        R2_val = stats_val(1);
        RMSE_val = stats_val(2);
        MAE_val = stats_val(3);
        max_error_val = stats_val(4);

        min_val = min(cellfun(@min,{y_train,y_val,t_train,t_val}));
        max_val = max(cellfun(@max,{y_train,y_val,t_train,t_val}));
        line = (min(min_val):.1:max(max_val));
        hline = zeros(numel(line));

        res_train = y_train - t_train;
        res_val = y_val - t_val;

        figure()
        subplot(2,4,1)
        hold on
        plot(t_train,y_train,'x')
        plot(line,line,'-')
        hold off
        title('Actual v. Predicted - Training Set')
        xlabel('Actual')
        ylabel('Predicted')
        xlim([min_val max_val])
        ylim([min_val max_val])

        subplot(2,4,2)
        hold on
        plot(y_train,res_train,'x')
        plot(line,hline)
        hold off
        title('Residual vs. Predicted - Training Set')
        xlabel('Predicted')
        ylabel('Residual')
        xlim([min_val max_val])
        subplot(2,4,3)
        hist(res_train)
        title('Residual Histogram - Training Set')
        xlabel('Residual')

        subplot(2,4,4)
        title('Fit Stats - Training Set')
        text(0.2,0.8,['R^2 = ',num2str(R2_train)])
        text(0.2,0.6,['RMSE = ',num2str(RMSE_train)])
        text(0.2,0.4,['MAE = ',num2str(MAE_train)])
        text(0.2,0.2,['Max Error = ',num2str(max_error_train)])

        subplot(2,4,5)
        hold on
        plot(t_val,y_val,'x')
        plot(line,line,'-')
        hold off
        title('Actual v. Predicted - Validation Set')
        xlabel('Actual')
        ylabel('Predicted')
        xlim([min_val max_val])
        ylim([min_val max_val])

        subplot(2,4,6)
        hold on
        plot(y_val,res_val,'x')
        plot(line,hline)
        hold off
        title('Residual vs. Predicted - Validation Set')
        xlabel('Predicted')
        ylabel('Residual')
        xlim([min_val max_val])

        subplot(2,4,7)
        hist(res_val)
        title('Residual Histogram - Validation Set')
        xlabel('Residual')

        subplot(2,4,8)
        text(0.2,0.8,['R^2 = ',num2str(R2_val)])
        text(0.2,0.6,['RMSE = ',num2str(RMSE_val)])
        text(0.2,0.4,['MAE = ',num2str(MAE_val)])
        text(0.2,0.2,['Max Error = ',num2str(max_error_val)])
        title('Fit Stats - Validation Set')
    end
end

function plot_results(y_train, y_val, t_train, t_val)

    [stats_train, stats_val] = get_results(y_train, y_val, t_train, t_val, 0);
    
    
end
