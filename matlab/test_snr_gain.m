



% 
% 
% % Load your struct
% %load('IQ_dataset_param.mat');
% 
% target_len = 100000;
% 
% for i = 1:numel(IQ_dataset_param.clean_sig)
%     % Trim clean signal
%     if numel(IQ_dataset_param.clean_sig{i}) > target_len
%         IQ_dataset_param.clean_sig{i} = IQ_dataset_param.clean_sig{i}(1:target_len);
%     end
%     
%     % Trim distorted signal
%     if numel(IQ_dataset_param.distor_sig{i}) > target_len
%         IQ_dataset_param.distor_sig{i} = IQ_dataset_param.distor_sig{i}(1:target_len);
%     end
% end
% 
% % Save trimmed dataset
% save('IQ_dataset_param_trimmed.mat', 'IQ_dataset_param');
% 
% 
% 
% %78106
% Load your struct
% load('dataset_param_coupling.mat');

%Define the valid length
% valid_len = 10000;
% 
% % List of fields to trim
% fields = fieldnames(dataset_param_coupling);
% 
% for i = 1:numel(fields)
%     f = fields{i};
%     val = dataset_param_coupling.(f);
%     
%     % Only trim arrays with length 100000
%     if isnumeric(val) || islogical(val)
%         if size(val, 1) == 1000
%             dataset_param_coupling.(f) = val(1:valid_len, :);
%         elseif size(val, 2) == 1000
%             dataset_param_coupling.(f) = val(:, 1:valid_len);
%         elseif ndims(val) == 3 && size(val, 3) == 1000
%             dataset_param_coupling.(f) = val(:,:,1:valid_len);
%         end
%     end
% end
% 
% % Save trimmed struct
% save('dataset_param_coupling_trimmed.mat', 'dataset_param_coupling');
% 
% 





ULA_0AZ_24_09 = importdata("ula_2_6Ghz_7_11.mat");
dataset = 'coupling' ; 

switch dataset 
    case 'IQ'
        for i=1:10
        R_sig_cur = signal_dataset.R_sig;
        R_distor_cur = signal_dataset.R_impaired;
         sig_est_dir =signal_dataset.doa(i);
            f_st = 0.5*sin( deg2rad(sig_est_dir));
            a_sv = [1 exp(1j*2*pi*f_st) exp(1j*4*pi*f_st) exp(1j*6*pi*f_st) ]';
            if (sig_est_dir>0)
                a_sv =flipud(a_sv);
            end
            asv_H = a_sv';
            w_mpdr_clean = ( R_sig_cur\a_sv) / (asv_H*( R_sig_cur\a_sv));
            w_mpdr_distor = (R_distor_cur\a_sv) / (asv_H*( R_distor_cur\a_sv));

        
        
        end
    case 'coupling'
        
        len_checked = 1000 ; 
        snr_gain_mpdr_pre = zeros(1,len_checked);
        snr_gain_mpdr_post = zeros(1,len_checked);
        cond_pred = zeros(1,len_checked);
        cond_rec = zeros(1,len_checked);
        snr_gain_final = zeros(1,len_checked);
        Z_inv_pred = zeros(size(dataset_param_coupling.Z_mat));
        ULA_0AZ_24_09 = importdata("ula_2_6Ghz_7_11.mat");
        for i=1:len_checked
            R_c_cur = dataset_param_coupling.R_coupled(:,:,i);
            R_sig_cur = dataset_param_coupling.R_sig(:,:,i);
            R_pred_cur  = R_sig_pred(:,:,i);
            cond_pred(i) = cond(R_pred_cur) ; 
            cond_rec(i) = cond(R_c_cur) ; 
            Z_inv_pred(:,:,i) = R_c_cur\R_sig_cur;
            sig_est_dir = dataset_param_coupling.doa(i);
            f_st = 0.5*sin( deg2rad(sig_est_dir));
            a_sv = [1 exp(1j*2*pi*f_st) exp(1j*4*pi*f_st) exp(1j*6*pi*f_st) ]';
            if (sig_est_dir>0)
                a_sv =flipud(a_sv);
            end
            asv_H = a_sv';
            w_mpdr_pred = ( R_pred_cur\a_sv) / (asv_H*( R_pred_cur\a_sv));
            w_mpdr_pre = ( R_c_cur\a_sv) / (asv_H*( R_c_cur\a_sv));
            w_mpdr_post =( R_sig_cur\a_sv) / (asv_H*( R_sig_cur\a_sv));
            inter_att = dataset_param_coupling.sir(i);
            del_ang = dataset_param_coupling.del_ang(i);
            snr_gain_mpdr_pre(i) =  snr_gain_wrapw(w_mpdr_pre,sig_est_dir,ULA_0AZ_24_09,inter_att, ...
            del_ang,false);
            snr_gain_mpdr_post(i) =  snr_gain_wrapw(w_mpdr_post,sig_est_dir,ULA_0AZ_24_09,inter_att, ...
                del_ang,false);
            snr_gain_mpdr_pred(i) =  snr_gain_wrapw(w_mpdr_pred,sig_est_dir,ULA_0AZ_24_09,inter_att, ...
                del_ang,false);
            snr_gain_final_tot_diff(i) = snr_gain_mpdr_post(i) - snr_gain_mpdr_pre(i);
            snr_gain_final_pred(i) = snr_gain_mpdr_pred(i) - snr_gain_mpdr_pre(i);
% 
%             if snr_gain_mpdr_pre(i)+ 2  <  snr_gain_mpdr_post(i) 
%                 snr_gain_wrapw(w_mpdr_pre,sig_est_dir,ULA_0AZ_24_09,inter_att, ...
%             del_ang,true);
%                 snr_gain_wrapw(w_mpdr_post,sig_est_dir,ULA_0AZ_24_09,inter_att, ...
%                 del_ang,true);
%             end
%         
        end
end

%%



dataset_param_coupling.snr= dataset_param_coupling.snr(1:1000);
dataset_param_coupling.del_ang = dataset_param_coupling.del_ang(1:1000);
dataset_param_coupling.sir= dataset_param_coupling.sir(1:1000);





binned_average_heatmap(snr_gain_final_tot_diff, ...
    abs(snr_gain_final_tot_diff-snr_gain_final_pred),30)


function binned_average_heatmap(x, y, heat_val, gridSize)
    if nargin < 4
        gridSize = 100;  % default grid resolution
    end

    % Remove NaNs
    valid_idx = ~isnan(x) & ~isnan(y) & ~isnan(heat_val);
    x = x(valid_idx);
    y = y(valid_idx);
    heat_val = heat_val(valid_idx);

    % Define grid
    x_edges = linspace(min(x), max(x), gridSize+1);
    y_edges = linspace(min(y), max(y), gridSize+1);
    x_centers = (x_edges(1:end-1) + x_edges(2:end)) / 2;
    y_centers = (y_edges(1:end-1) + y_edges(2:end)) / 2;

    % Assign each point to a bin
    [~, x_bin] = histc(x, x_edges);
    [~, y_bin] = histc(y, y_edges);

    % Initialize grid
    Z = nan(gridSize, gridSize);
    count = zeros(gridSize, gridSize);  % count points per bin
    sum_heat = zeros(gridSize, gridSize);  % sum of heat values per bin

    % Accumulate heat values
    for i = 1:length(x)
        xi = x_bin(i);
        yi = y_bin(i);
        if xi >= 1 && xi <= gridSize && yi >= 1 && yi <= gridSize
            sum_heat(yi, xi) = sum_heat(yi, xi) + heat_val(i);
            count(yi, xi) = count(yi, xi) + 1;
        end
    end

    % Compute average heat per bin
    Z = sum_heat ./ count;
    Z(count == 0) = NaN;  % leave empty bins blank
% --- Fill NaNs using 3x3 neighbor averaging ---
      fillRadius = 4;  % Use 1 for 3x3, 2 for 5x5, etc.

    Z_filled = Z;  % copy of Z for filling
    for i = 1:gridSize
        for j = 1:gridSize
           % if isnan(Z(i,j))
                % Get neighborhood bounds based on fillRadius
                i_min = max(i - fillRadius, 1);
                i_max = min(i + fillRadius, gridSize);
                j_min = max(j - fillRadius, 1);
                j_max = min(j + fillRadius, gridSize);

                % Extract neighborhood
                neighbors = Z(i_min:i_max, j_min:j_max);
                valid_vals = neighbors(~isnan(neighbors));

                if ~isempty(valid_vals)
                    Z_filled(i,j) = mean(valid_vals);
                end
            %end
        end
    end

    % --- Plot heatmap ---
    figure;
    imagesc(x_centers, y_centers, Z_filled);
    axis xy; grid on; colorbar;

    % Set colormap range: coldest=0, hottest=10
    caxis([-1 3]);       
    colormap(jet(256));

    % Transparency for NaNs (optional: show background for missing bins)
    set(gca, 'Color', [0.8 0.8 0.8]);  
    set(findobj(gca,'Type','image'), 'AlphaData', ~isnan(Z_filled));





end

function snr_gain_final = snr_gain_wrapw(w,sig_est_dir,ULA_0AZ_24_09,inter_att, ...
    del_ang,plot)

 if ( (sig_est_dir+del_ang < 90) && (sig_est_dir-del_ang > -90))
        snr_gain(1,1) = snr_calc_int(w,sig_est_dir,ULA_0AZ_24_09, ...
            inter_att,sig_est_dir+del_ang,plot);
        snr_gain(1,2) = snr_calc_int(w,sig_est_dir,ULA_0AZ_24_09, ...
            inter_att,sig_est_dir-del_ang,plot);
        snr_gain_final = max(snr_gain);

 else
     if(sig_est_dir+del_ang < 90)
         snr_gain_final = snr_calc_int(w,sig_est_dir,ULA_0AZ_24_09, ...
            inter_att,sig_est_dir+del_ang,plot);
     else
        snr_gain_final = snr_calc_int(w,sig_est_dir,ULA_0AZ_24_09, ...
                    inter_att,sig_est_dir-del_ang,plot);
    end
 end
end

function snr_gain =snr_calc_int(w,sig_est_dir,ULA_0AZ_24_09,inter_att, ...
    inter_ang,plot_flag)

fc = 2.6e9;
arrayGain = phased.ArrayGain("SensorArray" , ULA_0AZ_24_09 , "WeightsInputPort" ...
    , true,"PropagationSpeed" ,physconst('LightSpeed') );

% pat = patten(ULA_0AZ_24_09 ,fc,-90:90,0,'Type','powerdb',...
%       'PropagationSpeed',physconst('LightSpeed'),'Weights',...
%       fliplr(w),'Normalize',false,...
%       'CoordinateSystem','rectangular');
     

        gain = arrayGain(fc,-90:90,w);
        if (plot_flag)
         figure ; 
         ang=-90:90;
         plot(ang,gain) 
         grid on
        end

       
%w = ones(4,1) ; 
        gain_flp = arrayGain(fc,-90:90,flipud(w));
        if (plot_flag)
         figure ; 
          ang=-90:90;
          plot(ang,gain_flp) 
          grid on
        end
        
        sig_gain = gain(91 + sig_est_dir);
        inter_gain = gain(91 +  inter_ang);
        sig_gain_flp = gain_flp(91 + sig_est_dir);
        inter_gain_flp = gain_flp(91 +  inter_ang);
        
        inter_plus_noise_flp = pow2db( ...
            db2pow(inter_gain_flp - inter_att ) + 1);   % Add 1 for normalized white noise power
        
        snr_gain_flp = (sig_gain_flp - inter_plus_noise_flp); 
% 

        inter_plus_noise = pow2db( ...
            db2pow(inter_gain - inter_att ) + 1);   % Add 1 for normalized white noise power
        snr_gain = (sig_gain - inter_plus_noise); 

        snr_gain = max(snr_gain,snr_gain_flp);
% sig_gain = gain(91 + sig_est_dir);
% inter_gain = gain(91 - inter_ang);
% 
% inter_plus_noise = mag2db( ...
%     db2mag(inter_gain - inter_att) + 1);   % Add 1 for normalized white noise power
% 
% snr_gain = (sig_gain - inter_plus_noise); 
        
end



%% do with knowen interfirence angle
