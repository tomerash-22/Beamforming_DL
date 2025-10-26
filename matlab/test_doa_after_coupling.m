warning('off', 'all')

doa_improv = zeros(1,1000);
err_output = zeros(1,1000);

trimmed_doa_err_post = dataset_param_coupling.doa_err_post(1:1000) ; 
trimmed_doa_err_pre = dataset_param_coupling.doa_err_pre(1:1000) ; 
err_input = trimmed_doa_err_post-trimmed_doa_err_pre ;
for i =1:1:1000
    R_cur = R_sig_pred(:,:,i);
    try
    err_output(i) =find_doa_err(R_cur, ...
        dataset_param_coupling.doa(i),[]);   
    doa_improv(i) = dataset_param_coupling.doa_err_post (i) - err_output(i);
    catch 
        continue
    end
    
end


%%
'DOA improvemnt mean'

mean (doa_improv)
std(doa_improv)
'over all mean degregesion'

mean(trimmed_doa_err_post - trimmed_doa_err_pre)

%%
ff = find (trimmed_doa_err_post ==7);
mean (doa_improv(ff))
std (doa_improv(ff))
%%
ff = find(dataset_param_coupling.doa_err_post>1) ; 
figure
histogram(err_output(ff),100)
hold on
histogram(dataset_param_coupling.doa_err_post(ff),100)
mean(doa_improv(ff))




'overall (doa_error<5) precentage'
'before'
zz= find(trimmed_doa_err_post<5);
(length(zz)/1000)
'after'
ff= find(err_output<5);
(length(ff)/1000)
'inital'
hh= find(trimmed_doa_err_pre<5);
(length(hh)/1000)


'overall sucsseful (doa_error<10) precentage'
'before'
zz= find(trimmed_doa_err_post<10);
(length(zz)/1000)
'after'
ff= find(err_output<10);
(length(ff)/1000)
'inital'
hh= find(trimmed_doa_err_pre<10);
(length(hh)/1000)

'overall sucsseful (doa_error<20) precentage'
'before'
zz= find(trimmed_doa_err_post<20);
(length(zz)/1000)
'after'
ff= find(err_output<20);
(length(ff)/1000)
'inital'
hh= find(trimmed_doa_err_pre<20);
(length(hh)/1000)




%%
ff = find(dataset_param_coupling.doa_err_post==0);
histogram(err_output(ff))

mean(err_output(ff))
std(err_output(ff))

%%
figure
zz=find(err_output<5);
histogram(dataset_param_coupling.doa_err_post(zz),90)



%%
%coupl_coeff = 
%ff = find(dataset_param_coupling.doa_err_post<5);
dataset_param_coupling.snr= dataset_param_coupling.snr(1:1000);
dataset_param_coupling.del_ang = dataset_param_coupling.del_ang(1:1000);
dataset_param_coupling.sir= dataset_param_coupling.sir(1:1000);
binned_average_heatmap(dataset_param_coupling.del_ang,...
    dataset_param_coupling.sir ,abs(err_output-err_input) ,20);

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
      fillRadius = 2;  % Use 1 for 3x3, 2 for 5x5, etc.

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
    caxis([0 15]);       
    colormap(jet(256));

    % Transparency for NaNs (optional: show background for missing bins)
    set(gca, 'Color', [0.8 0.8 0.8]);  
    set(findobj(gca,'Type','image'), 'AlphaData', ~isnan(Z_filled));
% % Define custom colormap with white near zero
% cmap = jet(256);
% n = size(cmap,1);
% 
% % Get color scale range
% caxis([-10 40]);  % adjust to your min/max range
% lims = caxis;
% 
% % Find indices near zero (within ±2 dB, adjust threshold as needed)
% zeroThresh = 2;  
% idx = round((0 - lims(1)) / (lims(2)-lims(1)) * n);
% band = round(zeroThresh / (lims(2)-lims(1)) * n);
% 
% cmap(max(idx-band,1):min(idx+band,n), :) = 1;  % set white
% 
% colormap(cmap);


end

function scatter_heatmap(heat_var, X_var, Y_var)
    % Only keep points where heat_var ~= 0 or X_var ~= 0
    valid_idx = heat_var ~= 0 | X_var ~= 0;

    % Filter the input variables
    X_valid = X_var(valid_idx);
    Y_valid = Y_var(valid_idx);
    heat_valid = heat_var(valid_idx);

    % Create the scatter plot
    figure;
    scatter(X_valid, Y_valid, 36, heat_valid, 'filled');
    xlabel('X');
    ylabel('Y');
    title('Scatter Heatmap');
    colorbar;
    colormap(jet);
    c = colorbar;
    c.Label.String = 'Heat Variable';
end

function kde_heatmap(x, y, gridSize)
    if nargin < 3
        gridSize = 100; % default grid resolution
    end

    % Remove NaNs
    valid_idx = ~isnan(x) & ~isnan(y);
    x = x(valid_idx);
    y = y(valid_idx);

    % Define grid
    xlin = linspace(min(x), max(x), gridSize);
    ylin = linspace(min(y), max(y), gridSize);
    [X, Y] = meshgrid(xlin, ylin);

    % Perform KDE on grid using mvksdensity (Multivariate KDE)
    gridPoints = [X(:), Y(:)];
    data = [x(:), y(:)];
    density = mvksdensity(data, gridPoints);

    % Reshape density to 2D grid
    Z = reshape(density, size(X));
% Plot heatmap and get handle
h = imagesc(x_centers, y_centers, Z_filled);
axis xy;

% Set colormap and handle NaNs
colormap(jet);
colorbar;

% Create a custom colormap with white at the end
cmap = jet(256);
cmap(end+1, :) = [1 1 1];  % white
colormap(cmap);

% Set NaNs to index beyond last color (to white)
nan_mask = isnan(Z_filled);
Z_plot = Z_filled;
Z_plot(nan_mask) = max(Z_filled(:)) + 1;  % ensure white maps beyond the normal range
set(h, 'CData', Z_plot);
caxis([min(Z_filled(:)), max(Z_filled(:)) + 1]);

%     % Plot as heatmap
%     figure;
%     imagesc(xlin, ylin, Z);  % x-y axes match data
%     axis xy;                 % flip y to match normal coords
%     colormap(jet);
%     colorbar;
%     title('Density Heatmap among the mtrisec with doa eror < 5 deg');
%     xlabel('snr'); ylabel('P mvdr doa error');
end



