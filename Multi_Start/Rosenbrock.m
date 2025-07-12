clc; clear; close all;

%% Settings
nStarts = 10;              % Number of random starting points
Iter_max = 2e3;            % Max iterations for all methods
lr = 0.002;                % Learning rate
tel = 1e-6;                % Termination tolerance
rng(1);                    % For reproducibility
initial_points = 3 * (2 * rand(nStarts, 2) - 1);

% Store results
results = [];

for i = 1:nStarts
    x0 = initial_points(i,:);

    fprintf("⏳ Running optimization from initial point #%d / %d\n", i, nStarts);

    [~, f_gd, t_gd, c_gd] = runGD(x0, lr, tel, Iter_max);
    [~, f_spgd, t_spgd, c_spgd] = runSPGD(x0, lr, tel, Iter_max);
    [~, f_bayes, t_bayes, c_bayes] = runBayesOpt(x0, Iter_max);
    [~, f_sa, t_sa, c_sa] = runSA(x0, Iter_max);
    [~, f_unc, t_unc, c_unc] = runFminunc(x0);

    results = [results; ...
        i, f_gd, t_gd, c_gd, ...
        f_spgd, t_spgd, c_spgd, ...
        f_bayes, t_bayes, c_bayes, ...
        f_sa, t_sa, c_sa, ...
        f_unc, t_unc, c_unc];

    fprintf("✅ Finished point #%d: GD=%.3f, SPGD=%.3f, BO=%.3f, SA=%.3f, Fminunc=%.3f\n\n", ...
        i, f_gd, f_spgd, f_bayes, f_sa, f_unc);
end

% Convert to table
colNames = {'Run', ...
    'GD_Fval', 'GD_Time', 'GD_Count', ...
    'SPGD_Fval', 'SPGD_Time', 'SPGD_Count', ...
    'BayesOpt_Fval', 'BayesOpt_Time', 'BayesOpt_Count', ...
    'SA_Fval', 'SA_Time', 'SA_Count', ...
    'Fminunc_Fval', 'Fminunc_Time', 'Fminunc_Count'};
results_table = array2table(results, 'VariableNames', colNames);

% Show summary
mean_stats = varfun(@mean, results_table(:, 2:end));
mean_stats.Properties.RowNames = {'Mean'};

std_stats = varfun(@std, results_table(:, 2:end));
std_stats.Properties.RowNames = {'StdDev'};

disp(results_table)
disp(mean_stats)
disp(std_stats)

%% 📊 Improvement of SPGD Over Other Methods (fval + CPU time)

% Define column indices for Fval and Time (according to your structure)
fval_idx  = [1, 4, 7, 10, 13];   % GD, SPGD, BayesOpt, SA, Fminunc
time_idx  = [2, 5, 8, 11, 14];
methods   = {'GD', 'SPGD', 'BayesOpt', 'SA', 'Fminunc'};

% Get mean values from summary
mean_fvals = mean_stats{1, fval_idx};
mean_times = mean_stats{1, time_idx};

% Get SPGD values
spgd_fval = mean_fvals(2);  % index 2 corresponds to SPGD_Fval
spgd_time = mean_times(2);  % index 2 corresponds to SPGD_Time

% Initialize
improvement_fval = zeros(1, numel(methods));
improvement_time = zeros(1, numel(methods));

for i = 1:numel(methods)
    if i == 2  % Skip SPGD
        improvement_fval(i) = NaN;
        improvement_time(i) = NaN;
        continue
    end
    
    % Avoid divide-by-zero
    denom_fval = mean_fvals(i);
    denom_time = mean_times(i);
    
    % % fval improvement: positive means SPGD achieved lower value
    if abs(denom_fval) > 1e-8
        improvement_fval(i) = 100 * (mean_fvals(i) - spgd_fval) / abs(denom_fval);
    else
        improvement_fval(i) = NaN;
    end

    % time improvement: positive means SPGD is faster
    if abs(denom_time) > 1e-8
        improvement_time(i) = 100 * (mean_times(i) - spgd_time) / denom_time;
    else
        improvement_time(i) = NaN;
    end
end

% Build table
spgd_improvement = table(methods', ...
    improvement_fval', improvement_time', ...
    'VariableNames', {'Method', ...
                      'SPGD_Fval_Improvement_pct', ...
                      'SPGD_Time_Improvement_pct'});

% Remove SPGD row (optional)
spgd_improvement = spgd_improvement(~strcmp(spgd_improvement.Method, 'SPGD'), :);

% Sort by function value improvement
spgd_improvement = sortrows(spgd_improvement, 'SPGD_Fval_Improvement_pct', 'descend');

disp("📈 SPGD Improvement over Other Methods (positive = SPGD better):")
disp(spgd_improvement)

%% 🎯 Relative Closeness to Global Optimum (SPGD vs Others)

% Set the global minimum value (manually based on your benchmark)
f_global = 0;  % <-- change this based on your test function!

% Extract function value means
mean_fvals = mean_stats{1, fval_idx};

% Compute distance from global optimum
dist_to_opt = abs(mean_fvals - f_global);

% SPGD's distance
spgd_dist = dist_to_opt(2);  % SPGD index

% Compute relative closeness improvement of SPGD over others
closeness_improvement = zeros(size(dist_to_opt));

for i = 1:numel(dist_to_opt)
    if i == 2 || dist_to_opt(i) == 0  % skip SPGD itself or avoid div by zero
        closeness_improvement(i) = NaN;
    else
        closeness_improvement(i) = 100 * (dist_to_opt(i) - spgd_dist) / dist_to_opt(i);
    end
end

% Make table
closeness_table = table(methods', dist_to_opt', closeness_improvement', ...
    'VariableNames', {'Method', 'DistToGlobalOpt', 'SPGD_Closer_pct'});

% Remove SPGD row
closeness_table = closeness_table(~strcmp(closeness_table.Method, 'SPGD'), :);

% Sort by SPGD_Closer_pct
closeness_table = sortrows(closeness_table, 'SPGD_Closer_pct', 'descend');

disp("🏁 Closeness to Global Optimum: How much closer SPGD is than others")
disp(closeness_table)

%%
function [xbest, fval, T, fcount] = runGD(x0, lr, tel, Iter_max)
    x = x0;
    fval = Object(x(1), x(2));
    grad = Object_prime(x(1), x(2));
    fcount = 1;
    tic
    iter = 1;
    while iter <= Iter_max && norm(grad) >= tel
        x = x - lr * grad;
        fval = Object(x(1), x(2));
        grad = Object_prime(x(1), x(2));
        fcount = fcount + 1;
        iter = iter + 1;
    end
    xbest = x;
    T = toc * 1000;
end

function [xbest, fval, T, fcount] = runSPGD(x0, lr, tel, Iter_max)
    x = x0;
    xb = x;
    fval = Object(x(1), x(2));
    grad = Object_prime(x(1), x(2));
    fcount = 1;
    amp = 3.5;
    Iter_p = 5;
    N_rrt = 20;
    tic
    for iter = 1:Iter_max
        x = x - lr * grad;
        tempf = Object(x(1), x(2));
        fcount = fcount + 1;
        if tempf < fval
            fval = tempf;
            xb = x;
        end
        if mod(iter, Iter_p) == 0
            test_list = x + amp * (2 * rand(N_rrt, 2) - 1);
            test_vals = arrayfun(@(i) Object(test_list(i,1), test_list(i,2)), 1:N_rrt);
            fcount = fcount + N_rrt;
            [min_val, idx] = min(test_vals);
            if min_val < fval
                x = test_list(idx, :);
                fval = min_val;
                xb = x;
            end
        end
        grad = Object_prime(x(1), x(2));
        if norm(grad) < tel
            break
        end
    end
    xbest = xb;
    T = toc * 1000;
end

function [xbest, fval, T, fcount] = runBayesOpt(x0, Iter_max)
    vars = [optimizableVariable('x1', [-3, 3]), ...
            optimizableVariable('x2', [-3, 3])];
    results = bayesopt(@bayesObjectiveFcn, vars, ...
        'MaxObjectiveEvaluations', Iter_max/20, ...
        'IsObjectiveDeterministic', true, ...
        'Verbose', 0, ...
        'PlotFcn', [], ...
        'InitialX', table(x0(1), x0(2)));
    xbest = [results.XAtMinObjective.x1, results.XAtMinObjective.x2];
    fval = results.MinObjective;
    T = results.TotalElapsedTime * 1000;
    fcount = height(results.XTrace);
end

function results = bayesObjectiveFcn(x)
    results = Object(x.x1, x.x2);
end

function z = Object(x, y)
    z = 100*(y - x^2)^2 + (1 - x)^2;
end

function grad = Object_prime(x, y)
    dx = 2*x - 400*x*(y - x^2) - 2;
    dy = 200*y - 200*x^2;
    grad = [dx, dy];
end


function [xbest, fval, T, fcount] = runSA(x0, Iter_max)
    lb = [-3, -3];
    ub = [3, 3];
    fcount = 0;

    % options = optimoptions(@simulannealbnd, ...
    %     'MaxIter', Iter_max, ...
    %     'Display', 'off', ...
    %     'OutputFcn', @(x, opt, state) sa_counter(x, opt, state));
    options = optimoptions(@simulannealbnd,'MaxIter',Iter_max,'OutputFcns',@saoutfun);
    tic
    [xbest, fval, ~, output] = simulannealbnd(@ObjectSA, x0, lb, ub, options);
    T = toc * 1000;
    fcount = output.funccount;
end

function f = ObjectSA(x)
    f = 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;
end

function [stop ,optnew , changed ] = saoutfun(x,optimValues,state)
     persistent history
     stop = false;
     changed = false;
     optnew = 0;
     switch state
         case 'init'
             history.x = [];
             history.fval = [];
         case 'iter'
         % Concatenate current point and objective function
         % value with history. in must be a row vector.
           history.fval = [history.fval; optimValues.fval];
           history.x = [history.x; optimValues.x];
         case 'done'
             assignin('base','history_sa',history);
         otherwise
     end
end

function [xbest, fval, T, fcount] = runFminunc(x0)
    options = optimoptions('fminunc', ...
        'Algorithm', 'trust-region', ...
        'SpecifyObjectiveGradient', true, ...
        'Display', 'off');
    
    fun = @(x) ObjectMinco(x);
    tic
    [xbest, fval, ~, output] = fminunc(fun, x0, options);
    T = toc * 1000;
    fcount = output.funcCount;
end

function [f, g] = ObjectMinco(x)
    f = 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;

    if nargout > 1
        dx = 2*x(1) - 400*x(1)*(x(2) - x(1)^2) - 2;
        dy = 200*x(2) - 200*x(1)^2;
        g = [dx, dy];
    end
end

