clc; clear; close all;

%% Settings
nStarts = 30;              % Number of random starting points
Iter_max = 2e3;            % Max iterations for all methods
lr = 0.01;                % Learning rate
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
    [~, f_pgd, t_pgd, c_pgd] = runPGD(x0, lr, tel, Iter_max);
    [~, f_bayes, t_bayes, c_bayes] = runBayesOpt(x0, Iter_max);
    [~, f_sa, t_sa, c_sa] = runSA(x0, Iter_max);
    [~, f_unc, t_unc, c_unc] = runFminunc(x0);
    [~, f_con, t_con, c_con] = runFmincon(x0);


    results = [results; ...
        i, f_gd, t_gd, c_gd, ...
        f_spgd, t_spgd, c_spgd, ...
        f_pgd, t_pgd, c_pgd, ...
        f_bayes, t_bayes, c_bayes, ...
        f_sa, t_sa, c_sa, ...
        f_unc, t_unc, c_unc, ...
        f_con, t_con, c_con];

    fprintf("✅ Finished point #%d: GD=%.3f, SPGD=%.3f, PGD=%.3f, BO=%.3f, SA=%.3f, Fminunc=%.3f, Fmincon=%.3f\n\n", ...
        i, f_gd, f_spgd, f_pgd, f_bayes, f_sa, f_unc, f_con);
end

colNames = {'Run', ...
    'GD_Fval', 'GD_Time', 'GD_Count', ...
    'SPGD_Fval', 'SPGD_Time', 'SPGD_Count', ...
    'PGD_Fval', 'PGD_Time', 'PGD_Count', ...
    'BayesOpt_Fval', 'BayesOpt_Time', 'BayesOpt_Count', ...
    'SA_Fval', 'SA_Time', 'SA_Count', ...
    'Fminunc_Fval', 'Fminunc_Time', 'Fminunc_Count',...
    'Fmincon_Fval', 'Fmincon_Time', 'Fmincon_Count'};
results_table = array2table(results, 'VariableNames', colNames);

mean_stats = varfun(@mean, results_table(:, 2:end));
mean_stats.Properties.RowNames = {'Mean'};
std_stats = varfun(@std, results_table(:, 2:end));
std_stats.Properties.RowNames = {'StdDev'};

disp(results_table)
disp(mean_stats)
disp(std_stats)

%% 📊 SPGD Improvement
methods = {'GD', 'SPGD', 'PGD', 'BayesOpt', 'SA', 'Fminunc', 'Fmincon'};
fval_idx  = [1, 4, 7, 10, 13, 16, 19];
time_idx  = [2, 5, 8, 11, 14, 17, 20];

mean_fvals = mean_stats{1, fval_idx};
mean_times = mean_stats{1, time_idx};
spgd_fval = mean_fvals(2);
spgd_time = mean_times(2);

improvement_fval = zeros(1, numel(methods));
improvement_time = zeros(1, numel(methods));

for i = 1:numel(methods)
    if i == 2
        improvement_fval(i) = NaN;
        improvement_time(i) = NaN;
        continue
    end
    denom_fval = mean_fvals(i);
    denom_time = mean_times(i);
    if abs(denom_fval) > 1e-8
        improvement_fval(i) = 100 * (mean_fvals(i) - spgd_fval) / abs(denom_fval);
    else
        improvement_fval(i) = NaN;
    end
    if abs(denom_time) > 1e-8
        improvement_time(i) = 100 * (mean_times(i) - spgd_time) / denom_time;
    else
        improvement_time(i) = NaN;
    end
end

spgd_improvement = table(methods', improvement_fval', improvement_time', ...
    'VariableNames', {'Method', 'SPGD_Fval_Improvement_pct', 'SPGD_Time_Improvement_pct'});
spgd_improvement = spgd_improvement(~strcmp(spgd_improvement.Method, 'SPGD'), :);
spgd_improvement = sortrows(spgd_improvement, 'SPGD_Fval_Improvement_pct', 'descend');
disp("📈 SPGD Improvement:")
disp(spgd_improvement)

%% 🎯 Closeness to Global Optimum
f_global = -6.5511;
dist_to_opt = abs(mean_fvals - f_global);
spgd_dist = dist_to_opt(2);

closeness_improvement = zeros(size(dist_to_opt));
for i = 1:numel(dist_to_opt)
    if i == 2 || dist_to_opt(i) == 0
        closeness_improvement(i) = NaN;
    else
        closeness_improvement(i) = 100 * (dist_to_opt(i) - spgd_dist) / dist_to_opt(i);
    end
end

closeness_table = table(methods', dist_to_opt', closeness_improvement', ...
    'VariableNames', {'Method', 'DistToGlobalOpt', 'SPGD_Closer_pct'});
closeness_table = closeness_table(~strcmp(closeness_table.Method, 'SPGD'), :);
closeness_table = sortrows(closeness_table, 'SPGD_Closer_pct', 'descend');
disp("🏁 Closeness to Global Optimum:")
disp(closeness_table)

%% ✅ Count Convergence Successes
tol = 1e-2;
global_min = -6.5511;
count_gd = sum(abs(results_table.GD_Fval - global_min) <= tol);
count_pgd = sum(abs(results_table.PGD_Fval - global_min) <= tol);
count_spgd = sum(abs(results_table.SPGD_Fval - global_min) <= tol);
count_bayes = sum(abs(results_table.BayesOpt_Fval - global_min) <= tol);
count_sa = sum(abs(results_table.SA_Fval - global_min) <= tol);
count_fminunc = sum(abs(results_table.Fminunc_Fval - global_min) <= tol);
count_fmincon = sum(abs(results_table.Fmincon_Fval - global_min) <= tol);

fprintf("\n🎯 Convergence to Global Optimum (within %.1e):\n", tol);
fprintf("GD:        %d / %d\n", count_gd, nStarts);
fprintf("PGD:       %d / %d\n", count_pgd, nStarts);
fprintf("SPGD:      %d / %d\n", count_spgd, nStarts);
fprintf("BayesOpt:  %d / %d\n", count_bayes, nStarts);
fprintf("SA:        %d / %d\n", count_sa, nStarts);
fprintf("Fminunc:   %d / %d\n", count_fminunc, nStarts);
fprintf("Fmincon:   %d / %d\n", count_fmincon, nStarts);

method_names = {'GD', 'PGD', 'SPGD', 'BayesOpt', 'SA', 'Fminunc', 'Fmincon'}';
success_counts = [count_gd; count_pgd; count_spgd; count_bayes; count_sa; count_fminunc; count_fmincon];
success_table = table(method_names, success_counts, ...
    'VariableNames', {'Method', 'ConvergedRuns'});

%% 💾 Save all results
benchmark_name = 'Peaks';
save_filename = ['results_' benchmark_name '.mat'];
save(save_filename, 'results_table', 'mean_stats', 'std_stats', ...
    'spgd_improvement', 'closeness_table', 'success_table');

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

function [xbest, fval, T, fcount] = runPGD(x0, lr, tel, Iter_max)
    N_rrt = 5;
    amp = 4.5;
    xb = x0;
    x = x0;
    xnoise = x;
    O = Object(x(1), x(2));
    fcount = 1;
    Onoise = O;
    grad = Object_prime(x(1), x(2));
    Iter = 1;
    Iter_threshold = 5;
    Iter_noise = -Iter_threshold - 1;

    tic
    while Iter <= Iter_max
        x = x - lr * grad;
        Otemp = Object(x(1), x(2));
        fcount = fcount + 1;

        if Otemp < O
            O = Otemp;
            xb = x;
        end

        if norm(grad) <= 1000 * tel && (Iter - Iter_noise) > Iter_threshold
            Iter_noise = Iter;
            Onoise = Otemp;
            xnoise = x;
            x = x + amp * (2 * rand(1, 2) - 1);
            Otemp = Object(x(1), x(2));
            fcount = fcount + 1;
            if Otemp < O
                O = Otemp;
                xb = x;
            end
        end

        if Iter - Iter_noise == Iter_threshold
            if Otemp > Onoise
                Otemp = Onoise;
                O = Otemp;
                xb = xnoise;
                break
            end
        end

        grad = Object_prime(x(1), x(2));
        Iter = Iter + 1;
    end
    T = toc * 1000;
    xbest = xb;
    fval = O;
end


function [xbest, fval, T, fcount] = runSPGD(x0, lr, tel, Iter_max)
    x = x0;
    xb = x;
    fval = Object(x(1), x(2));
    grad = Object_prime(x(1), x(2));
    fcount = 1;
    amp = 4.5;
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
            if min_val <= tempf
                x = test_list(idx, :);
                if min_val<= fval
                    fval = min_val;
                    xb = x;
                end
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
    vars = [optimizableVariable('x1', [-4, 4]), ...
            optimizableVariable('x2', [-4, 4])];
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
    term1 = 3 * (1 - x)^2 * exp(-x^2 - (y + 1)^2);
    term2 = -10 * (x/5 - x^3 - y^5) * exp(-x^2 - y^2);
    term3 = -exp(-(x + 1)^2 - y^2) / 3;
    z = term1 + term2 + term3;
end

function grad = Object_prime(x, y)
    % Shared terms
    exp1 = exp(-x^2 - (y + 1)^2);
    exp2 = exp(-x^2 - y^2);
    exp3 = exp(-(x + 1)^2 - y^2);

    % Gradients
    dfdx = exp3 * (2*x + 2)/3 + ...
           3 * exp1 * (2*x - 2) + ...
           exp2 * (30*x^2 - 2) - ...
           6*x * exp1 * (1 - x)^2 - ...
           2*x * exp2 * (10*x^3 - 2*x + 10*y^5);

    dfdy = 2*y * exp3 / 3 + ...
           50*y^4 * exp2 - ...
           3 * exp1 * (2*y + 2) * (1 - x)^2 - ...
           2*y * exp2 * (10*x^3 - 2*x + 10*y^5);

    grad = [dfdx, dfdy];
end


function [xbest, fval, T, fcount] = runSA(x0, Iter_max)
    lb = [-4, -4];
    ub = [4, 4];
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
    term1 = 3 * (1 - x(1))^2 * exp(-x(1)^2 - (x(2) + 1)^2);
    term2 = -10 * (x(1)/5 - x(1)^3 - x(2)^5) * exp(-x(1)^2 - x(2)^2);
    term3 = -exp(-(x(1) + 1)^2 - x(2)^2) / 3;
    f = term1 + term2 + term3;
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
    x1 = x(1); y1 = x(2);

    exp1 = exp(-x1^2 - (y1 + 1)^2);
    exp2 = exp(-x1^2 - y1^2);
    exp3 = exp(-(x1 + 1)^2 - y1^2);

    term1 = 3 * (1 - x1)^2 * exp1;
    term2 = -10 * (x1/5 - x1^3 - y1^5) * exp2;
    term3 = -exp3 / 3;
    f = term1 + term2 + term3;

    if nargout > 1
        dfdx = exp3 * (2*x1 + 2)/3 + ...
               3 * exp1 * (2*x1 - 2) + ...
               exp2 * (30*x1^2 - 2) - ...
               6*x1 * exp1 * (1 - x1)^2 - ...
               2*x1 * exp2 * (10*x1^3 - 2*x1 + 10*y1^5);

        dfdy = 2*y1 * exp3 / 3 + ...
               50*y1^4 * exp2 - ...
               3 * exp1 * (2*y1 + 2) * (1 - x1)^2 - ...
               2*y1 * exp2 * (10*x1^3 - 2*x1 + 10*y1^5);

        g = [dfdx, dfdy];
    end
end

function [xbest, fval, T, fcount] = runFmincon(x0)
    options = optimoptions('fmincon', ...
        'Algorithm', 'interior-point', ...
        'SpecifyObjectiveGradient', true, ...
        'Display', 'off');

    fun = @(x) ObjectMinco(x);
    lb = [-4, -4];
    ub = [4, 4];
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    nonlcon = [];

    tic
    [xbest, fval, ~, output] = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options);
    T = toc * 1000;
    fcount = output.funcCount;
end