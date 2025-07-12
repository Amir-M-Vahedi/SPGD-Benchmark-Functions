% 📂 Define benchmark names and filenames
benchmarks = {'Ackley', 'Easom', 'Levi', 'Peaks'};
all_results = [];

% 📊 Initialize storage for summary tables
success_tables = struct();
improvement_tables = struct();
closeness_tables = struct();

for i = 1:length(benchmarks)
    name = benchmarks{i};
    filename = ['results_' name '.mat'];
    
    % Load file
    load(filename, 'results_table', 'success_table', 'spgd_improvement', 'closeness_table');
    
    % Add Benchmark name
    results_table.Benchmark = repmat({name}, height(results_table), 1);
    success_table.Benchmark = repmat({name}, height(success_table), 1);
    spgd_improvement.Benchmark = repmat({name}, height(spgd_improvement), 1);
    closeness_table.Benchmark = repmat({name}, height(closeness_table), 1);
    
    % Store
    all_results = [all_results; results_table]; %#ok<AGROW>
    success_tables.(name) = success_table;
    improvement_tables.(name) = spgd_improvement;
    closeness_tables.(name) = closeness_table;
end

% 💾 Optionally, save to file
save('all_benchmark_results.mat', 'all_results', 'success_tables', 'improvement_tables', 'closeness_tables');

% 📄 Display clean format for one example
disp("===== Example: Convergence Successes - Ackley =====")
disp(success_tables.Ackley)

disp("===== Example: SPGD Improvement - Levi =====")
disp(improvement_tables.Levi)

disp("===== Example: Closeness to Optimum - Peaks =====")
disp(closeness_tables.Peaks)

% 🧾 Optional: export to LaTeX or CSV
% writetable(success_tables.Ackley, 'success_ackley.csv')
% writetable(improvement_tables.Ackley, 'improvement_ackley.csv')
% writetable(closeness_tables.Ackley, 'closeness_ackley.csv')
