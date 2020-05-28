%% Initialise
clc
clear
opengl software % to mimic the remote environment
remote = 1;
ShowFigs = 1;
ExportFigs = 1;
FullMetrics = 1;
FullLeaf = 0;
%% set up parameters
micron_per_pixel = 1.6807;
DownSample = 2;
%% set up starting directory on the remote server
if remote == 1
    addpath('/soge-home/projects/leaf-gpu/Matlab working')
    addpath('/soge-home/projects/leaf-gpu/results')
    dir_in = '/soge-home/projects/leaf-gpu/results';
    cd '/soge-home/projects/leaf-gpu/results'
else
    dir_in = pwd;
end
%% Set-up working directories and write permisions
cd(dir_in)
if ~exist('summary','dir')
    mkdir('summary')
    if isunix
        fileattrib('summary','+w','a')
    end
end
dir_out_summary = ['..' filesep 'summary' filesep];
if ~exist([dir_in filesep 'summary' filesep 'images'],'dir')
    mkdir([dir_in filesep 'summary' filesep 'images'])
    if isunix
        fileattrib([dir_in filesep 'summary' filesep 'images'],'+w','a')
    end
end
if ~exist([dir_in filesep 'summary' filesep 'width'],'dir')
    mkdir([dir_in filesep 'summary' filesep 'width'])
    if isunix
        fileattrib([dir_in filesep 'summary' filesep 'width'],'+w','a')
    end
end
if ~exist([dir_in filesep 'summary' filesep 'data'],'dir')
    mkdir([dir_in filesep 'summary' filesep 'data'])
    if isunix
        fileattrib([dir_in filesep 'summary' filesep 'data'],'+w','a')
    end
end
if ~exist([dir_in filesep 'summary' filesep 'HLD'],'dir')
    mkdir([dir_in filesep 'summary' filesep 'HLD'])
    if isunix
        fileattrib([dir_in filesep 'summary' filesep 'HLD'],'+w','a')
    end
end
%% get all the image directories (identified by -CLAHEm extension) in the
% current folder
dir_struct = dir(dir_in);% read in the directory structure
dir_idx = [dir_struct.isdir];
[sorted_names,~] = sortrows({dir_struct(dir_idx).name}');
include_idx = contains(sorted_names,'-CLAHE');
FolderNames = sorted_names(include_idx);
% check whether a full PR analysis has been complete to get the optimal
% threshold value
if exist([ 'summary' filesep 'PR_summary.xlsx'],'file')
    % load in the full skeleton analysis
    PR_results = readtable(['summary' filesep 'PR_summary.xlsx'],'FileType','Spreadsheet','Sheet','SK evaluation');
    PR_flag = 1;
else
    PR_flag = 0;
end
%% set the start folder
%start = find(contains(FolderNames,'insert filename'));
start = 1;
for iF = start:numel(FolderNames)
    try
        % change the working directory
        cd(FolderNames{iF});
        % get the threshold 
        if PR_flag == 1
            idx = find(ismember(PR_results.Filename,FolderNames{iF}) & ismember(PR_results.Method,'CNN'));
            threshold = PR_results{idx,'FBeta2_threshold'};
        else
            threshold = 0.379;
        end
        % run the analysis
        results = MDFLeafVeinAnalysis_v9(FolderNames{iF},micron_per_pixel,DownSample,threshold,ShowFigs,ExportFigs,FullLeaf,FullMetrics);
    catch ME
        ME
        disp(['ERROR: Could not process folder ' FolderNames{iF}])
        ME.identifier
        results = cell2table({FolderNames(iF), ME.identifier});
    end
    % save the results
    try
        % reset the working directory
        cd(dir_out_summary)
        % save the results to a single file
        if iF == 1
            writetable(results,'results.xlsx','FileType','Spreadsheet','Range', 'A1','WriteVariableNames',1,'WriteRowNames',1)
            if isunix
                fileattrib('results.xlsx', '+w','a')
            end
        else
            writetable(results,'results.xlsx','FileType','Spreadsheet','Range', ['A' num2str(iF+1)],'WriteVariableNames',0,'WriteRowNames',1)
        end
        cd(dir_in);
        disp([FolderNames{iF} ' analysis complete'])
    catch ME
        ME
        disp(['ERROR: Could not save data from folder ' FolderNames{iF}])
        % reset the working directory
        cd(dir_in);
    end
end
warning on

