%% Initialise
clc
clear
remote = 1;
ShowFigs = 1;
ExportFigs = 1;
%% set up parameters
micron_per_pixel = 1.6807;
DownSample = 1;
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
if ~exist([dir_in filesep 'summary' filesep 'PR' filesep 'results'],'dir')
    mkdir([dir_in filesep 'summary' filesep 'PR' filesep 'results'])
    if isunix
        fileattrib([dir_in filesep 'summary' filesep 'PR' filesep 'results'],'+w','a')
    end
end
if ~exist([dir_in filesep 'summary' filesep 'PR' filesep 'images'],'dir')
    mkdir([dir_in filesep 'summary' filesep 'PR' filesep 'images'])
    if isunix
        fileattrib([dir_in filesep 'summary' filesep 'PR' filesep 'images'],'+w','a')
    end
end
if ~exist([dir_in filesep 'summary' filesep 'PR' filesep 'graphs'],'dir')
    mkdir([dir_in filesep 'summary' filesep 'PR' filesep 'graphs'])
    if isunix
        fileattrib([dir_in filesep 'summary' filesep 'PR' filesep 'graphs'],'+w','a')
    end
end
% get all the image directories (identified by -CLAHEm extension) in the
% current folder
%% get the folders for processing
dir_struct = dir(dir_in);% read in the directory structure
dir_idx = [dir_struct.isdir];
[sorted_names,~] = sortrows({dir_struct(dir_idx).name}');
include_idx = contains(sorted_names,'-CLAHE');
FolderNames = sorted_names(include_idx);
%% set the start folder
% start = find(contains(FolderNames,'insert filename'));
%start = find(contains(FolderNames,'BEL-T210-B2SH-CLAHEm'));
start = 1;
%% run the program
for iF = start:numel(FolderNames)
    try
        % change the working directory
        cd(FolderNames{iF});
        % run the analysis
        [results, PR_methods] = MDFLeafVeinAnalysis_PR_v2(FolderNames{iF},micron_per_pixel,DownSample,ShowFigs,ExportFigs);
        % reset the working directory
        cd(dir_out_summary)
        % save the best results from each experiment to a single file
        if iF == 1
            writetable(results.F1,'PR_summary.xlsx','FileType','Spreadsheet','Sheet','F1 Results','Range', 'A1','WriteVariableNames',1,'WriteRowNames',0)
            writetable(results.F1_ratio,'PR_summary.xlsx','FileType','Spreadsheet','Sheet','F1 Ratio','Range', 'A1','WriteVariableNames',1,'WriteRowNames',0)
            writetable(results.FBeta2,'PR_summary.xlsx','FileType','Spreadsheet','Sheet','FBeta2 Results','Range', 'A1','WriteVariableNames',1,'WriteRowNames',0)
            writetable(results.FBeta2_ratio,'PR_summary.xlsx','FileType','Spreadsheet','Sheet','FBeta2 Ratio','Range', 'A1','WriteVariableNames',1,'WriteRowNames',0)
            writetable(PR_methods.evaluation_fw,'PR_summary.xlsx','FileType','Spreadsheet','Sheet','FW evaluation','Range', 'A1','WriteVariableNames',1,'WriteRowNames',0)
            writetable(PR_methods.evaluation_sk,'PR_summary.xlsx','FileType','Spreadsheet','Sheet','SK evaluation','Range', 'A1','WriteVariableNames',1,'WriteRowNames',0)            
            if isunix
                fileattrib('PR_summary.xlsx', '+w','a')
            end
        else
            writetable(results.F1,'PR_summary.xlsx','FileType','Spreadsheet','Sheet','F1 Results','Range', ['A' num2str(((iF-1)*11)+2)],'WriteVariableNames',0,'WriteRowNames',0)
            writetable(results.F1_ratio,'PR_summary.xlsx','FileType','Spreadsheet','Sheet','F1 Ratio','Range', ['A' num2str(((iF-1)*11)+2)],'WriteVariableNames',0,'WriteRowNames',0)
            writetable(results.FBeta2,'PR_summary.xlsx','FileType','Spreadsheet','Sheet','FBeta2 Results','Range', ['A' num2str(((iF-1)*11)+2)],'WriteVariableNames',0,'WriteRowNames',0)
            writetable(results.FBeta2_ratio,'PR_summary.xlsx','FileType','Spreadsheet','Sheet','FBeta2 Ratio','Range', ['A' num2str(((iF-1)*11)+2)],'WriteVariableNames',0,'WriteRowNames',0)
            writetable(PR_methods.evaluation_fw,'PR_summary.xlsx','FileType','Spreadsheet','Sheet','FW evaluation','Range', ['A' num2str(((iF-1)*11)+3)],'WriteVariableNames',0,'WriteRowNames',0)
            writetable(PR_methods.evaluation_sk,'PR_summary.xlsx','FileType','Spreadsheet','Sheet','SK evaluation','Range', ['A' num2str(((iF-1)*11)+3)],'WriteVariableNames',0,'WriteRowNames',0)            
        end
%                     xls_delete_sheets('PR_summary.xlsx',{'Sheet1','Sheet2','Sheet3'}) 
        cd(dir_in);
        disp([FolderNames{iF} ' analysis complete'])
    catch ME
        ME
        disp(['ERROR: Could not process folder ' FolderNames{iF}])
        % reset the working directory
        cd(dir_in);
    end
end
warning on

