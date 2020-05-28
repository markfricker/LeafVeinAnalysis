% select the folder with the trained networks
dir_in = uigetdir('Select folder with trained models');
cd(dir_in)
if ~exist('Matlab models','dir')
    mkdir('Matlab models')
end
dir_out = ['Matlab models' filesep];
% get all the trained networks in the current folder
dir_struct = dir(dir_in);% read in the directory structure
file_idx = ~[dir_struct.isdir];% strip out any directories
[sorted_names,~] = sortrows({dir_struct(file_idx).name}');% sort the filenames in order
% now show only files with selected extensions
fnc_find_ext = @(file_ext) ~cellfun(@isempty, regexpi(sorted_names,['\w+' file_ext],'match')); % find any files with the appropriate extension
included = cellfun(fnc_find_ext, {'.h5'}, 'UniformOutput',0);% feed array of extensions to the find function
include_idx = max(cell2mat(included'),[],2);% collapse the boolean output into a linear index
filenames = sorted_names(include_idx);
for iF = 1:numel(filenames)
    modelfile = filenames{iF};
    lgraph = importKerasLayers(modelfile,'ImportWeights',true);
    placeholderLayers = findPlaceholderLayers(lgraph);
    for i = 1:numel(placeholderLayers)
        %Number of filters needs to be the same as number of channels in the previous layer.
        numFilters = 2^(10-i);
        filterSize = 2;
        name = ['up_sampling2d_' num2str(i)];
        upSample = transposedConv2dLayer( ...
            [filterSize filterSize], ...
            numFilters, ...
            'stride',[2 2], ...
            'cropping','same', ...
            'weights',zeros(filterSize,filterSize,numFilters,numFilters), ...
            'bias',zeros(1,1,numFilters), ...
            'name',name);
        %First filter should act only on the first channel in the previous layer.
        %Second filter should act only on the second channel in the previous layer and so on.
        %So we need to 'turn on' only the corresponding filter for the channel of interest.
        for j = 1:numFilters
            upSample.Weights(:,:,j,j) = 1;
        end
        lgraph = replaceLayer(lgraph,placeholderLayers(i).Name,upSample);
    end
    net = assembleNetwork(lgraph);
    [~,fname,~] = fileparts(modelfile);
    save(fullfile(dir_out,[fname '.mat']),'net')
end
