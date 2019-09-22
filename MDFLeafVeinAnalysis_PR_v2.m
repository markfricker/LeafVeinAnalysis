function [results, PR_methods] = MDFLeafVeinAnalysis_PR_v2(FolderName,micron_per_pixel,DownSample,ShowFigs,ExportFigs)
%% set up directories
dir_out_PR_results = ['..' filesep 'summary' filesep 'PR' filesep 'results' filesep];
dir_out_PR_images = ['..' filesep 'summary' filesep 'PR' filesep 'images' filesep];
dir_out_PR_graphs = ['..' filesep 'summary' filesep 'PR' filesep 'graphs' filesep];
%% set up parameters
micron_per_pixel = micron_per_pixel.*DownSample;
sk_width = 3;
E_width = 1;
%% set up default colour map
cmap = jet(256);
cmap(1,:) = 0;
%% initialise main program
step = 0;
warning off
%% Load in the images
step = step+1;
disp(['Step ' num2str(step) ': Processing ' FolderName])
[im,im_cnn,bw_mask,bw_vein,bw_roi,bw_GT] = fnc_load_CNN_images(FolderName,DownSample);
%% subsample the images to match the ground truth region
step = step+1;
disp(['Step ' num2str(step) ': Precision-Recall analysis'])
% subsample images to match the area used for the ground truth
stats = regionprops(bw_roi,'BoundingBox');
BB = round(stats.BoundingBox);
PR_methods.roi.bw = bw_roi(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
PR_methods.roi.im = im(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
PR_methods.roi.cnn = im_cnn(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
PR_methods.roi.mask = bw_mask(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
PR_methods.roi.vein = bw_vein(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
PR_methods.roi.GT = bw_GT(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
% keep the largest connected component
% roi_GT = bwareafilt(roi_GT,1);
% invert and normalise the original image
PR_methods.roi.invert = imcomplement(mat2gray(PR_methods.roi.im));
%% set up threshold parameters
Tmin = 0;
Tmax = 1;
nT = 20;
Tint = (Tmax-Tmin)/(nT);
T = Tmin:Tint:Tmax-Tint;
%% set up an array of methods
PR_methods.name = {'CNN';'Vesselness';'FeatureType';'BowlerHat';'midgrey';'Niblack';'Bernsen';'Sauvola';'MFATl';'MFATp'};
% choose which methods to test
PR_methods.select = logical([1 1 1 1 1 0 0 0 0 0]);
% set up the evaluation table for the full width images
PR_methods.evaluation_fw = array2table(zeros(numel(PR_methods.name),6,'single'));
PR_methods.evaluation_fw.Properties.VariableNames = { ...
    'F1' 'F1_idx' 'F1_threshold' ...
    'FBeta2' 'FBeta2_idx' 'FBeta2_threshold'};
PR_methods.evaluation_fw.Filename = repelem({FolderName},height(PR_methods.evaluation_fw),1);
PR_methods.evaluation_fw = PR_methods.evaluation_fw(:,[end, 1:end-1]);
PR_methods.evaluation_fw.Properties.RowNames = PR_methods.name;
% set up an evaluation table for the skeleton comparison
PR_methods.evaluation_sk = array2table(zeros(numel(PR_methods.name),6,'single'));
PR_methods.evaluation_sk.Properties.VariableNames = { ...
    'F1' 'F1_idx' 'F1_threshold' ...
    'FBeta2' 'FBeta2_idx' 'FBeta2_threshold'};
PR_methods.evaluation_sk.Filename = repelem({FolderName},height(PR_methods.evaluation_sk),1);
PR_methods.evaluation_sk = PR_methods.evaluation_sk(:,[end, 1:end-1]);
PR_methods.evaluation_sk.Properties.RowNames = PR_methods.name;
%% add in file information to the results file
results.File = FolderName;
results.TimeStamp = datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss Z');
results.MicronPerPixel = micron_per_pixel;
results.DownSample = DownSample;
%% set up blank results tables
results.F1 = array2table(nan(11,11));
results.F1.Properties.RowNames = [{'GT'};PR_methods.name];
results.FBeta2 = results.F1;
results.F1_ratio = results.F1;
results.FBeta2_ratio = results.F1;
%% loop through each method
n = 0;
for iM = 1:sum(PR_methods.select)
    n = n+1;
    idx = find(PR_methods.select);
    method = PR_methods.name{idx(iM)};
    % normalise and enhance the images with variable thresholds
    switch method
        case {'CNN'}
            % normalise the CNN image
            PR_methods.enhanced.(method) = (PR_methods.roi.cnn./255);
        case {'Vesselness';'FeatureType';'BowlerHat';'MFATl';'MFATp'}
            % calculate the enhanced images
            PR_methods.enhanced.(method) = fnc_enhance_im(PR_methods.roi.invert,DownSample,method);
        otherwise
            % copy the original image across for local intensity
            % segmentation
            PR_methods.enhanced.(method) = fnc_enhance_im(PR_methods.roi.im,DownSample,method);
    end
    % extract the full width binary image
    switch method
        case {'CNN';'Vesselness';'FeatureType';'BowlerHat';'MFATl';'MFATp'}
            % use a hysteresis threshold for the enhanced images
            PR_methods.fullwidth.(method) = fnc_im_to_bw(PR_methods.enhanced.(method),PR_methods.roi.bw,PR_methods.roi.vein,PR_methods.roi.mask,T,'hysteresis');
        case {'Niblack';'midgrey';'Bernsen';'Sauvola'}
            % apply the local thresholding to the other images
            PR_methods.fullwidth.(method) = fnc_im_to_bw(PR_methods.enhanced.(method),PR_methods.roi.bw,PR_methods.roi.vein,PR_methods.roi.mask,T,method);
        otherwise
            PR_methods.fullwidth.(method) = PR_methods.enhanced.(method);
    end
    % evaluate performance against the ground truth using Precision-Recall analysis
    [PR_methods.results_fw.(method), PR_methods.images_fw.(method)] = fnc_PRC_bw(PR_methods.roi.GT,PR_methods.fullwidth.(method),T);
    % get the best values from the results table
    [PR_methods.evaluation_fw{method,'F1'}, PR_methods.evaluation_fw{method,'F1_idx'}] = max(PR_methods.results_fw.(method){:,'F1'});
    PR_methods.evaluation_fw{method,'F1_threshold'} = T(PR_methods.evaluation_fw{method,'F1_idx'});
    [PR_methods.evaluation_fw{method,'FBeta2'}, PR_methods.evaluation_fw{method,'FBeta2_idx'}] = max(PR_methods.results_fw.(method){:,'FBeta2'});
    PR_methods.evaluation_fw{method,'FBeta2_threshold'} = T(PR_methods.evaluation_fw{method,'FBeta2_idx'});
    % keep the best full width images
    F1_idx_fw = PR_methods.evaluation_fw{method,'F1_idx'};
    FBeta2_idx_fw = PR_methods.evaluation_fw{method,'FBeta2_idx'};
    PR_methods.F1_fw{idx(iM)} = PR_methods.images_fw.(method)(:,:,F1_idx_fw);
    PR_methods.FBeta2_fw{idx(iM)} = PR_methods.images_fw.(method)(:,:,FBeta2_idx_fw);
    % convert to a skeleton
    if n == 1
        % get the ground truth skeleton
        [PR_methods.roi.sk.('GT')] = fnc_im_to_sk(PR_methods.roi.GT,PR_methods.roi.vein,PR_methods.roi.bw,T,'GT');
    end
    % get the processed skeleton for all threshold values
    PR_methods.sk.(method) = fnc_im_to_sk(PR_methods.fullwidth.(method),PR_methods.roi.vein,PR_methods.roi.bw,T,method);
    % compare the skeletons with the skeleton ground truth within a given tolerance (in pixels)
    tolerance = 3;
    [PR_methods.results_sk.(method),PR_methods.images_sk.(method)] = fnc_PRC_sk(PR_methods.roi.sk.('GT'),PR_methods.sk.(method),tolerance,T);
    % get the best values from the results table
    [PR_methods.evaluation_sk{method,'F1'}, PR_methods.evaluation_sk{method,'F1_idx'}] = max(PR_methods.results_sk.(method){:,'F1'});
    PR_methods.evaluation_sk{method,'F1_threshold'} = T(PR_methods.evaluation_sk{method,'F1_idx'});
    [PR_methods.evaluation_sk{method,'FBeta2'}, PR_methods.evaluation_sk{method,'FBeta2_idx'}] = max(PR_methods.results_sk.(method){:,'FBeta2'});
    PR_methods.evaluation_sk{method,'FBeta2_threshold'} = T(PR_methods.evaluation_sk{method,'FBeta2_idx'});
    % keep the best skeleton
    F1_idx_sk = PR_methods.evaluation_sk{method,'F1_idx'};
    FBeta2_idx_sk = PR_methods.evaluation_sk{method,'FBeta2_idx'};
    PR_methods.F1_sk{idx(iM)} = PR_methods.sk.(method)(:,:,F1_idx_sk);
    PR_methods.FBeta2_sk{idx(iM)} = PR_methods.sk.(method)(:,:,FBeta2_idx_sk);
    % get some basic graph results for the ground truth
    if n ==1
        GT_full = fnc_skeleton_analysis(PR_methods.roi.sk.('GT'),PR_methods.roi.bw);
    end
    results_full = fnc_skeleton_analysis(PR_methods.sk.(method),PR_methods.roi.bw);
    if iM == 1
        results.('GT') = GT_full;
        results.F1(1,:) = GT_full;
        results.FBeta2(1,:) = GT_full;
        results.F1.Properties.VariableNames = results_full.Properties.VariableNames;
        results.FBeta2.Properties.VariableNames = results_full.Properties.VariableNames;
    end
    results.(method) = results_full;
    results.F1(idx(iM)+1,:) = results_full(F1_idx_sk,:);
    results.FBeta2(idx(iM)+1,:) = results_full(FBeta2_idx_sk,:);
end
%% calcuate the ratio values against the ground truth
results.F1_ratio([true PR_methods.select],:) = array2table(log(results.F1{[true PR_methods.select],:}./results.F1{1,:}));
results.F1_ratio.Properties.VariableNames = results.F1.Properties.VariableNames;
results.F1_ratio.Properties.RowNames = results.F1.Properties.RowNames;
results.FBeta2_ratio([true PR_methods.select],:) = array2table(log(results.FBeta2{[true PR_methods.select],:}./results.FBeta2{1,:}));
results.FBeta2_ratio.Properties.VariableNames = results.FBeta2.Properties.VariableNames;
results.FBeta2_ratio.Properties.RowNames = results.FBeta2.Properties.RowNames;
%% display the PR graphs
hfig = fnc_display_threshold_results(results,PR_methods);
export_fig([dir_out_PR_graphs FolderName '_PR_threshold'],'-native','-png',hfig)
delete(hfig)
hfig = fnc_display_fw_PR(PR_methods);
export_fig([dir_out_PR_graphs FolderName '_PR_fw_plots'],'-native','-png',hfig)
delete(hfig)
hfig = fnc_display_sk_PR(PR_methods);
export_fig([dir_out_PR_graphs FolderName '_PR_sk_plots'],'-native','-png',hfig)
delete(hfig)
%% display the figure
hfig = fnc_display_images(PR_methods);
export_fig([dir_out_PR_images FolderName '_PR_images'],'-native','-png',hfig)
delete(hfig)
%% add in the file and method indexes
results.F1.Filename = repelem({FolderName},height(results.F1),1);
results.F1.Method = results.F1.Properties.RowNames;
results.F1 = results.F1(:,[end-1:end, 1:end-2]);
results.F1_ratio.Filename = repelem({FolderName},height(results.F1_ratio),1);
results.F1_ratio.Method = results.F1_ratio.Properties.RowNames;
results.F1_ratio = results.F1_ratio(:,[end-1:end, 1:end-2]);
results.FBeta2.Filename = repelem({FolderName},height(results.FBeta2),1);
results.FBeta2.Method = results.FBeta2.Properties.RowNames;
results.Beta2 = results.FBeta2(:,[end-1:end, 1:end-2]);
results.FBeta2_ratio.Filename = repelem({FolderName},height(results.FBeta2_ratio),1);
results.FBeta2_ratio.Method = results.FBeta2_ratio.Properties.RowNames;
results.FBeta2_ratio = results.FBeta2_ratio(:,[end-1:end, 1:end-2]);
%% save the results
save([dir_out_PR_results [FolderName '_results']],'results','PR_methods')
end
%% all functions
function [im,im_cnn,bw_mask,bw_vein,bw_roi,bw_GT] = fnc_load_CNN_images(FolderName,DownSample)
% get the contents of the directory
Files = dir;
% find the img files to be averaged
matches = regexpi({Files.name},'_cnn(|_\d*).png');
idx = find(~cellfun('isempty',matches));
% get the size of the first image
info = imfinfo(Files(idx(1)).name);
nX = info.Width;
nY = info.Height;
nI = length(idx);
% accumulate each cnn file into a stack
cnn = zeros(nY,nX,nI,'uint8');
for iI = 1:nI
    % get the name of the original image
    FileName = Files(idx(iI)).name;
    % read in the original png image using the matlab filters
    cnn(:,:,iI) = imread(FileName);
end
% average and downsample the cnn images
if DownSample > 1
    im_cnn = imresize(mean(cnn,3),1/DownSample);
else
    im_cnn = mean(cnn,3);
end
% reset the size
[nY,nX] = size(im_cnn);
% set up the default filenames
img_name = [FolderName '_img.png'];
mask_name = [FolderName '_mask.png'];
vein_name = [FolderName '_big_mask.png'];
cnn_mask_name = [FolderName '_cnn_mask.png'];
roi_name = [FolderName '_roi.png'];
GT_name = [FolderName '_seg.png'];
% load the original file
if exist(img_name,'file') == 2
    im = imresize(imread(img_name),[nY,nX]);
else
    disp('no original image')
    im = zeros(nY,nX, 'uint8');
end
% load in the mask images
if exist(mask_name,'file') == 2
    bw_mask = imresize(logical(imread(mask_name)),[nY,nX]);
else
    disp('no mask image')
    bw_mask = true(nY,nX);
end
if exist(cnn_mask_name,'file') == 2
    bw_cnn_mask = imresize(logical(imread(cnn_mask_name)),[nY,nX]);
else
    disp('no cnn mask image')
    bw_cnn_mask = true(nY,nX);
end
% load in the big vein image if present
if exist(vein_name,'file') == 2
    bw_vein = imresize(logical(imread(vein_name)),[nY,nX]);
else
    disp('no manual vein image')
    bw_vein = false(nY,nX);
end
% load in the manual roi and ground truth images
if exist(roi_name,'file') == 2
    bw_roi = imresize(logical(imread(roi_name)>0),[nY,nX]);
else
    disp('no roi image')
    bw_roi = true(nY,nX);
end
if exist(GT_name,'file') == 2
    bw_GT = imresize(logical(imread(GT_name)>0),[nY,nX]);
else
    disp('no GT image')
    bw_GT = false(nY,nX);
end
% apply the masks
bw_mask = bw_mask & bw_cnn_mask;
im_cnn(~bw_mask) = 0;
end

function im_out = fnc_enhance_im(im_in,DownSample,method)
switch method
    case {'im';'midgrey';'Niblack';'Bernsen';'Sauvola';'Singh'}
        im_out = imcomplement(mat2gray(im_in));
    case {'Vesselness';'FeatureType';'BowlerHat';'MFATl';'MFATp'} % use an image pyramid to span scales
        % find the scales for the filter
        minW = floor(9/DownSample);
        maxW = ceil(150/DownSample);
        % set up an image pyramid
        nLevels = 4;
        I = cell(nLevels,1);
        I{1} = mat2gray(im_in);
        for iL = 2:nLevels
            I{iL} = impyramid(I{iL-1},'reduce');
        end
        im_out = zeros(size(im_in),'single');
        D = ones(3);
end
switch method
    case 'Vesselness'
        % filter each scale with an overlap
        for iL = 1:nLevels
            for iW = minW:2*minW+1
                temp = fibermetric(I{iL}, iW, 'ObjectPolarity', 'bright');
                temp = imclose(temp,D);
                im_out = max(im_out,imresize(temp,size(im_in)));
            end
        end
    case 'FeatureType'
        for iL = 1:nLevels-1
            [M,m,or,featType,pc,EO,T,pcSum] = phasecong3(I{iL}, ...
                5, 6, minW, 2.1, 0.55, 2, 0.5, 10, -1);
            temp = single((featType+pi/2)/pi);
            temp = imclose(temp,D);
            im_out = max(im_out,imresize(temp,size(im_in)));
        end
    case 'BowlerHat'
        for iL = 1:nLevels-1
            for iW = minW:2*minW+1
                [temp,~] = Granulo2D(I{iL},iW*3,6);
                temp = imclose(temp,D);
                im_out = max(im_out,imresize(temp,size(im_out)));
            end
        end
    case 'MFATl'
        for iL = 1:3
            % iL = 1;
            % Parameters setting
            sigmas = [1:1:3];
            spacing = .7; whiteondark = true;
            %                     tau = 0.05; tau2 = 0.25; D = 0.45;
            tau = 0.03; tau2 = 0.5; D = 0.01;
            % Proposed Method (Eign values based version)
            temp = FractionalIstropicTensor(I{iL},sigmas,tau,tau2,D,spacing,whiteondark);
            im_out = max(im_out,imresize(temp,size(im_in)));
        end
    case 'MFATp'
        for iL = 1:3
            % iL = 1;
            % Parameters setting
            sigmas = [1:1:3];
            
            spacing = .7; whiteondark = true;
            tau = 0.05; tau2 = 0.25; D = 0.45;
            % Proposed Method (Eign values based version)
            temp = ProbabiliticFractionalIstropicTensor(I{iL},sigmas,tau,tau2,D,spacing,whiteondark);
            im_out = max(im_out,imresize(temp,size(im_in)));
        end
end
% normalise
im_out = mat2gray(im_out);
end

function bw_out = fnc_im_to_bw(im_in,roi,roi_vein,roi_mask,T,method)
radius = 45;
se = strel('disk',radius);
nT = length(T);
bw_out = false([size(im_in) nT]);
switch method
    case {'Niblack';'midgrey';'Bernsen';'Sauvola'}
        T = T-0.5; % allow the offset to run from -0.5 to 0.5
end
switch method
    case 'Niblack'
        m = imfilter(im_in,fspecial('disk',radius)); % local mean
        s = stdfilt(im_in,getnhood(se)); % local std
        for iT = 1:nT
            level = m + T(iT) * s;
            bw_out(:,:,iT) = im_in > level;
        end
    case 'midgrey'
        lmin = imerode(im_in,se);
        lmax = imdilate(im_in,se);
        mg = (lmin + lmax)/2;
        for iT = 1:nT
            level =  mg - T(iT);
            bw_out(:,:,iT) = im_in > level;
        end
    case 'Bernsen'
        lmin = imerode(im_in,se);
        lmax = imdilate(im_in,se);
        lc = lmax - lmin; % local contrast
        mg = (lmin + lmax)/2; % mid grey
        for iT = 1:nT
            ix1 = lc < T(iT); % weight = contrast threshold in Bernsen algorithm
            ix2 = lc >= T(iT);
            temp = false(size(im_in));
            temp(ix1) = mg(ix1) >= 0.5;
            temp(ix2) = im_in(ix2) >= mg(ix2);
            bw_out(:,:,iT) = temp;
        end
    case 'Sauvola'
        % t = mean * ( 1 + k * ( stdev / r - 1 ) ) )
        m = imfilter(im_in,fspecial('disk',radius)); % local mean
        s = stdfilt(im_in,getnhood(se)); % local std
        R = max(s(:)); % 0.5
        for iT = 1:nT
            level = m .* (1.0 + T(iT) * (s / R - 1.0));
            bw_out(:,:,iT) = im_in > level;
        end
    case 'Phansalkar'
        % modification of Sauvola for low contrast images
        % t = mean * (1 + p * exp(-q * mean) + k * ((stdev / r) - 1))
        m = imfilter(im_in,fspecial('disk',radius)); % local mean
        s = stdfilt(im_in,getnhood(se)); % local std
        R = max(s(:));
        p = 2;
        q = 10;
        for iT = 1:nT
            level = m .* (1.0 + p.*exp(-q.*m) + T(iT) * (s / R - 1.0));
            bw_out(:,:,iT) = im_in > level;
        end
    case {'mean','median','Gaussian'}
        % adative threshold based on local mean, median or a gaussian
        % filter. Note Matlab internally rescales T to fit in the range 0.6 + (1-sensitivity)
        for iT = 1:nT
            % T controls the sensitivity
            level = adaptthresh(im_in, T(iT), 'NeighbourhoodSize',radius, 'Statistic',method);
            bw_out(:,:,iT) = im_in > level;
        end
    otherwise
        % hysteresis threshold using T as lower threshold and T+0.05 as
        % upper threshold
        for iT = 1:nT
            aboveT2 = im_in > T(iT); % Edge points above the lower threshold.
            Tmax = min(1,T(iT)+0.05);
            [aboveT1r, aboveT1c] = find(im_in > Tmax);  % Row and column coords of points above upper threshold.
            % Obtain all connected regions in aboveT2 that include a point that has a
            % value above T1
            bw_out(:,:,iT) = bwselect(aboveT2, aboveT1c, aboveT1r, 8);
        end
end
for iT = 1:nT
    % add in the manual large vein if present
    if ~isempty(roi_vein)
        bw_out(:,:,iT) = bw_out(:,:,iT) | roi_vein;
    end
    % apply the masks if present
    if ~isempty(roi_mask)
        temp = bw_out(:,:,iT);
        temp(~roi_mask) = 0;
        bw_out(:,:,iT)=temp;
    end
    if ~isempty(roi)
        temp = bw_out(:,:,iT);
        temp(~roi) = 0;
        bw_out(:,:,iT)=temp;
    end
    % keep the largest connected component
    %bw_out(:,:,iT) = bwareafilt(bw_out(:,:,iT),1);
end
end

function [sk_out] = fnc_im_to_sk(im_in,roi_vein,roi,T,method)
% calculate the skeleton
nT = length(T);
sk_out = false([size(im_in,1) size(im_in,2) nT]);
im_in = mat2gray(im_in);
if ~isempty(roi_vein)
    im_in = max(im_in,roi_vein);
end
switch method
    case {'CNN';'Vesselness';'FeatureType';'BowlerHat';'MFATl';'MFATp'}
        if ~isempty(roi)
            im_in(~roi) = 0;
        end
end
switch method
    case 'GT'
        % thin the binary image to a single pixel skeleton
        sk_out = bwmorph(im_in,'thin',inf);
        % fill in any single pixel holes
        sk_out = bwmorph(sk_out,'fill');
        % repeat to ensure a single pixel skeleton
        sk_out = bwmorph(sk_out,'thin',inf);
    otherwise
%     case {'CNN';'Vesselness';'FeatureType';'BowlerHat';'MFATl';'MFATp'}
%         % apply the threshold during the skeletonisation of the enhanced
%         % image
        for iT = 1:nT
            [~, sk_out(:,:,iT)] = fnc_skeleton(im_in(:,:,iT),roi_vein,T(iT));
        end
%     case {'Niblack';'midgrey';'Bernsen';'Sauvola'}
%         % use the binary image calculated at different threshold values
%         for iT = 1:nT
%             [~, sk_out(:,:,iT)] = fnc_skeleton(im_in(:,:,iT),roi_vein,T(iT));
%         end
end
end

function [PR_results,PR_im] = fnc_PRC_bw(roi_GT,bw_in,T)
[nY,nX,nT] = size(bw_in);
PR_im = zeros(nY,nX,3,nT,'single');
PR_results = array2table(zeros(nT,5,'single'));
PR_results.Properties.VariableNames = {'Threshold' 'Precision' 'Recall' 'F1' 'FBeta2'};
for iT = 1:size(bw_in,3)
    % calculate true positives present in GT and binary image
    TP = bw_in(:,:,iT) & roi_GT;
    % false positives are in the binary image; but not GT
    FP = bw_in(:,:,iT) & ~roi_GT;
    % false negatives are in the ground truth, but not the
    % binary image
    FN = roi_GT & ~bw_in(:,:,iT);
    % true negatives are all the pixels that are not in either
    TN = ~roi_GT & ~bw_in(:,:,iT);
    % sum the quantities across the whole image
    STP = sum(TP(:));
    SFP = sum(FP(:));
    STN = sum(TN(:));
    SFN = sum(FN(:));
    % precision is the number of true positives compared to true
    % plus false positive
    precision = STP/(STP+SFP);
    % recall is the number of true positive compared to the number
    % of true positive and false negatives. It is the same as the
    % true positive rate (TPR)
    recall = STP/(STP+SFN);
    % F1 statistic
    F = (2.*precision.*recall)./(precision + recall);
    % Fbeta2 statistic
    beta = 2;
    Fbeta2 = (1+beta^2).*((precision.*recall)./((beta^2.*precision)+recall));
    % results table
    PR_results{iT,:} = [T(iT) precision, recall, F, Fbeta2];
    temp = single(cat(3,FN,TP,FP));
    back = repmat(1-max(temp,[],3),1,1,3);% to give a white background
    PR_im(:,:,:,iT) = temp + back;
end
end

function [PR_results,PR_im] = fnc_PRC_sk(sk_GT,sk_in,tolerance,T)
[nY,nX,nT] = size(sk_in);
PR_im = zeros(nY,nX,3,nT,'single');
PR_results = array2table(zeros(nT,7,'single'));
PR_results.Properties.VariableNames = {'Threshold' 'Precision' 'Recall' 'F1' 'FBeta2' 'Distance_error' 'DE_sd'};
for iT = 1:size(sk_in,3)
    % get the distance away from the skeleton
    sk_distance = bwdist(sk_in(:,:,iT),'euclidean');
    % get the distance away from the ground_truth skeleton
    GT_distance = bwdist(sk_GT,'euclidean');
    % calculate whether the skeleton pixels are within the
    % tolerance limits for a true positive
    TP = sk_in(:,:,iT) & GT_distance<=tolerance;
    % false positives are in the skeleton but not within the
    % tolerance of the gold standard
    FP = sk_in(:,:,iT) & GT_distance>tolerance;
    % false negatives are in the gold standard, but not the
    % skeleton within the tolerance
    FN = sk_GT & sk_distance>tolerance;
    % true negatives are all the pixels that have not been assigned
    TN = ~(TP  | FP | FN);
    % the distance arror is distance of each pixel in the skeleton
    % away from the gold standard
    distance_error = mean(GT_distance(sk_in(:,:,iT)));
    distance_error_SD = std(GT_distance(sk_in(:,:,iT)));
    % sum the quantities across the whole image
    STP = sum(TP(:));
    SFP = sum(FP(:));
    STN = sum(TN(:));
    SFN = sum(FN(:));
    % precision is the number of true positives compared to true
    % plus false positive
    precision = STP/(STP+SFP);
    % recall is the number of true positive compared to the number
    % of true positive and false negatives. It is the same as the
    % true positive rate (TPR)
    recall = STP/(STP+SFN);
    % F1 statistic
    F = (2.*precision.*recall)./(precision + recall);
    % Fbeta2 statistic
    beta = 2;
    Fbeta2 = (1+beta^2).*((precision.*recall)./((beta^2.*precision)+recall));
    % results table
    PR_results{iT,:} = [T(iT) precision, recall, F, Fbeta2,distance_error,distance_error_SD];
    temp = single(cat(3,FN,TP,FP));
    back = repmat(1-max(temp,[],3),1,1,3);% to give a white background
    PR_im(:,:,:,iT) = temp + back;
end
end

function [bw_im, sk_out] = fnc_skeleton(im_in,bw_vein,threshold)
warning off
if islogical(im_in)
    % the input image is already a binary image
    bw_im = im_in;
else
    % impose local minima to smooth out background noise
    exmin = imextendedmin(mat2gray(im_in),0.2);
    im = imimposemin(mat2gray(im_in),exmin);
    % convert to a binary image
    bw_im = imbinarize(im,threshold);
end
% add in the big vein image if present
if ~isempty(bw_vein)
    bw_im = bw_im | bw_vein;
end
% smooth the binary image
bw_im = medfilt2(bw_im,[3 3]);
% thin the binary image to a single pixel skeleton
sk_out = bwmorph(im_in,'thin',inf);
% fill in any single pixel holes
sk_out = bwmorph(sk_out,'fill');
% repeat to ensure a single pixel skeleton
sk_out = bwmorph(sk_out,'thin',inf);
end

function results = fnc_skeleton_analysis(sk_in,roi_in,FolderName)
results = table;
ROIArea = sum(roi_in(:));
for iT = 1:size(sk_in,3)
    sk = sk_in(:,:,iT);
    bp = bwmorph(sk,'branchpoints');
    ep = bwmorph(sk,'endpoints');
    nodes = bwmorph(bp|ep,'thicken',1);
    edges = bwconncomp(sk&~nodes,8);
    results.Edges(iT) = edges.NumObjects;
    results.Length(iT) = sum(sum(sk));
    results.LengthDensity(iT) = sum(sum(sk))/ROIArea;
    results.Nodes(iT) = sum(bp(:)) + sum(ep(:));
    results.NodeDensity(iT) = results.Nodes(iT)/ROIArea;
    results.Junctions(iT) = sum(bp(:));
    results.FreeEnds(iT) = sum(ep(:));
    results.FreeEndRatio(iT) = results.FreeEnds(iT)/results.Edges(iT);
    areas = bwconncomp(imclearborder(~sk,4),4);
    stats = regionprops(areas,'Area','Eccentricity');
    results.AreaCount(iT) = areas.NumObjects;
    results.AreaMean(iT) = mean([stats.Area]);
    results.AreaEccentricity(iT) = mean([stats.Eccentricity]);
end
end

function hfig = fnc_display_fw_PR(PR_methods)
methods = PR_methods.name(PR_methods.select);
cols = {'r-';'g-';'b-';'c-';'m-';'g:';'b:';'c:'};
F1 = {'r*';'g*';'b*';'c*';'m*:';'g*';'b*';'c*'};
FBeta2 = {'ro';'go';'bo';'co';'mo:';'go';'bo';'co'};
mrks = {'.';'.';'.';'.';'.';'.';'.';'.'};
% plot the precision-recall plots and mark the optimum
hfig = figure('Renderer','painters');
hfig.Color = 'w';
for iM = 1:numel(methods)
    h(iM) = plot(PR_methods.results_fw.(methods{iM}).Recall,PR_methods.results_fw.(methods{iM}).Precision,cols{iM},'Marker',mrks{iM});
    hold on
    plot(PR_methods.results_fw.(methods{iM}).Recall(PR_methods.evaluation_fw{methods{iM},'F1_idx'}),PR_methods.results_fw.(methods{iM}).Precision(PR_methods.evaluation_fw{methods{iM},'F1_idx'}),F1{iM})
    plot(PR_methods.results_fw.(methods{iM}).Recall(PR_methods.evaluation_fw{methods{iM},'FBeta2_idx'}),PR_methods.results_fw.(methods{iM}).Precision(PR_methods.evaluation_fw{methods{iM},'FBeta2_idx'}),FBeta2{iM})
end
xlabel('Recall')
ylabel('Precision')
ax = gca;
ax.FontUnits = 'points';
ax.FontSize = 14;
legend(h,methods,'Location','SouthWest')
end

function hfig = fnc_display_sk_PR(PR_methods)
methods = PR_methods.name(PR_methods.select);
cols = {'r-';'g-';'b-';'c-';'m-';'g:';'b:';'c:'};
F1 = {'r*';'g*';'b*';'c*';'m*:';'g*';'b*';'c*'};
FBeta2 = {'ro';'go';'bo';'co';'mo:';'go';'bo';'co'};
mrks = {'.';'.';'.';'.';'.';'.';'.';'.'};
% plot the precision-recall plots and mark the optimum
hfig = figure('Renderer','painters');
hfig.Color = 'w';
for iM = 1:numel(methods)
    method = methods{iM};
    h(iM) = plot(PR_methods.results_sk.(method).Recall,PR_methods.results_sk.(method).Precision,cols{iM},'Marker',mrks{iM});
    hold on
    plot(PR_methods.results_sk.(method).Recall(PR_methods.evaluation_sk{method,'F1_idx'}),PR_methods.results_sk.(method).Precision(PR_methods.evaluation_sk{method,'F1_idx'}),F1{iM})
    plot(PR_methods.results_sk.(method).Recall(PR_methods.evaluation_sk{method,'FBeta2_idx'}),PR_methods.results_sk.(method).Precision(PR_methods.evaluation_sk{method,'FBeta2_idx'}),FBeta2{iM})
end
xlabel('Recall')
ylabel('Precision')
ax = gca;
ax.FontUnits = 'points';
ax.FontSize = 14;
legend(h,strrep(methods,'_sk',''),'Location','SouthWest')
end

function hfig = fnc_display_threshold_results(results,PR_methods)
methods = PR_methods.name(PR_methods.select);
cols = {'r';'g';'b';'c';'m';'g';'b';'c'};
F1 = {'r*';'g*';'b*';'c*';'m*:';'g*';'b*';'c*'};
FBeta2 = {'ro';'go';'bo';'co';'mo:';'go';'bo';'co'};
hfig = figure('Renderer','painters','Units','normalized','Position',[0 0 1 1]);
hfig.Color = 'w';
for iM = 1:numel(methods)
method = methods{iM};
    data = results.(method);
    GT = results.GT;
    names = GT.Properties.VariableNames;
    ratio = log(data{:,:}./GT{:,:});
        F1_idx_sk = PR_methods.evaluation_sk{method,'F1_idx'};
    FBeta2_idx_sk = PR_methods.evaluation_sk{method,'FBeta2_idx'};
    x = (1:size(ratio,1));
    plot_metrics = [1 2 3 4 5 6 7 8 9 10 11];
    for i = 1:numel(plot_metrics)
        ax = subplot(3,4,i);
        plot(x,ratio(:,plot_metrics(i)),'LineStyle','-','Marker','*','MarkerIndices',F1_idx_sk,'Color',cols{iM})
        hold on
        plot(x,ratio(:,plot_metrics(i)),'LineStyle','none','Marker','o','MarkerIndices',FBeta2_idx_sk,'Color',cols{iM})
        title(names{plot_metrics(i)})
        ylim([-1.2 1.2])
        hold on
        plot([1 20],[0 0],'k:')
        xlabel('Threshold')
        xticklabels = string((0:5:100)');
        ax.FontSize = 8;
        ax.Title.FontWeight = 'normal';
        ax.Title.FontUnits = 'points';
        ax.Title.FontSize = 10;
        if i == 1
            subplot(3,4,12)
            plot(x,ratio(:,plot_metrics(i)),'LineStyle','-','Marker','*','MarkerIndices',F1_idx_sk,'Color',cols{iM})
            hold on
            legend(methods,'Location','BestOutside')
        end
    end
end
end

function hfig = fnc_display_images(PR_methods)
        hfig = figure('Renderer','painters');
        offset = 0;
        % set up the axes to fill the figure
        for ia = 1:18
            ax(ia) = subplot(3,6,ia);
            axes(ax(ia))
            pos = ax(ia).OuterPosition;
            ax(ia).Position = pos;
            ax(ia).XTick = [];
            ax(ia).YTick = [];
        end
        hfig.Units = 'normalized';
        hfig.Position = [0 0 1 1];
        hfig.Color = 'w';
        %
        idx = PR_methods.select;
        methods = {PR_methods.name{idx}};
        % choose the orientation to be portrait
        axes(ax(1))
        if size(PR_methods.roi.im,1) < size(PR_methods.roi.im,2)
            rotate_angle = 90;
        else
            rotate_angle = 0;
        end
        % show the original image in inverse greyscale
        im_dis = imrotate(PR_methods.roi.im,rotate_angle);
        imshow(max(im_dis(:))-im_dis,[])
        title('original')
        axis off
        % display five methods
        for iP = 1:5
            n = iP+offset;
            axes(ax(iP+1))
            method = methods{iP};
            im_dis = imrotate(PR_methods.enhanced.(method),rotate_angle);
            imshow(im_dis,[])
            title(methods{n})
            axis off
        end
        % display the full-width ground-truth
        axes(ax(7))
        im_dis = imrotate(PR_methods.roi.GT,rotate_angle);
        imshow(~im_dis,[])
        % display the full width PR images
        axis off
        for iP = 1:5
            axes(ax(iP+7))
            method = methods{iP};
            n = iP+offset;
            imshow(imrotate(PR_methods.images_fw.(method)(:,:,:,PR_methods.evaluation_fw{method,'F1_idx'}),rotate_angle),[])
            axis off
        end
        % display the skeleton PR images
        % choose the thickness to dilate (original) or erode (complement) the skeleton to be visible
        width = 3;
        axes(ax(13))
        imshow(imerode(imrotate(~PR_methods.roi.sk.('GT'),rotate_angle), ones(width)),[])
        axis off
        for iP = 1:5
            axes(ax(iP+13))
            n = iP+offset;
            method = methods{iP};
            imshow(imerode(imrotate(PR_methods.images_sk.(method)(:,:,:,PR_methods.evaluation_sk{method,'F1_idx'}),rotate_angle),ones(width)),[])
            axis off
        end
        % tidy up the axes
        for ia = 1:18
            axes(ax(ia))
            ax(ia).XTick = [];
            ax(ia).YTick = [];
            axis off
            box on
            ax(ia).Title.FontWeight = 'normal';
            ax(ia).Title.FontUnits = 'points';
            ax(ia).Title.FontSize = 12;
            if ia == 1
                ax(ia).YLabel.String = 'enhanced image';
            elseif ia == 7
                ax(ia).YLabel.String = 'full-width binary';
            elseif ia == 13
                ax(ia).YLabel.String = 'skeleton';
            end
        end
    end



