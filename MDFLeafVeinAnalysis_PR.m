%function results = MDFLeafVeinAnalysis_v5(FolderName,micron_per_pixel,DownSample,ShowFigs,ExportFigs,FullLeaf,FullMetrics)
%% set up directories
dir_out_images = ['..' filesep 'summary' filesep 'images' filesep];
dir_out_width = ['..' filesep 'summary' filesep 'width' filesep];
dir_out_data = ['..' filesep 'summary' filesep 'data' filesep];
dir_out_HLD = ['..' filesep 'summary' filesep 'HLD' filesep];
dir_out_PR_results = ['..' filesep 'summary' filesep 'PR' filesep 'results' filesep];
dir_out_PR_images = ['..' filesep 'summary' filesep 'PR' filesep 'images' filesep];
dir_out_PR_graphs = ['..' filesep 'summary' filesep 'PR' filesep 'graphs' filesep];
dir_out_PR_fig = ['..' filesep 'summary' filesep 'PR' filesep 'fig' filesep];
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
%% test the performance against the internal ground truth
step = step+1;
disp(['Step ' num2str(step) ': Precision-Recall analysis'])
dirout = dir_out_PR_images;
% subsample images to match the area used for the ground truth
stats = regionprops(bw_roi,'BoundingBox');
BB = round(stats.BoundingBox);
roi_bw = bw_roi(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
roi_im = im(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
roi_cnn = im_cnn(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
roi_mask = bw_mask(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
roi_vein = bw_vein(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
roi_GT = bw_GT(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
% keep the largest connected component
roi_GT = bwareafilt(roi_GT,1);
% invert and normalise the original image
roi_invert = imcomplement(mat2gray(roi_im));
% set up threshold parameters
Tmin = 0;
Tmax = 1;
nT = 20;
Tint = (Tmax-Tmin)/(nT);
T = Tmin:Tint:Tmax-Tint;
% set up an array of methods
PR_methods.name = {'CNN';'midgrey';'Niblack';'Bernsen';'Sauvola';'Vesselness';'MFATl';'MFATp';'BowlerHat';'FeatureType'};
% choose which methods to test
PR_methods.select = [1 1 1 1 1 1 1 1 1 1];
% set up the evaluation table for the full width images
PR_methods.evaluation_fw = array2table(zeros(numel(PR_methods.name),6,'single'));
PR_methods.evaluation_fw.Properties.VariableNames = { ...
    'F1' 'F1_idx' 'F1_threshold' ...
    'FBeta2' 'FBeta2_idx' 'FBeta2_threshold'};
PR_methods.evaluation_fw.Properties.RowNames = PR_methods.name;
% set up an evaluation table for the skeleton comparison
PR_methods.evaluation_sk = array2table(zeros(numel(PR_methods.name),6,'single'));
PR_methods.evaluation_sk.Properties.VariableNames = { ...
    'F1' 'F1_idx' 'F1_threshold' ...
    'FBeta2' 'FBeta2_idx' 'FBeta2_threshold'};
PR_methods.evaluation_sk.Properties.RowNames = PR_methods.name;
n = 0;
% loop through each method
hfig2 = figure;
hfig3 = figure;
for iM = 1:sum(PR_methods.select)
    n = n+1;
    idx = find(PR_methods.select);
    method = PR_methods.name{idx(iM)};
    % normalise and enhance the images with variable thresholds
    switch method
        case {'CNN'}
            % normalise the CNN image
            PR_methods.enhanced{idx(iM)} = (roi_cnn./255);
        case {'Vesselness';'FeatureType';'BowlerHat';'MFATl';'MFATp'}
            % calculate the enhanced images
            PR_methods.enhanced{idx(iM)} = fnc_enhance_im(roi_invert,DownSample,method);
        otherwise
            % copy the original image across for local intensity
            % segmentation
            PR_methods.enhanced{idx(iM)} = fnc_enhance_im(roi_im,DownSample,method);
    end
    % extract the full width binary image
    switch method
        case {'CNN';'Vesselness';'FeatureType';'BowlerHat';'MFATl';'MFATp'}
            % use a hysteresis thresholf for the enhanced images
            PR_methods.fullwidth{idx(iM)} = fnc_im_to_bw(PR_methods.enhanced{idx(iM)},roi_bw,roi_vein,roi_mask,T,'hysteresis');
        case {'Niblack';'midgrey';'Bernsen';'Sauvola'}
            % apply the local thresholding to the other images
            PR_methods.fullwidth{idx(iM)} = fnc_im_to_bw(PR_methods.enhanced{idx(iM)},roi_bw,roi_vein,roi_mask,T,method);
        otherwise
            PR_methods.fullwidth{idx(iM)} = PR_methods.enhanced{idx(iM)};
    end
    
    if ShowFigs == 1
        % display the segmentation
        figure
        for i = 1:20
            subplot(3,7,i)
            imshow(imrotate(PR_methods.fullwidth{idx(iM)}(:,:,i),90))
        end
    end
    if ShowFigs == 1
        % display the segmentation as a height coded image
        figure(hfig2)
        subplot(2,5,iM)
        if sum(sum(PR_methods.fullwidth{idx(iM)}(:,:,1)))>sum(sum(PR_methods.fullwidth{idx(iM)}(:,:,end)))
            [mx,hgt] = max(flip(PR_methods.fullwidth{idx(iM)},3),[],3);
        else
            [mx,hgt] = max(PR_methods.fullwidth{idx(iM)},[],3);
        end
        hgt(mx==0) = 0;
        cmap = jet(21);
        cmap(1,:) = 0;
        imshow(imrotate(uint8(hgt),90),cmap)
    end
    % evaluate performance against the ground truth using Precision-Recall analysis
    [PR_methods.results_fw{idx(iM)}, PR_methods.images_fw{idx(iM)}] = fnc_PRC_bw(roi_GT,PR_methods.fullwidth{idx(iM)},T);
    % get the best values from the results table
    [PR_methods.evaluation_fw{method,'F1'}, PR_methods.evaluation_fw{method,'F1_idx'}] = max(PR_methods.results_fw{idx(iM)}{:,'F1'});
    PR_methods.evaluation_fw{method,'F1_threshold'} = T(PR_methods.evaluation_fw{method,'F1_idx'});
    [PR_methods.evaluation_fw{method,'FBeta2'}, PR_methods.evaluation_fw{method,'FBeta2_idx'}] = max(PR_methods.results_fw{idx(iM)}{:,'FBeta2'});
    PR_methods.evaluation_fw{method,'FBeta2_threshold'} = T(PR_methods.evaluation_fw{method,'FBeta2_idx'});
    % keep the best full width image
    best_idx_fw = PR_methods.evaluation_fw{method,'F1_idx'};
    PR_methods.best_fw{idx(iM)} = PR_methods.images_fw{idx(iM)}(:,:,best_idx_fw);
    % convert to a skeleton
    if n == 1
        % get the ground truth skeleton
        [sk_GT, skLoop_GT, skTree_GT] = fnc_im_to_sk(roi_GT,roi_vein,roi_bw,T,'GT');
    end
    % get the processed skeleton for all threshold values
    [PR_methods.sk{idx(iM)}, PR_methods.Loop{idx(iM)}, PR_methods.Tree{idx(iM)}] = fnc_im_to_sk(PR_methods.fullwidth{idx(iM)},roi_vein,roi_bw,T,method);
    % compare the skeletons with the skeleton ground truth within a given tolerance (in pixels)
    tolerance = 3;
    [PR_methods.results_sk{idx(iM)},PR_methods.images_sk{idx(iM)}] = fnc_PRC_sk(sk_GT,PR_methods.sk{idx(iM)},tolerance,T);
    % get the best values from the results table
    [PR_methods.evaluation_sk{method,'F1'}, PR_methods.evaluation_sk{method,'F1_idx'}] = max(PR_methods.results_sk{idx(iM)}{:,'F1'});
    PR_methods.evaluation_sk{method,'F1_threshold'} = T(PR_methods.evaluation_sk{method,'F1_idx'});
    [PR_methods.evaluation_sk{method,'FBeta2'}, PR_methods.evaluation_sk{method,'FBeta2_idx'}] = max(PR_methods.results_sk{idx(iM)}{:,'FBeta2'});
    PR_methods.evaluation_sk{method,'FBeta2_threshold'} = T(PR_methods.evaluation_sk{method,'FBeta2_idx'});
    % keep the best skeleton
    best_idx_sk = PR_methods.evaluation_sk{method,'F1_idx'};
    PR_methods.skFull{idx(iM)} = PR_methods.sk{idx(iM)}(:,:,best_idx_sk);
    PR_methods.skLoop{idx(iM)} = PR_methods.Loop{idx(iM)}(:,:,best_idx_sk);
    PR_methods.skTree{idx(iM)} = PR_methods.Tree{idx(iM)}(:,:,best_idx_sk);
    % calculate the width from the distance transform of the binarized fullwidth image
    [im_distance, ~] = bwdist(~PR_methods.fullwidth{idx(iM)}(:,:,best_idx_sk),'Euclidean');
    % extract the initial width along the best skeleton from the distance transform
    W_pixels = zeros(size(im_distance),'single');
    W_pixels(PR_methods.skFull{idx(iM)}) = im_distance(PR_methods.skFull{idx(iM)}).*2;
    PR_methods.width{idx(iM)} = W_pixels;
    % get the graph representation and results for the ground truth
    if n ==1
        methods = {'GT'};
        results = fnc_PR_graph(sk_GT,roi_im,roi_cnn,PR_methods.enhanced{1},roi_mask,skLoop_GT,skTree_GT,0,1);
    end
    % get the graph representation for the optimum threshold from the
    % PR analysis
    methods = [methods;method];
    temp = fnc_PR_graph(PR_methods.skFull{idx(iM)},roi_im,PR_methods.fullwidth{idx(iM)},PR_methods.enhanced{idx(iM)},roi_mask,PR_methods.skLoop{idx(iM)},PR_methods.skTree{idx(iM)},0,1);
    results = [results; temp];
    % calculate the network metrics for the skeleton for each threshold
    results_full = [];
    for iT = 1:size(PR_methods.sk{idx(iM)},3)
        if n == 1 && iT == 1
            % start with the ground truth
            GT_full = fnc_PR_graph(sk_GT, roi_im, roi_bw, roi_GT, roi_mask, skLoop_GT, skTree_GT, FullMetrics, micron_per_pixel);
        end
        % check that there is a skeleton present to analyse at this
        % threshold
        if sum(any(any(PR_methods.sk{idx(iM)}(:,:,iT))))>0
            try
                % get the appropriate images for this threshold
                sk = PR_methods.sk{idx(iM)}(:,:,iT);
                im_enhanced = PR_methods.enhanced{idx(iM)};
                bw_enhanced = PR_methods.fullwidth{idx(iM)}(:,:,iT);
                bw_mask = roi_mask;
                skLoop = PR_methods.Loop{idx(iM)}(:,:,iT);
                skTree = PR_methods.Tree{idx(iM)}(:,:,iT);
                % calculate the graph metrics for this threshold
                temp1 = fnc_PR_graph(sk, roi_im, im_enhanced, bw_enhanced, bw_mask, skLoop, skTree, FullMetrics, micron_per_pixel);
                % make a full table entry and set values to zero
                temp2 = GT_full;
                temp2{1,1:end} = nan;
                % update measured values even if there are no enclosed polygons
                temp2{1,1:size(temp1,2)} = temp1{1,:};
                % save the threshold step
                temp2.thresh_idx = iT;
                % update the results table
                results_full = [results_full; temp2];
            catch
                'catch'
                % set up a dummy entry
                temp2 = GT_full;
                temp2{1,1:end} = nan;
                % save the threshold step
                temp2.thresh_idx = iT;
                results_full = [results_full; temp2];
            end
        end
    end
    results_thresholds{idx(iM)} = results_full;
    % compare the results as a (O-E)^2/E against the ground truth
    O = results_full{:,1:end-1};
    E = GT_full{1,:};
    chi = ((O-E).^2)./E;
    results_chi{idx(iM)} = chi;
    % compare as a log ratio against the ground truth
    ratio = (O./E);
    results_ratio{idx(iM)} = ratio;
    % find the best performance and corresponding threshold for each
    % metric
    % plot the results for selected variables
    names = GT_full.Properties.VariableNames;
    x = (1:size(ratio,1));
    plot_metrics = [1 2 3 4 6 13 20 25 27 28];
    figure(hfig3)
    for i = 1:numel(plot_metrics)
        subplot(3,4,i)
        plot(x,ratio(:,plot_metrics(i)),'-*','MarkerIndices',best_idx_sk)
        title(names{plot_metrics(i)})
        ylim([-1 1])
        hold on
        plot([1 20],[1 1],'k:')
        if i == 1
            subplot(3,4,11)
            plot(x,ratio(:,plot_metrics(i)),'-*','MarkerIndices',best_idx_sk)
            title(names{plot_metrics(i)})
            ylim([-1 1])
            legend(methods{2:end})
        end
    end
    
end
% concatenate the methods and the results for the best skeleton
results.Method = methods;
% re-order so the methods appears in the first column
results = results(:,[end 1:end-1]);
%% display the figure
hfig = figure('Renderer','painters');
offset = 0;
% set up the axes to fill the figure
for ia = 1:33
    ax(ia) = subplot(3,11,ia);
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
idx = find(PR_methods.select);
methods = {PR_methods.name{idx}};

%methods = {'CNN';'Vesselness';'FeatureType';'BowlerHat';'midgrey'};
% choose the orientation to be portrait
axes(ax(1))
if size(roi_im,1) > size(roi_im,2)
    rotate_angle = 90;
else
    rotate_angle = 0;
end
% show the original image in inverse greyscale
im_dis = imrotate(roi_im,rotate_angle);
imshow(max(im_dis(:))-im_dis,[])
title('original')
axis off
%export_fig([dirout 'roi_im'],'-png',ax(1))

for iP = 1:10
    n = iP+offset;
    axes(ax(iP+1))
    im_dis = imrotate(PR_methods.enhanced{idx(n)},rotate_angle);
    %imshow(max(im(:))-im,[])
    imshow(im_dis,[])
    title(methods{n})
    axis off
    %     export_fig([dirout methods{iP}],'-png',ax(iP+1))
end
% % show the midgrey images
% axes(ax(6))
% im = imrotate(bw_midgrey(:,:,PR.evaluation{'midgrey','F1_idx'}),rotate_angle);
% imshow(max(im(:))-im,[])
% title('Mid Grey')
% axis off
% export_fig([dirout 'midgrey'],'-png',ax(6))
% display the full-width ground-truth
axes(ax(12))
im_dis = imrotate(roi_GT,rotate_angle);
imshow(~im_dis,[])
% display the full width PR images
axis off
% export_fig([dirout 'GT'],'-png',ax(7))
for iP = 1:10
    axes(ax(iP+12))
    n = iP+offset;
    imshow(imrotate(PR_methods.images_fw{idx(n)}(:,:,:,PR_methods.evaluation_fw{methods{n},'F1_idx'}),rotate_angle),[])
    axis off
    %     export_fig([dirout methods{iP} '_2'],'-png',ax(iP+7))
end
% display the skeleton PR images
% choose the thickness to dilate (original) or erode (complement) the skeleton to be visible
width = 3;
% methods = {'cnn_sk';'vesselness_sk';'featuretype_sk';'bowlerhat_sk';'midgrey_sk'};
axes(ax(23))
imshow(imerode(imrotate(~sk_GT,rotate_angle), ones(width)),[])
axis off
% export_fig([dirout 'sk_GT'],'-png',ax(13))
for iP = 1:10
    axes(ax(iP+23))
    n = iP+offset;
    imshow(imerode(imrotate(PR_methods.images_sk{idx(n)}(:,:,:,PR_methods.evaluation_sk{methods{n},'F1_idx'}),rotate_angle),ones(width)),[])
    axis off
    %     export_fig([dirout methods{iP}],'-png',ax(iP+13))
end
% tidy up the axes
for ia = 1:33
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

% save as an image
% export_fig([dirout 'all'],'-native','-png')
% linkaxes
% drawnow
% saveas(hfig,[fout '_PRall.fig'])
% delete(hfig)
% end
return
%     PR = fnc_precision_recall(im,im_cnn,bw_mask,bw_vein,bw_roi,bw_GT,DownSample,dirout);
%     if ShowFigs == 1
%         % display the PR curves for the full-width binary image and the
%         % skeleton
%         display_PR(PR);
%         display_sk_PR(PR);
%     end
%     if ExportFigs == 1
%         step = step+1;
%         disp(['Step ' num2str(step) ': Saving PR data'])
%         % save the results
%         save([dir_out_PR_results FolderName '_PR.mat'],'PR')
%         % save the color-coded full-width PR image
%         fout = [dir_out_PR_images FolderName '_PR.png'];
%         imwrite(PR.images.cnn(:,:,:,PR.evaluation{'cnn','F1_idx'}),fout,'png')
%         % save the color-coded PR skeleton image
%         fout = [dir_out_PR_images FolderName '_sk_PR.png'];
%         imwrite(PR.images.cnn_sk(:,:,:,PR.evaluation{'cnn_sk','F1_idx'}),fout,'png')
%         % save the full-width PR curve
%         hfig1 = display_PR(PR);
%         saveas(hfig1,[dir_out_PR_graphs FolderName '_PRC.png'])
%         delete(hfig1);
%         % save the PR skeleton curve
%         hfig2 = display_sk_PR(PR);
%         saveas(hfig2,[dir_out_PR_graphs FolderName '_sk_PRC.png'])
%         delete(hfig2);
%     end
% else
%     PR = [];
% end
% %% get the skeleton
% step = step+1;
% disp(['Step ' num2str(step) ': Skeleton extraction'])
% if exist('PR','var') && ~isempty(PR)
%     % use the optimum threshold value determined from the PR analysis
%     [bw_cnn, sk, skLoop, skTree] = fnc_skeleton(im_cnn,bw_vein,PR.evaluation{'cnn','F1_threshold'});
% else
%     % use a standard threshold
%     [bw_cnn, sk, skLoop, skTree] = fnc_skeleton(im_cnn,bw_vein,0.45);
% end
% %% calculate the width from the distance transform of the binarized cnn image
% step = step+1;
% disp(['Step ' num2str(step) ': Calculating width from distance transform'])
% [im_distance, ~] = bwdist(~bw_cnn,'Euclidean');
% % extract the initial width along the skeleton from the distance transform
% W_pixels = zeros(size(im_distance),'single');
% W_pixels(sk) = single(im_distance(sk).*2);
% %% extract network using the thinned skeleton
% step = step+1;
% disp(['Step ' num2str(step) ': Extracting the network'])
% % collect a cell array of connected edge pixels
% [edgelist, edgeim] = edgelink(sk);
% %% find any self loops and split them
% step = step+1;
% disp(['Step ' num2str(step) ': Resolving self loops'])
% [edgelist, edgeim] = fnc_resolve_self_loops(edgelist,edgeim);
% %% resolve duplicate edges by splitting one edge into two
% step = step+1;
% disp(['Step ' num2str(step) ': Resolving duplicates'])
% [edgelist, ~] = fnc_resolve_duplicates(edgelist,edgeim);
% %% construct the weighted graph
% step = step+1;
% disp(['Step ' num2str(step) ': Weighted graph'])
% [G_veins,edgelist] = fnc_weighted_graph(edgelist,W_pixels,skTree);
% %% Refine the width
% step = step+1;
% disp(['Step ' num2str(step) ': Refining width'])
% [G_veins,edgelist_center] = fnc_refine_width(G_veins,edgelist,im,im_cnn,W_pixels);
% %% calculate a pixel skeleton for the center weighted edges
% step = step+1;
% disp(['Step ' num2str(step) ': Colour-coded skeleton'])
% [CW_pixels,im_width,coded] = fnc_coded_skeleton(im,sk,bw_mask,G_veins,edgelist,edgelist_center,sk_width,cmap);
% %% display the weighted network
% step = step+1;
% disp(['Step ' num2str(step) ': Image display'])
% if ExportFigs == 1
%     step = step+1;
%     disp(['Step ' num2str(step) ': Saving width images'])
%     [nY,nX,~] = size(coded);
%     % save the color-coded width image
%     fout = [dir_out_images FolderName '_width.png'];
%     imwrite(coded,fout,'png','Xresolution',nX,'Yresolution',nY)
%     % save the greyscale width array as a matlab file. Note outside the
%     % masked area is now coded as -1
%     save([dir_out_width FolderName '_Width_array'],'im_width')
% end
% %% find the areoles
% step = step+1;
% disp(['Step ' num2str(step) ': polygon analysis'])
% % find the polygon and areole areas
% [G_veins, sk_polygon, bw_polygons, bw_areoles, total_area_mask, polygon_LM] = fnc_polygon_find(G_veins,bw_cnn,sk,skLoop,bw_mask);
% [areole_stats,polygon_stats] = fnc_polygon_analysis(bw_polygons,bw_areoles, polygon_LM,FullMetrics);
% % construct color-coded image based on log area for display
% im_areoles_rgb = fnc_polygon_image(areole_stats, sk_polygon, total_area_mask);
% % im_polygons_rgb = fnc_polygon_image(polygon_stats, sk_polygon, total_area_mask);
% %% convert to an areole graph and a polygon graph
% step = step+1;
% disp(['Step ' num2str(step) ': Dual graph'])
% [G_areoles] = fnc_area_graph(G_veins,areole_stats);
% [G_polygons] = fnc_area_graph(G_veins,polygon_stats);
% %% collect summary statistics into a results array
% step = step+1;
% disp(['Step ' num2str(step) ': Summary statistics'])
% total_area = sum(bw_mask(:));
% polygon_area = sum(G_polygons.Nodes.Area);
% veins = fnc_summary_veins(G_veins,total_area,polygon_area,micron_per_pixel);
% areoles = fnc_summary_areoles(G_areoles,polygon_area,micron_per_pixel,FullMetrics);
% polygons = fnc_summary_polygons(G_polygons,micron_per_pixel,FullMetrics);
% results = [veins areoles polygons];
%% add in results for the ROC analysis (if present)
if exist('PR','var') && ~isempty(PR)
    results.PR_threshold = PR.evaluation{'cnn_sk','F1_threshold'};
    results.PR_Precision = PR.results.cnn.Precision(PR.evaluation{'cnn_sk','F1_idx'});
    results.PR_Recall = PR.results.cnn.Recall(PR.evaluation{'cnn_sk','F1_idx'});
    results.PR_F = PR.evaluation{'cnn_sk','F1'};
    results.PR_FBeta2 = PR.evaluation{'cnn_sk','FBeta2'};
end
%% add in file information
results.File = FolderName;
results.TimeStamp = datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss Z');
results.MicronPerPixel = micron_per_pixel;
results.DownSample = DownSample;
% reorder the table to get the file info first
results = results(:, [end-3:end 1:end-4]);
%% save the graphs to matlab
step = step+1;
disp(['Step ' num2str(step) ': Saving graphs data'])
% save the results
save([dir_out_data FolderName '_Graphs.mat'],'G_veins','G_areoles','G_polygons','results')

%% set up the results figure
if ShowFigs == 1
    warning off images:initSize:adjustingMag
    warning off MATLAB:LargeImage
    skel = imerode(single(cat(3,1-skTree,1-skLoop,1-skTree)), ones(5));
    images = {im,max(im_cnn(:))-im_cnn,skel,coded,im_areoles_rgb,max(im_cnn(:))-im_cnn};
    graphs = {'none','none','none','none','none','Width'};
    titles = {'original','CNN','Skeleton','width','areoles','dual graph'};
    display_figure(images,graphs,titles,G_polygons,E_width,[1:6],[dir_out_images FolderName '_Figure'],ExportFigs);
end
%% Hierarchical loop decomposition
step = step+1;
disp(['Step ' num2str(step) ': Hierarchical loop decomposition'])
[G_HLD, parent] = fnc_HLD(G_veins, G_polygons, polygon_stats, areole_stats, polygon_LM, bw_polygons, micron_per_pixel);
save([dir_out_HLD FolderName '_HLD_results.mat'],'G_HLD','parent')
%% HLD display
if ShowFigs == 1 && ExportFigs == 0
    display_HLD(G_polygons,im_cnn,G_HLD,FullLeaf,[dir_out_HLD FolderName '_HLD'],ExportFigs);
end
if ExportFigs == 1
    display_HLD(G_polygons,im_cnn,G_HLD,FullLeaf,[dir_out_HLD FolderName '_HLD'],ExportFigs);
    %     display_HLD_figure(G_polygons,im_cnn,G_HLD,FullLeaf,[dir_out_HLD FolderName '_HLD_2'],ExportFigs);
end
%% save results to Excel
step = step+1;
disp(['Step ' num2str(step) ': saving results to Excel'])
% save as excel spreadsheets
warning('off','MATLAB:xlswrite:AddSheet')
writetable(G_veins.Edges,[dir_out_data FolderName '_results.xlsx'],'FileType','spreadsheet','WriteVariableNames',1,'Sheet','Vein Edges')
writetable(G_veins.Nodes,[dir_out_data FolderName '_results.xlsx'],'FileType','spreadsheet','WriteVariableNames',1,'Sheet','Vein Nodes')
writetable(G_areoles.Edges,[dir_out_data FolderName '_results.xlsx'],'FileType','spreadsheet','WriteVariableNames',1,'Sheet','Areole Edges')
writetable(G_areoles.Nodes,[dir_out_data FolderName '_results.xlsx'],'FileType','spreadsheet','WriteVariableNames',1,'Sheet','Areole Nodes')
writetable(G_polygons.Edges,[dir_out_data FolderName '_results.xlsx'],'FileType','spreadsheet','WriteVariableNames',1,'Sheet','Polygon Edges')
writetable(G_polygons.Nodes,[dir_out_data FolderName '_results.xlsx'],'FileType','spreadsheet','WriteVariableNames',1,'Sheet','Polygon Nodes')
writetable(G_HLD.Edges,[dir_out_data FolderName '_results.xlsx'],'FileType','spreadsheet','WriteVariableNames',1,'Sheet','HLD Edges')
writetable(G_HLD.Nodes,[dir_out_data FolderName '_results.xlsx'],'FileType','spreadsheet','WriteVariableNames',1,'Sheet','HLD Nodes')
% remove the unnecessary default sheets. Note this requires the full path.
dir_current = pwd;
cd(dir_out_data);
dir_in = pwd;
xls_delete_sheets([dir_in filesep FolderName '_results.xlsx'],{'Sheet1','Sheet2','Sheet3'})
cd(dir_current);
%end

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
        nLevels = 3;
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
        for iL = 1;%:2
            % iL = 1;
            % Parameters setting
            sigmas = [1:1:5];
            spacing = .7; whiteondark = true;
            %                     tau = 0.05; tau2 = 0.25; D = 0.45;
            tau = 0.03; tau2 = 0.5; D = 0.01;
            % Proposed Method (Eign values based version)
            temp = FractionalIstropicTensor(I{iL},sigmas,tau,tau2,D,spacing,whiteondark);
            im_out = max(im_out,imresize(temp,size(im_in)));
        end
    case 'MFATp'
        for iL = 1;%:2
            % iL = 1;
            % Parameters setting
            sigmas = [1:1:5];
            spacing = .7; whiteondark = true;
            tau = 0.05; tau2 = 0.25; D = 0.45;
            % Proposed Method (probability based version)
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
    bw_out(:,:,iT) = bwareafilt(bw_out(:,:,iT),1);
end
end

function [sk_out, sk_Loop, sk_Tree] = fnc_im_to_sk(im_in,roi_vein,roi,T,method)
% calculate the skeleton
nT = length(T);
sk_out = false([size(im_in,1) size(im_in,2) nT]);
sk_Loop = false([size(im_in,1) size(im_in,2) nT]);
sk_Tree = false([size(im_in,1) size(im_in,2) nT]);
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
        % only keep the largest connected component
        sk_out = bwareafilt(sk_out,1);
        sk_Loop = bwmorph(sk_out,'spur','inf');
        sk_Tree = sk_out & ~sk_Loop;
    case {'CNN';'Vesselness';'FeatureType';'BowlerHat';'MFATl';'MFATp'}
        % apply the threshold during the skeletonisation of the enhanced
        % image
        for iT = 1:nT
            [~, sk_out(:,:,iT),sk_Loop(:,:,iT),sk_Tree(:,:,iT)] = fnc_skeleton(im_in(:,:,iT),roi_vein,T(iT));
        end
    case {'Niblack';'midgrey';'Bernsen';'Sauvola'}
        % use the binary image calculated at different threshold values
        for iT = 1:nT
            [~, sk_out(:,:,iT),sk_Loop(:,:,iT),sk_Tree(:,:,iT)] = fnc_skeleton(im_in(:,:,iT),roi_vein,T(iT));
        end
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

function evaluation = fnc_PRC_evaluation(PR_methods,thresholds)
methods = PR_methods.name;
nM = numel(methods);
evaluation = array2table(zeros(nM,6,'single'));
evaluation.Properties.VariableNames = { ...
    'F1' 'F1_idx' 'F1_threshold' ...
    'FBeta2' 'FBeta2_idx' 'FBeta2_threshold'};
evaluation.Properties.RowNames = methods;
for iM = 1:nM
    method = methods{iM};
    switch method
        case {'Niblack';'midgrey';'Bernsen';'Sauvola'}
            T = thresholds-0.5; % allow the offset to run from -0.5 to 0.5
        otherwise
            T = thresholds;
    end
    
    [evaluation{iM,'F1'}, evaluation{iM,'F1_idx'}] = max(PR_methods.results{iM}{:,'F1'});
    evaluation{iM,'F1_threshold'} = T(evaluation{iM,'F1_idx'});
    [evaluation{iM,'FBeta2'}, evaluation{iM,'FBeta2_idx'}] = max(PR_methods.results{iM}{:,'FBeta2'});
    evaluation{iM,'FBeta2_threshold'} = T(evaluation{iM,'FBeta2_idx'});
end
end

function results = fnc_PR_graph(sk,im,im_cnn,bw_cnn,bw_mask,skLoop,skTree,FullMetrics,micron_per_pixel)
%% calculate the width from the distance transform of the binarized cnn image
[im_distance, ~] = bwdist(~bw_cnn,'Euclidean');
% extract the initial width along the skeleton from the distance transform
W_pixels = zeros(size(im_distance),'single');
W_pixels(sk) = single(im_distance(sk).*2);
%% extract network using the thinned skeleton
% collect a cell array of connected edge pixels
[edgelist, edgeim] = edgelink(sk);
if numel(edgelist)>1
    %% find any self loops and split them
    [edgelist, edgeim] = fnc_resolve_self_loops(edgelist,edgeim);
    %% resolve duplicate edges by splitting one edge into two
    [edgelist, ~] = fnc_resolve_duplicates(edgelist,edgeim);
    %% construct the weighted graph
    [G_veins,edgelist] = fnc_weighted_graph(edgelist,W_pixels,skTree);
    %% Refine the width
    [G_veins,edgelist_center] = fnc_refine_width(G_veins,edgelist,im,im_cnn,W_pixels);
    %% find the areoles
    % find the polygon and areole areas
    [G_veins, sk_polygon, bw_polygons, bw_areoles, total_area_mask, polygon_LM] = fnc_polygon_find(G_veins,bw_cnn,sk,skLoop,bw_mask);
    [areole_stats,polygon_stats] = fnc_polygon_analysis(bw_polygons,bw_areoles, polygon_LM,FullMetrics);
    %% convert to an areole graph and a polygon graph
    % [G_areoles] = fnc_area_graph(G_veins,areole_stats);
    [G_polygons] = fnc_area_graph(G_veins,polygon_stats);
    %% collect summary statistics into a results array
    total_area = sum(bw_mask(:));
    if ~isempty(G_veins)
        veins = fnc_summary_veins(G_veins,total_area,micron_per_pixel);
        % areoles = fnc_summary_areoles(G_areoles,micron_per_pixel,FullMetrics);
        if ~isempty(G_polygons)
            polygons = fnc_summary_polygons(G_polygons,micron_per_pixel,FullMetrics);
        else
            polygons = [];
        end
        results = [veins polygons];
    end
end
end

function [bw_cnn, skfinal, skLoop, skTree] = fnc_skeleton(im_in,bw_vein,threshold)
warning off
if islogical(im_in)
    % the input image is already a binary image
    bw_cnn = im_in;
else
    % impose local minima to smooth out background noise
    exmin = imextendedmin(mat2gray(im_in),0.2);
    im = imimposemin(mat2gray(im_in),exmin);
    % convert to a binary image
    bw_cnn = imbinarize(im,threshold);
end
% add in the big vein image if present
if ~isempty(bw_vein)
    bw_cnn = bw_cnn | bw_vein;
end
% smooth the binary image
bw_cnn = medfilt2(bw_cnn,[3 3]);
% keep the connected component
bw_cnn = bwareafilt(bw_cnn,1);
% calculate the distance transform as the input to a watershed segmentation
D = bwdist(~bw_cnn,'Euclidean');
W = watershed(D,4);
% get the watershed skeleton comprising only loops
skLoop = W == 0;
% thin to a single pixel skeleton
skLoop = bwmorph(skLoop,'thin',Inf);
% find any isolated loops
skRing = skLoop & ~bwareafilt(skLoop,1);
if any(skRing(:))
    % fill the loops
    skLoopFill = imfill(skRing,'holes');
    % punch te loop skeleton back out. This ensure two touching loops are
    % treated separately
    skLoopFill = skLoopFill & ~skRing;
    % erode to a single point in the middle of the loop
    skLoopPoints = bwulterode(skLoopFill);
    % use these points to fill the loop in the binary image
    bw_cnn_fill = imfill(bw_cnn,find(skLoopPoints));
    % thin the binary image to a single pixel skeleton
    sk = bwmorph(bw_cnn_fill,'thin',Inf);
else
    sk =  bwmorph(bw_cnn,'thin',Inf);
end
% only keep the largest connected component
sk = bwareafilt(sk,1);
% find tree regions in the thinned skeleton as the difference after spur
% removal
skSp = bwmorph(sk,'spur',Inf);
skTree = xor(sk,skSp);
% remove single pixel spurs now present as isolated points
skTree = bwmorph(skTree,'clean');
if any(skRing(:))
    % punch out the isolated loops
    skTree = skTree & ~skLoopFill;
    % add back in the watershed loop
    skTree = skTree | skRing;
end
% get the endpoints of the thinned skeleton, representing the free end
% veins
epsk = bwmorph(sk,'endpoints');
% find the endpoints at the root of the tree
epskTree = bwmorph(skTree,'endpoints');
epskTree = xor(epsk,epskTree);
% only work with endpoints that are not already connected to
% the watershed skeleton
connected = bwareafilt(epskTree | skLoop,1);
epskTree = epskTree & ~connected;
% get the feature map from the watershed skeleton to find the nearest pixel
% to connect to
[~,skW_idx] = bwdist(skLoop);
% get the pixel co-ordinates for the nearest point on the skeleton
[y1,x1] = ind2sub(size(skRing),skW_idx(epskTree));
% get the pixel co-ordinates for the endpoints
[y2,x2] = find(epskTree);
% use the bresenham algorithm to find the best pixel line
[x,y] = arrayfun(@(x1,y1,x2,y2) bresenham(x1,y1,x2,y2),x1,y1,x2,y2,'uniformoutput',0);
% get the index of the line points
P = cellfun(@(x,y) sub2ind(size(skLoop),y,x),x,y,'UniformOutput',0);
% add the lines into the tree skeleton
skTree(cat(1,P{:})) = 1;
if any(skRing(:))
    % remove any parts of the skeleton overlapping the loops
    skTree(skLoopFill) = 0;
    % remove the rings from the loop skeleton as they will be included in
    % the tree skeleton
    skLoop(skRing) = 0;
end
% add the watershed skeleton back in
skfinal = skTree | skLoop;
% keep the largest connected component
skfinal = bwareafilt(skfinal,1);
end

function [edgelist,edgeim] = fnc_resolve_self_loops(edgelist,edgeim)
D_idx = 1;
[nY,nX] = size(edgeim);
while ~isempty(D_idx)
    % F_idx is the linear index to the first pixel in each edge array
    F_idx = cellfun(@(x) sub2ind([nY,nX],x(1,1),x(1,2)),edgelist);
    % L_idx is the linear index to the last pixel in each edge array
    L_idx = cellfun(@(x) sub2ind([nY,nX],x(end,1),x(end,2)),edgelist);
    % check if nodei and nodej are equal
    D_idx = find(F_idx==L_idx);
    % See if there are any loops
    if ~isempty(D_idx)
        % Go through each loop and split it in two if it is long enough,
        % adding an extra edge at the end of the edgelist. If the duplicate
        % edge is less than 10 pixels long, record the index and then delete it
        % once all dupicates have been resolved after the while loop has
        % finished (to make sure the indexing doesn't change in the main
        % loop)
        remove_idx = [];
        %h = waitbar(0,['Resolving loops for timepoint ' num2str(iN) '. Please wait...']);
        for loop = 1:length(D_idx)
            % get the pixels for the first duplicate edge
            pix = edgelist{D_idx(loop)};
            % find the middle of the edge
            midnode = floor(length(pix)./2);
            if midnode > 5
                % replace this edge with the first half of the original edge
                edgelist{D_idx(loop)} = pix(1:midnode,:);
                % find the current number of edges at this time point
                n = length(edgelist)+1;
                % add a new edge at the end of the list with the second
                % half of the edge, including an overlap
                edgelist{n} = pix(midnode:end,:);
                % get the pixel co-ordinates for the new edge
                idx = sub2ind([nY,nX],edgelist{D_idx(loop)}(:,1),edgelist{D_idx(loop)}(:,2));
                % write in the new edge to the edgeim
                edgeim(idx) = n;
            else
                % store the index of the edge to be deleted, but don't
                % delete yet as the indexing will alter for the other edges
                remove_idx(end+1) = D_idx(loop);
                %                 % make sure the pixel has the value of the original edge in the
                %                 % edge image
                %                 OE = edgeim(D_idx(loop));
                %                 idx = sub2ind([nY,nX],edgelist{OE}(:,1),edgelist{OE}(:,2));
                %                 % write in the old edge back into the edgeim
                %                 edgeim(idx) = OE;
            end
        end
        % remove the duplicate edges
        if ~isempty(remove_idx)
            % disp('removing duplicate edges')
            edgelist(remove_idx) = [];
        end
    end
end
end

function [edgelist,edgeim] = fnc_resolve_duplicates(edgelist,edgeim)
D_idx = 1;
iteration = 0;
[nY,nX] = size(edgeim);
while ~isempty(D_idx)
    iteration = iteration+1;
    % F_idx is the linear index to the first pixel in each edge array
    F_idx = cellfun(@(x) sub2ind([nY,nX],x(1,1),x(1,2)),edgelist);
    % L_idx is the linear index to the last pixel in each edge array
    L_idx = cellfun(@(x) sub2ind([nY,nX],x(end,1),x(end,2)),edgelist);
    % combine to give the endpoints (EP) of the edge, i.e. nodei and nodej
    EP = [F_idx' L_idx'];
    % Order the rows of nodei and nodej to be smallest first. This makes sure
    % that duplicates are found whichever way round the nodes were
    % initially, but doesn't alter the ordering of the edges
    EP_ordered = [min(EP, [],2) max(EP,[],2)];
    % find the unique rows (EP_unique) with the corresponding index into
    % the original node pairs (ia). The ia index skips any row
    % that is duplicated. The ic index picks the value from EP_unique to
    % recreate the original array. We can use this index to find the index
    % of the first duplicate node.
    [~,ia,ic] = unique(EP_ordered, 'rows');
    % Compare the ia index with a full list of node IDs to give the index
    % of the edges that were skipped
    D_idx = setdiff(1:length(EP),ia);
    % See if there are any duplicate edges
    if ~isempty(D_idx)
        % Go through each loop and split it in two if it is long enough,
        % adding an extra edge at the end of the edgelist. If the duplicate
        % edge is only one pixel long, record the index and then delete it
        % once all dupicates have been resolved after the while loop has
        % finished (to make sure the indexing doesn't change in the main
        % loop)
        remove_idx = [];
        %h = waitbar(0,['Resolving loops for timepoint ' num2str(iN) '. Please wait...']);
        for loop = 1:length(D_idx)
            % get the pixels for the first duplicate edge
            pix = edgelist{D_idx(loop)};
            % find the middle of the edge
            midnode = floor(length(pix)./2);
            if midnode > 1
                % replace this edge with the first half of the original edge
                edgelist{D_idx(loop)} = pix(1:midnode,:);
                % find the current number of edges at this time point
                n = length(edgelist)+1;
                % add a new edge at the end of the list with the second
                % half of the edge, including an overlap
                edgelist{n} = pix(midnode:end,:);
                % get the pixel co-ordinates for the new edge
                idx = sub2ind([nY,nX],edgelist{D_idx(loop)}(:,1),edgelist{D_idx(loop)}(:,2));
                % write in the new edge to the edgeim
                edgeim(idx) = n;
            else
                % store the index of the edge to be deleted, but don't
                % delete yet as the indexing will alter for the other edges
                remove_idx(end+1) = D_idx(loop);
                % make sure the pixel has the value of the original edge in the
                % edge image
                OE = ic(D_idx(loop));
                idx = sub2ind([nY,nX],edgelist{OE}(:,1),edgelist{OE}(:,2));
                % write in the old edge back into the edgeim
                edgeim(idx) = OE;
            end
        end
        % remove the duplicate edges
        if ~isempty(remove_idx)
            % disp('removing duplicate edges')
            edgelist(remove_idx) = [];
        end
    end
end
end

function [G_veins,edgelist] = fnc_weighted_graph(edgelist,W_pixels,skTree)
[nY,nX] = size(W_pixels);
% calculate the node indices to account for the edges added and removed
% I_idx is the linear index to the first pixel in each edge
I_idx = cellfun(@(x) sub2ind([nY nX],x(1,1),x(1,2)),edgelist);
% J_idx is the linear index to the last pixel in each edge array
J_idx = cellfun(@(x) sub2ind([nY nX],x(end,1),x(end,2)),edgelist);
% P_idx is the linear index of each pixel in the cell array of edges
P_idx = cellfun(@(x) sub2ind([nY nX],x(:,1),x(:,2)),edgelist,'UniformOutput',0);
% N_pix is the number of pixels in the edge list
N_pix = cellfun(@(x) numel(x(:,1)), edgelist);
% M_pix is the mid pixel
M_pix = cellfun(@(x,y) x(round(y/2)), P_idx, num2cell(N_pix));
% Get various metrics associated with each edge from the various processing
% stages. Initially the values are returned as cell arrays with individual
% values for each pixel. These are then summarised to give a single value
% for each edge for that metric.
% Get the length as the difference in euclidean distance between pixels
L_val = cellfun(@(x) hypot(diff(x(:,1)),diff(x(:,2))),edgelist,'UniformOutput',0);
L_sum = cellfun(@sum, L_val);
% get the orientation using atan2
O_val = cellfun(@(x) atan2(x(1,1)-x(end,1),x(1,2)-x(end,2)),edgelist,'UniformOutput',1);
% Get the width
W_val = cellfun(@(x) W_pixels(x),P_idx,'UniformOutput',0);
W_mean = cellfun(@mean, W_val);
% Set the edge ID
nEdges = length(I_idx);
EName = (1:nEdges)';
% set all edges ('E') to belong to a loop ('L') initially
EType = repmat({'EL'},nEdges,1);
% Set any edges that belong to a tree with 'T'. Use M_pix to sample the
% middle point in the skeleton
EType(skTree(M_pix)) = {'ET'};
% set the edge weight to the average width
E_weight = W_mean;
% initially the nodes are empty
nodei = zeros(nEdges,1, 'single');
nodej = zeros(nEdges,1, 'single');
node_idx = unique([I_idx'; J_idx']);
% combine the edge metrics into an EdgeTable
names = {'EndNodes', 'node_Idx', 'Name', 'Type', 'Weight', ...
    'Width_initial',  ...
    'Length_initial', ...
    'Orientation_initial', ...
    'N_pix', 'M_pix'};

EdgeTable = table([nodei, nodej], [I_idx', J_idx'], EName, EType, E_weight', ...
    W_mean', ...
    L_sum', ...
    O_val', ...
    N_pix', M_pix', ...
    'VariableNames', names);
% now collect all the node indices to construct a NodeTable. Get the index
% of all the nodes present in i or j
node_ID = (1:length(node_idx))';
node_type = repmat({'E'},length(node_idx),1);
% get the coordinates of each node
[y_pix,x_pix] = ind2sub([nY nX],node_idx);
% Construct the full NodeTable (note at this point the co-ordinates are all
% in pixels
NodeTable = table(node_ID, node_idx, node_type, x_pix, y_pix, ...
    'VariableNames',{'node_ID' 'node_Idx' 'node_Type' 'node_X_pix' 'node_Y_pix'});
% Assemble the MatLab graph object
% Convert the linear node index into a node ID
[~,EdgeTable{:,'EndNodes'}(:,1)] = ismember(EdgeTable{:,'node_Idx'}(:,1), node_idx);
[~,EdgeTable{:,'EndNodes'}(:,2)] = ismember(EdgeTable{:,'node_Idx'}(:,2), node_idx);
G_veins = graph(EdgeTable, NodeTable);
% The graph automatically sorts edge rows based on node i. The
% original edge index is kept in Name. Reoder the edgelist to match the
% graph order, including nodei <-> node j
edgelist = edgelist(G_veins.Edges.Name);
% I_idx is the linear index to the first pixel in each edge array
I_idx = cellfun(@(x) sub2ind([nY nX],x(1,1),x(1,2)),edgelist);
% Find which EndNodes in the edge table do not start with nodei, and have
% therefore been re-ordered
switch_idx = G_veins.Nodes{G_veins.Edges.EndNodes(:,1),'node_Idx'} ~= I_idx';
% flip the order of pixels in the edgelist to match
edgelist(switch_idx) = cellfun(@(x) flipud(x), edgelist(switch_idx), 'UniformOutput',0);
end

function [G_veins,edgelist_center] = fnc_refine_width(G_veins,edgelist,im,im_cnn,W_pixels)
[nY,nX] = size(W_pixels);
% Calculate the node degree
G_veins.Nodes.node_Degree = degree(G_veins);
% Get the edges from the graph
nN = numnodes(G_veins);
[i,j] = findedge(G_veins);
% Construct a sparse, symmetric weighted adjacency matrix from the graph
% widths. The values have to be in double precision for
% sparse to work.
width = G_veins.Edges.Width_initial;
A = sparse(i,j,double(width),nN,nN);
% make the matrix symmetric
A = A + A.' - diag(diag(A));
% Calculate the major edge width and the edge index incident at the node
% (initially in double precision inherited from the sparse matrix
[edge_Maj, max_idx] = max(A,[],2); % column format to match the table later
edge_Maj = full(edge_Maj);
% Calculate the width of the penultimate edge width by removing the max and
% recalculating the max for the remainder
rows = (1:size(A,1))';
mx_idx = sub2ind(size(A),rows,max_idx);
Amid = A;
Amid(mx_idx) = 0;
G_veins.Nodes.node_Mid = single(full(max(Amid,[],2)));
% Calculate the minimum edge width. To calculate the initial min of a
% sparse matrix, take the negative and then add back the maximum
[nnzr, nnzc] = find(A);
B = -A + sparse(nnzr,nnzc,max(edge_Maj),size(A,1),size(A,2));
mn = max(B,[],2);
G_veins.Nodes.node_Min = single(-(mn-max(edge_Maj)));
G_veins.Nodes.node_Maj = single(edge_Maj);
% get the degree for nodei and nodej
NDegI = G_veins.Nodes{G_veins.Edges.EndNodes(:,1),'node_Degree'};
NDegJ = G_veins.Nodes{G_veins.Edges.EndNodes(:,2),'node_Degree'};
% get the maximum edge weight for nodei and nodej
NMajI = G_veins.Nodes{G_veins.Edges.EndNodes(:,1),'node_Maj'};
NMajJ = G_veins.Nodes{G_veins.Edges.EndNodes(:,2),'node_Maj'};
% Get the penultimate edge weight for nodei and nodej
NMidI = G_veins.Nodes{G_veins.Edges.EndNodes(:,1),'node_Mid'};
NMidJ = G_veins.Nodes{G_veins.Edges.EndNodes(:,2),'node_Mid'};
% Get the number of pixels in the edge
N_pix = G_veins.Edges.N_pix;
% Get the length as the difference in euclidean distance between pixels
L_val = cellfun(@(x) single(hypot(diff(x(:,1)),diff(x(:,2)))),edgelist,'UniformOutput',0);
% L_sum is the cumulative length of the edge starting from nodei
L_sum = cellfun(@(x) cumsum(x),L_val,'UniformOutput',0);
% Initially use half the average (NAveI) or maximum (NMajI) at
% the node as the overlap distance to avoid. Limit the index to
% 50% of the number of pixels minus 1 (to ensure that there is
% a minimum length of edge remaining). Find where the index
% where cumulative length equals or exceeeds half the overlap
% width
L_idx = cellfun(@(x,y) find(x>=y/2,1),L_sum,num2cell(NMajI)','UniformOutput',0);
% empty cells means the overlap distance is longer than the
% edge. Set these values to be infinite initially, but will be
% set to half the number of pixels
L_max = cellfun('isempty',L_idx);
L_idx(L_max) = {inf};
% constrain the maximum offset to be half the number of pixels - 1
% for edges with overlap to make sure there is some edge left.
first = min(cell2mat(L_idx)',max(round(N_pix.*0.5)-1,1));
first(first<1) = 1;
% If the edge width is equal to or greater than the penultimate width of
% all edges at the node i, or the node degree is 2 or less, then keep the
% full length of the edge. i.e. set the starting index to 1;
idx =  width >= NMidI | NDegI <= 2;
%idx =  NDegI <= 2 & NPerimI == 0;
first(idx) = 1;
% Repeat for node j, using half the largest edge width as an
% index into the array from a minimum of some fraction of the
% number of pixels + 1 (to ensure that there is a minimum edge
% length remaining). Switch the order to set nodej at the start
% for every edge
L_val = cellfun(@(x) flipud(x),L_val,'UniformOutput',0);
L_sum = cellfun(@(x) cumsum(x),L_val,'UniformOutput',0);
% use half of the average (NAveJ) or maximum (NMajJ) at the
% node as the overlap distance to avoid. L_idx is the index
% into the flipped edgelist.
L_idx = cellfun(@(x,y) find(x>=y/2,1),L_sum,num2cell(NMajJ)','UniformOutput',0);
% if empty then the edge is within the overlap. Set the index
% to be inf initially, it will be set to half the width in the
% next step
L_max = cellfun('isempty',L_idx);
L_idx(L_max) = {inf};
% set the maximum index as N_pix-1 and subtract from N_pix
last = N_pix - min(cell2mat(L_idx)',max(round(N_pix.*0.5)-1,1));
last(last<1) = 1;
last(last>N_pix) = N_pix(last>N_pix);
% If the edge width is equal to or greater than the penultimate width of
% all edges at the node i, or the node degree is 2 or less, then keep the
% full length of the edge. i.e. set the final index to N_pix;
idx =  width >= NMidJ | NDegJ <= 2;
%idx =  NDegJ <= 2 & NPerimJ == 0;
% Combine the sets of indices
last(idx) = N_pix(idx);
% now collect the linear index of the pixels representing the center of the
% edge using the first and last indices
edgelist_center = cellfun(@(x,f1,l1) [x(f1:l1,1),x(f1:l1,2)],edgelist,num2cell(first'),num2cell(last'),'UniformOutput',0);
% Convert the selected pixels to a linear index
P_idx = cellfun(@(x) sub2ind([nY nX],x(:,1),x(:,2)),edgelist_center,'UniformOutput',0);
% Get the intensity for each pixel from the original image using P_idx
im = single(im);
I_val = cellfun(@(x) im(x),P_idx,'UniformOutput',0);
I_mean = cellfun(@mean, I_val);
I_std = cellfun(@std, I_val);
G_veins.Edges.Intensity = I_mean';
G_veins.Edges.Intensity_cv = I_std'./I_mean';
% Get the probability for each pixel from the cnn image using P_idx
im_cnn = single(im_cnn);
P_val = cellfun(@(x) im_cnn(x),P_idx,'UniformOutput',0);
P_mean = cellfun(@mean, P_val);
P_std = cellfun(@std, P_val);
G_veins.Edges.Probability = P_mean';
G_veins.Edges.Probability_cv = P_std'./P_mean';
% Get the length as the difference in euclidean distance between pixels
CL_val = cellfun(@(x) hypot(diff(x(:,1)),diff(x(:,2))),edgelist_center,'UniformOutput',0);
CL_sum = cellfun(@sum, CL_val);
% Get the width from the integral granulometry
CW_val = cellfun(@(x) W_pixels(x),P_idx,'UniformOutput',0);
CW_mean = cellfun(@mean, CW_val);
CW_std = cellfun(@std, CW_val);
CW_cv = CW_std./CW_mean;
% get the average orientation of the center-width region. These
% will tend to be in the range pi/2 to -pi/2 because nodei for all
% edges is likely to be to the left of node j because of the
% re-ordering in the graph.
[midy, midx] = ind2sub([nY,nX],G_veins.Edges.M_pix);
mid = mat2cell([midy midx],ones(length(midx),1),2)';
% i-mid for y, mid-i for x for ij and j-mid for y, mid-i for x for ji
% flip image axis so positive is counterclockwise from
% horizontal
CO_ij = cellfun(@(x,y) atan2(x(1,1)-y(1,1), y(1,2)-x(1,2)),edgelist_center,mid,'UniformOutput',1);
CO_ji = cellfun(@(x,y) atan2(x(end,1)-y(1,1), y(1,2)-x(end,2)),edgelist_center,mid,'UniformOutput',1);
% update the edgetable
G_veins.Edges.Length = CL_sum';
G_veins.Edges.Width = CW_mean';
G_veins.Edges.Width_cv = CW_cv';
G_veins.Edges.Or_ij = CO_ij';
G_veins.Edges.Or_ji = CO_ji';
% calculate the tortuosity as the ratio of the distance between the end
% points (chord length) and the total length
CL_chord = cellfun(@(x) hypot(x(1,1)-x(end,1),x(1,2)-x(end,2)),edgelist_center);
G_veins.Edges.Tortuosity = CL_sum'./CL_chord';
G_veins.Edges.Tortuosity(isinf(G_veins.Edges.Tortuosity)) = 1;
% calculate the center-weighted radius
R = G_veins.Edges.Width/2;
% calculate the center-weighted area
G_veins.Edges.Area = pi.*(R.^2);
% calculate the center-weighted surface area
G_veins.Edges.SurfaceArea = pi.*2.*R.*G_veins.Edges.Length ;
% calculate the center-weighted volume
G_veins.Edges.Volume = G_veins.Edges.Length.*G_veins.Edges.Area;
% calculate the center-weighted resistance to flow for area or Poiseuille flow
G_veins.Edges.R2 = G_veins.Edges.Length./(R.^2);
G_veins.Edges.R4 = G_veins.Edges.Length./(R.^4);
% find the feature edges and set them to the max
F_idx = find(strcmp(G_veins.Edges.Type,'F'));
G_veins.Edges.Width(F_idx) = max(CW_mean);
G_veins.Edges.Width_cv(F_idx) = 0;
% recalculate the node parameters from the center-width estimate
rows = (1:size(A,1))';
% calculate a weighted adjacency matrix for the center width
A = sparse(i,j,double(G_veins.Edges.Width),nN,nN);
A = A + A.' - diag(diag(A));
G_veins.Nodes.node_Strength = single(full(sum(A,2)));
% calculate a weighted adjacency matrix for orientation for
% edges ij
O = sparse(i,j,double(G_veins.Edges.Or_ij),nN,nN);
% swap the order of the nodes and subtract pi to get the
% orientation of edges ji
Oji = sparse(j,i,double(G_veins.Edges.Or_ji),nN,nN);
O = O + Oji - diag(diag(O));
% Calculate the initial max edge width incident at the node (initially
% in double precision)
[edge_Maj, max_idx] = max(A,[],2);
edge_Maj = full(edge_Maj);
mx_idx = sub2ind(size(A),rows,max_idx);
% get the orientation of the largest edge from the orientation adjacency
% matrix using the max_idx index
G_veins.Nodes.node_Omaj = single(full(O(mx_idx)));
%G_veins.Nodes.node_Omaj(abs(G_veins.Nodes.node_Omaj)>180) = mod(G_veins.Nodes.node_Omaj(abs(G_veins.Nodes.node_Omaj)>180),180);
% Calculate the minimum edge width. To calculate the min of a
% sparse matrix, take the negative and then add back the maximum. This
% avoids finding a full array of zeros!
MaxA = sparse(i,j,max(edge_Maj),nN,nN);
MaxA = MaxA + MaxA.' - diag(diag(MaxA));
B = -A + MaxA;
[edge_Min, min_idx] = max(B,[],2);
mn_idx = sub2ind(size(A),rows,min_idx);
% convert back to absolute positive value by negating and adding back the max
G_veins.Nodes.node_Min = single(-full(edge_Min)+max(edge_Maj)); % convert to single precision
% get the orentation of the weakest edge from the orientation adjacency
% matric and the min_idx
G_veins.Nodes.node_Omin = single(full(O(mn_idx)));
%G_veins.Nodes.node_Omin(abs(G_veins.Nodes.node_Omin)>180) = mod(G_veins.Nodes.node_Omin(abs(G_veins.Nodes.node_Omin)>180),180);
% Calculate the width of the penultimate edge width by removing the max
% values from the adjacency matrix and recalculating max for the remainder
Amid = A;
Amid(mx_idx) = 0;
[edge_Mid, mid_idx] = max(Amid,[],2);
pn_idx = sub2ind(size(A),rows,mid_idx);
G_veins.Nodes.node_Mid = single(full(edge_Mid)); % convert to single precision
% get the orientation of the penultimate edge from the orientation
% adjacency matrix using the pn_idx
G_veins.Nodes.node_Omid = single(full(O(pn_idx)));
%G_veins.Nodes.node_Omid(abs(G_veins.Nodes.node_Omid)>180) = mod(G_veins.Nodes.node_Omid(abs(G_veins.Nodes.node_Omid)>180),180);
G_veins.Nodes.node_Maj = single(edge_Maj); % convert max to single precision
G_veins.Nodes.node_Average = G_veins.Nodes.node_Strength./G_veins.Nodes.node_Degree;
% tidy up results for k=1 nodes
idx = G_veins.Nodes.node_Degree == 1;
G_veins.Nodes.node_Mid(idx) = nan;
G_veins.Nodes.node_Min(idx) = nan;
G_veins.Nodes.node_Omid(idx) = nan;
G_veins.Nodes.node_Omin(idx) = nan;
% calculate the ratios for the radii of smallest to largest and
% intermediate to largest
G_veins.Nodes.node_Min_Maj = G_veins.Nodes.node_Min./G_veins.Nodes.node_Maj;
G_veins.Nodes.node_Mid_Maj = G_veins.Nodes.node_Mid./G_veins.Nodes.node_Maj;
G_veins.Nodes.node_Min_Mid = G_veins.Nodes.node_Min./G_veins.Nodes.node_Mid;
% calculate the absolute angles around the branch
G_veins.Nodes.node_Omin_Omaj = pi - abs(pi - abs(G_veins.Nodes.node_Omin-G_veins.Nodes.node_Omaj));
G_veins.Nodes.node_Omid_Omaj = pi - abs(pi - abs(G_veins.Nodes.node_Omid-G_veins.Nodes.node_Omaj));
G_veins.Nodes.node_Omin_Omid = pi - abs(pi - abs(G_veins.Nodes.node_Omin-G_veins.Nodes.node_Omid));
end

function [CW_pixels,width,coded] = fnc_coded_skeleton(im,sk,bw_mask,G_veins,edgelist,edgelist_center,sk_width,cmap)
[nY,nX] = size(sk);
% set up a blank image
CW_pixels = zeros([nY,nX], 'single');
CW_scaled = zeros([nY,nX], 'single');
width = zeros([nY,nX], 'single');
% get the value of the edges from the graph. These are sorted
% automatically when the graph is set up in order of node i
E = G_veins.Edges.Width;
% get just the edges (not any features)
E_idx = contains(G_veins.Edges.Type,'E');
E = E(E_idx);
% get the linear pixel index for each edge
% excluding the features
P_idx = cellfun(@(x) sub2ind([nY nX],x(:,1),x(:,2)),edgelist_center(E_idx),'UniformOutput',0);
% concatenate all the pixels indices to give a single vector
P_all = cat(1,P_idx{:});
% calculate an edge width skeleton based on the central width
% for the selected edges
V_int = num2cell(E);
% duplicate the value for the number of pixels in the center part of each edge
V_idx = cellfun(@(x,y) repmat(y,length(x),1), P_idx, V_int','UniformOutput',0);
% concatenate all the values into a single vector
V_all = cat(1,V_idx{:});
% set the edge values to the central width for the center part of each
% edge. The CW_pixels image is a grayscale image with the intensity
% of the skeleton making up each edge set to the width in pixels. The
% overlap region is set to zero.
CW_pixels(P_all) = single(V_all);
% To construct the color-coded skeleton, normalise the range between 2 and
% 256 as an index into the colourmap
Emin = min(E);
Emax = max(E);
cmap_idx = ceil(254.*((E-Emin)./(Emax-Emin)))+2;
% set any nan values to 1
cmap_idx(isnan(cmap_idx)) = 1;
% convert to a cell array
V_int = num2cell(cmap_idx);
% duplicate the value for the number of pixels in each edge
V_idx = cellfun(@(x,y) repmat(y,length(x),1), P_idx, V_int','UniformOutput',0);
% concatenate all the values into a single vector
V_all = cat(1,V_idx{:});
% set the edge values to the central width for each edge in the image
% scaled between the min and the max
CW_scaled(P_all) = single(V_all);
if sk_width > 1
    sk_D = imdilate(sk, ones(sk_width));
    CW_scaled = imdilate(CW_scaled, ones(sk_width));
else
    sk_D = sk;
end
im2 = im;%uint8(255.*im);
im2(sk_D) = 0;
im_rgb = uint8(255.*ind2rgb(im2,gray(256)));
sk_rgb = uint8(255.*ind2rgb(uint8(CW_scaled),cmap));
coded = imadd(im_rgb,sk_rgb);
% add the boundaries of the mask
coded = imoverlay(coded,bwperim(bw_mask),'m');
% Repeat the process but this time calculate the full pixel skeleton
% (including the overlap regions) and coded by the center weight for
% export. Get the linear pixel index for each edge excluding the features
P_idx = cellfun(@(x) sub2ind([nY nX],x(:,1),x(:,2)),edgelist(E_idx),'UniformOutput',0);
% concatenate all the pixels indices to give a single vector
P_all = cat(1,P_idx{:});
% duplicate the value for the number of pixels in each edge
V_idx = cellfun(@(x,y) repmat(y,length(x),1), P_idx, V_int','UniformOutput',0);
% concatenate all the values into a single vector
V_all = cat(1,V_idx{:});
% set the edge values to the central width for each edge in the image
width(P_all) = single(V_all);
% make sure that the junctions are set to the local maximum, so that
% junctions are preserved as part of the strongest edge
temp = colfilt(width,[3 3],'sliding',@max);
bp = bwmorph(sk,'branchpoints');
% The width image is a complete skeleton with no breaks in the overlap
% region coded by the center-weighted thickness of the edge
width(bp) = temp(bp);
% % set the masked regions to -1
width(~bw_mask) = -1;
end

function [G_veins, sk_polygon, bw_polygons, bw_areoles, total_area_mask, polygon_LM] = fnc_polygon_find(G_veins,bw_cnn,sk,skLoop,bw_mask)
% remove any partial areas that are not fully bounded and therefore contact
% the edge
area = imclearborder(~sk & bw_mask,4);
% remove any partial areoles that are touching any masked regions
[r,c] = find(~bw_mask);
area_mask = ~(bwselect(area | ~bw_mask,c,r,4));
% get rid of any vestigial skeleton lines within the mask
area_mask = bwmorph(area_mask, 'open');
% fill the skeleton lines on the total area
total_area_mask = imdilate(area, ones(3)) & area_mask;
% remove isolated polygons
total_area_mask = bwareafilt(total_area_mask, 1,'largest');
% The polygons include the areoles and the vein itself.
bw_polygons = ~skLoop & total_area_mask;
% find areas that are too small (note bwareafilt does not work as it is not
% possible to set connectivity to 4)
CC = bwconncomp(bw_polygons,4);
stats = regionprops(CC, 'Area');
% arbitary threshold of ~10x10 pixels
idx = find([stats.Area] > 100);
bw_polygons  = ismember(labelmatrix(CC), idx);
% The areoles exclude the full width of the veins
bw_areoles = ~bw_cnn & total_area_mask & bw_polygons ;
% remove any orphan pixels
bw_areoles = bwmorph(bw_areoles,'clean');
% trim the skeleton to match
sk_polygon = sk & total_area_mask;
% construct a label matrix of areas including the veins
LM = single(bwlabel(bw_polygons,4));
% remove the skeleton in the calculation of the neighbouring
% area
LM(sk_polygon) = NaN;
% find the neighbours of each edge
Neighbour1 = single(colfilt(LM,[3 3],'sliding',@max));
Neighbour2 = single(colfilt(LM,[3 3],'sliding',@(x) min(x,[],'Omitnan')));
% add the neighbours to the graph
G_veins.Edges.Ai = Neighbour1(G_veins.Edges.M_pix);
G_veins.Edges.Aj = Neighbour2(G_veins.Edges.M_pix);
% replace the skeleton pixels
LM(sk_polygon) = 0;
% dilate the area to include the pixel skeleton
polygon_LM = imdilate(LM, [0 1 0; 1 1 1; 0 1 0]);
end

function im_polygons_rgb = fnc_polygon_image(polygon_stats, sk_polygon, total_area_mask)
[nY,nX] = size(total_area_mask);
% construct a colour-coded area image from the log area
idx = cat(1,polygon_stats.PixelIdxList);
polygon_areas = [polygon_stats.Area];
%
logA = log10(polygon_areas);
logA(isinf(logA)) = nan;
Amin = 1;
Amax = log10(sum(total_area_mask(:)));
% normalise the range between 2 and 256 as an index into the
% colourmap
A_idx = ceil(254.*((logA-Amin)./(Amax-Amin)))+1;
% set any nan values to 1
A_idx(isnan(A_idx)) = 1;
% duplicate the area value for the number of pixels in each edge
A_val = cellfun(@(x,y) repmat(x,y,1), num2cell(A_idx),num2cell(polygon_areas),'UniformOutput',0);
% concatenate all the values into a single vector
A_all = single(cat(1,A_val{:}));
% set the edge values to the edge ID for each edge in the image
%im_polygons = zeros(nY,nX);
im_polygons = zeros(nY,nX, 'single');
im_polygons(idx) = A_all;
cmap = cool(256);
cmap(1,:) = 1;
% punch out the skeleton to make the polygons distinct
im_polygons(sk_polygon) = 0;
im_polygons = uint8(im_polygons);
im_polygons_rgb = uint8(255.*ind2rgb(im_polygons,cmap));
end

function [areole_stats,polygon_stats] = fnc_polygon_analysis(bw_polygons,bw_areoles, polygon_LM,FullMetrics)
% measure the stats for the polygonal regions using the label matrix
if FullMetrics == 0
    % don't calculate the most time consuming metrics (ConvexArea,
    % Solidity) and the internal distance metrics.
    polygon_stats = regionprops(polygon_LM, ...
        'Area', ...
        'Centroid', ...
        'Eccentricity', ...
        'EquivDiameter', ...
        'MajorAxisLength', ...
        'MinorAxisLength', ...
        'Orientation', ...
        'Perimeter', ...
        'PixelIdxList');
else
    P_stats = regionprops(polygon_LM, ...
        'Area', ...
        'Centroid', ...
        'ConvexArea', ...
        'Eccentricity', ...
        'EquivDiameter', ...
        'MajorAxisLength', ...
        'MinorAxisLength', ...
        'Orientation', ...
        'Perimeter', ...
        'Solidity', ...
        'PixelIdxList');
    % get the maximum distance to the skeleton for each area
    D_stats = regionprops(polygon_LM,bwdist(~bw_polygons),'MaxIntensity','MeanIntensity');
    % rename the fields
    [D_stats.('MaxDistance')] = D_stats.('MaxIntensity');
    D_stats = rmfield(D_stats,'MaxIntensity');
    % get the average distance to the skeleton for each area
    [D_stats.('MeanDistance')] = D_stats.('MeanIntensity');
    D_stats = rmfield(D_stats,'MeanIntensity');
    names = [fieldnames(P_stats); fieldnames(D_stats)];
    % combine the two stats arrays
    polygon_stats = cell2struct([struct2cell(P_stats);struct2cell(D_stats)],names,1);
end
% calculate additional parameters
% polygon_stats = P_stats;
ID = num2cell(1:length(polygon_stats));
Circularity = num2cell((4.*pi.*[polygon_stats.Area])./([polygon_stats.Perimeter].^2));
Elongation = num2cell([polygon_stats.MajorAxisLength]./[polygon_stats.MinorAxisLength]);
Roughness = num2cell(([polygon_stats.Perimeter].^2)./[polygon_stats.Area]);
[polygon_stats(:).Circularity] = deal(Circularity{:});
[polygon_stats(:).Elongation] = deal(Elongation{:});
[polygon_stats(:).Roughness] = deal(Roughness{:});
[polygon_stats(:).ID] = deal(ID{:});
% modify the label matrix to only include the areoles, but with the same ID
% as the polygons
areole_LM = polygon_LM;
areole_LM(~bw_areoles) = 0;
% run all the stats for the areoles
if FullMetrics == 0
    % don't calculate the most time consuming metrics (ConvexArea,
    % Solidity) and don't calculate the internal distance measures
    areole_stats = regionprops(areole_LM, ...
        'Area', ...
        'Centroid', ...
        'Eccentricity', ...
        'EquivDiameter', ...
        'MajorAxisLength', ...
        'MinorAxisLength', ...
        'Orientation', ...
        'Perimeter', ...
        'PixelIdxList');
else
    A_stats = regionprops(areole_LM, ...
        'Area', ...
        'Centroid', ...
        'ConvexArea', ...
        'Eccentricity', ...
        'EquivDiameter', ...
        'MajorAxisLength', ...
        'MinorAxisLength', ...
        'Orientation', ...
        'Perimeter', ...
        'Solidity', ...
        'PixelIdxList');
    % get the maximum distance to the skeleton for each area
    D_stats = regionprops(areole_LM,bwdist(~bw_areoles),'MaxIntensity','MeanIntensity');
    % find any empty cells in the maxIntensity and replace with nan. These
    % occur when an area is present in the polygons, but without a
    % corresponding areole. This ensures that the metric is saved as a number
    % rather than a cell array
    test4empty = cellfun(@isempty,{D_stats.MaxIntensity});
    if any(test4empty)
        [D_stats(test4empty).MaxIntensity] = deal(nan);
    end
    % rename the fields
    [D_stats.('MaxDistance')] = D_stats.('MaxIntensity');
    D_stats = rmfield(D_stats,'MaxIntensity');
    % get the average distance to the skeleton for each area
    [D_stats.('MeanDistance')] = D_stats.('MeanIntensity');
    D_stats = rmfield(D_stats,'MeanIntensity');
    names = [fieldnames(A_stats); fieldnames(D_stats)];
    % combine the two stats arrays
    areole_stats = cell2struct([struct2cell(A_stats);struct2cell(D_stats)],names,1);
end
% calculate additional parameters
ID = num2cell(1:length(areole_stats));
Circularity = num2cell((4.*pi.*[areole_stats.Area])./([areole_stats.Perimeter].^2));
Elongation = num2cell([areole_stats.MajorAxisLength]./[areole_stats.MinorAxisLength]);
Roughness = num2cell(([areole_stats.Perimeter].^2)./[areole_stats.Area]);
[areole_stats(:).Circularity] = deal(Circularity{:});
[areole_stats(:).Elongation] = deal(Elongation{:});
[areole_stats(:).Roughness] = deal(Roughness{:});
[areole_stats(:).ID] = deal(ID{:});
end

function G_areas = fnc_area_graph(G_veins,area_stats)
% Construct a NodeTable with the node for each area, along with the
% corresponding metrics
if ~isempty(area_stats)
    % extract the centroid values
    polygon_Centroid = cat(1,area_stats.Centroid);
    % remove the centroid and PixelIdxList fields
    stats = rmfield(area_stats,{'Centroid';'PixelIdxList'});
    % convert to a NodeTable
    NodeTable = struct2table(stats);
    % add the centroid positions back in as separate columns in the table
    NodeTable.node_X_pix = polygon_Centroid(:,1);
    NodeTable.node_Y_pix = polygon_Centroid(:,2);
    % Construct an EdgeTable with the width of the pixel skeleton edge that it
    % crosses
    names = {'EndNodes' 'Width' 'Name'};
    % If an area has a k=1 edge within it (i.e. a terminal vein) there could be
    % two possible edges to an adjacent area on either side with the same ID.
    % Therefore duplicate edges are resolved to the minimum. Reorder nodes to
    % be minimum first
    i = min(G_veins.Edges.Ai, G_veins.Edges.Aj);
    j = max(G_veins.Edges.Ai, G_veins.Edges.Aj);
    edges = [i j G_veins.Edges.Width G_veins.Edges.Name];
    % sort by the edge width and keep the smallest of the duplicate edges
    edges = sortrows(edges,3);
    [~,idx] = unique(edges(:,1:2),'rows');
    edges = double(edges(idx,:));
    idx = max(edges(:,1:2)==0, [],2);
    edges(idx,:) = [];
    % remove edges connected to themselves
    idx = diff(edges(:,1:2),[],2)==0;
    edges(idx,:) = [];
    % remove edges not present in the polygons
    
    % construct the graph
    EdgeTable = table([edges(:,1) edges(:,2)],edges(:,3),edges(:,4), 'VariableNames', names);
    G_areas = graph(EdgeTable,NodeTable);%,'OmitSelfLoops');
    % check the number of components and only keep the largest
    CC = conncomp(G_areas);
    G_areas = rmnode(G_areas,find(CC>1));
else
    G_areas = [];
end
end

function T = fnc_summary_veins(G_veins,total_area,micron_per_pixel)
% set calibration factors
mm = micron_per_pixel./1000;
% get the index for all edges, loops and trees
E_idx = find(contains(G_veins.Edges.Type,'E'));
L_idx = find(contains(G_veins.Edges.Type,'L'));
T_idx = find(contains(G_veins.Edges.Type,'T'));
% get vein and node numbers
T = table;
T.VNE = numedges(G_veins); % dimensionless
T.VNN = numnodes(G_veins); % dimensionless
Deg = degree(G_veins); % dimensionless
T.VFE = sum(Deg==1); % freely ending veins have degree one
T.VFEratio = T.VFE / T.VNE; % Number of FEV divided by number of edges
T.Valpha = (T.VNE - T.VNN + 1)/((2*T.VNN)-5); % dimensionless
% get the total length of veins (in mm)
T.VTotL = sum(G_veins.Edges.Length(E_idx)).*mm; %
T.VLoopL = sum(G_veins.Edges.Length(L_idx)).*mm; %
T.VTreeL = sum(G_veins.Edges.Length(T_idx)).*mm; %
% % % % get the total volume of the veins
% % % T.VTotV = sum(G_veins.Edges.Volume(E_idx)).*(mm.^3); %
% % % T.VLoopV = sum(G_veins.Edges.Volume(L_idx)).*(mm.^3); %
% % % T.VTreeV = sum(G_veins.Edges.Volume(T_idx)).*(mm.^3); %
% get the weighted vein width
T.VTotW = (1/T.VTotL).*(sum(G_veins.Edges.Length(E_idx).*G_veins.Edges.Width(E_idx))).*mm;
T.VLoopW = (1/T.VLoopL).*(sum(G_veins.Edges.Length(L_idx).*G_veins.Edges.Width(L_idx))).*mm;
T.VTreeW = (1/T.VTreeL).*(sum(G_veins.Edges.Length(T_idx).*G_veins.Edges.Width(T_idx))).*mm;
% set the graph weight to be the length (in microns);
G_veins.Edges.Weight = G_veins.Edges.Length.*mm;
% total areas analysed within the mask
T.TotA = total_area.*(mm.^2);
% get the density measurements
T.VTotLD = T.VTotL / T.TotA;
T.VLoopLD = T.VLoopL / T.TotA;
T.VTreeLD = T.VTreeL / T.TotA;
% % % T.VTotVD = T.VTotV / T.TotA;
% % % T.VLoopVD = T.VLoopV / T.TotA;
% % % T.VTreeVD = T.VTreeV / T.TotA;
T.VNND = T.VNN / T.TotA;
% get the summary statistics for all relevant edge metrics in the table
% usage: T = fnc_summary(T,metric,prefix,suffix,transform,units)
T = fnc_summary(T,G_veins.Edges.Length(E_idx),'VTot','Len','none',mm);
T = fnc_summary(T,G_veins.Edges.Length(L_idx),'VLoop','Len','none',mm);
T = fnc_summary(T,G_veins.Edges.Length(T_idx),'VTree','Len','none',mm);
T = fnc_summary(T,G_veins.Edges.Width(E_idx),'VTot','Wid','none',mm);
T = fnc_summary(T,G_veins.Edges.Width(L_idx),'VLoop','Wid','none',mm);
T = fnc_summary(T,G_veins.Edges.Width(T_idx),'VTree','Wid','none',mm);
% % % T = fnc_summary(T,G_veins.Edges.SurfaceArea(E_idx),'VTot','SAr','none',mm);
% % % T = fnc_summary(T,G_veins.Edges.SurfaceArea(L_idx),'VLoop','SAr','none',mm);
% % % T = fnc_summary(T,G_veins.Edges.SurfaceArea(T_idx),'VTree','SAr','none',mm);
% % % T = fnc_summary(T,G_veins.Edges.Volume(E_idx),'VTot','Vol','none',mm^3);
% % % T = fnc_summary(T,G_veins.Edges.Volume(L_idx),'VLoop','Vol','none',mm^3);
% % % T = fnc_summary(T,G_veins.Edges.Volume(T_idx),'VTree','Vol','none',mm^3);
T = fnc_summary(T,G_veins.Edges.Tortuosity(E_idx),'V','Tor','none',1);
% % % T = fnc_summary(T,G_veins.Edges.Or_ij(E_idx),'V','Ori','circ',1);
% % % T = fnc_summary(T,G_veins.Nodes.node_Min,'N','Bmn','none',1);
% % % T = fnc_summary(T,G_veins.Nodes.node_Mid,'N','Bmd','none',1);
% % % T = fnc_summary(T,G_veins.Nodes.node_Maj,'N','Bmx','none',1);
% % % T = fnc_summary(T,G_veins.Nodes.node_Min_Mid,'N','Rnd','none',1);
% % % T = fnc_summary(T,G_veins.Nodes.node_Min_Maj,'N','Rnx','none',1);
% % % T = fnc_summary(T,G_veins.Nodes.node_Mid_Maj,'N','Rdx','none',1);
% % % T = fnc_summary(T,G_veins.Nodes.node_Omin_Omid,'N','And','circ',1);
% % % T = fnc_summary(T,G_veins.Nodes.node_Omin_Omaj,'N','Anx','circ',1);
% % % T = fnc_summary(T,G_veins.Nodes.node_Omid_Omaj,'N','Adx','circ',1);
end

function T = fnc_summary_areoles(G_areoles,micron_per_pixel,FullMetrics)
% set calibration factors
mm = micron_per_pixel./1000;
T = table;
T.ATA = sum(G_areoles.Nodes.Area).*(mm.^2);
T.ANN = numnodes(G_areoles);
% get areole statistics
T = fnc_summary(T,G_areoles.Nodes.Area,'A','Are','none',mm^2);
% % % T = fnc_summary(T,G_areoles.Nodes.Eccentricity,'A','Ecc','none',1);
% % % T = fnc_summary(T,G_areoles.Nodes.MajorAxisLength,'A','Maj','none',mm);
% % % T = fnc_summary(T,G_areoles.Nodes.MinorAxisLength,'A','Min','none',mm);
% % % T = fnc_summary(T,G_areoles.Nodes.EquivDiameter,'A','EqD','none',mm);
% % % T = fnc_summary(T,G_areoles.Nodes.Perimeter,'A','Per','none',mm);
T = fnc_summary(T,G_areoles.Nodes.Elongation,'A','Elg','none',1);
% % % T = fnc_summary(T,G_areoles.Nodes.Circularity,'A','Cir','none',1);
% % % T = fnc_summary(T,G_areoles.Nodes.Roughness,'A','Rgh','none',1);
% % % T = fnc_summary(T,G_areoles.Nodes.Orientation,'A','Ori','circ',1);
% % % % add in additional metrics if required
% % % if FullMetrics == 1
% % %     T = fnc_summary(T,G_areoles.Nodes.ConvexArea,'A','CnA','none',mm^2);
% % %     T = fnc_summary(T,G_areoles.Nodes.Solidity,'A','Sld','none',mm);
% % %     T = fnc_summary(T,G_areoles.Nodes.MeanDistance,'A','Dav','none',1);
% % %     T = fnc_summary(T,G_areoles.Nodes.MaxDistance,'A','Dmx','none',1);
% % % end
end

function T = fnc_summary_polygons(G_polygons,micron_per_pixel,FullMetrics)
% set calibration factors
mm = micron_per_pixel./1000;
T = table;
T.PTA = sum(G_polygons.Nodes.Area).*(mm.^2);
T.PNN = numnodes(G_polygons);
T.Ploop = T.PNN / T.PTA; % should be the same as T.Aloop
% get polgonal area statistics
T = fnc_summary(T,G_polygons.Nodes.Area,'P','Are','none',mm^2);
% % % T = fnc_summary(T,G_polygons.Nodes.Eccentricity,'P','Ecc','none',1);
% % % T = fnc_summary(T,G_polygons.Nodes.MajorAxisLength,'P','Maj','none',mm);
% % % T = fnc_summary(T,G_polygons.Nodes.MinorAxisLength,'P','Min','none',mm);
% % % T = fnc_summary(T,G_polygons.Nodes.EquivDiameter,'P','EqD','none',mm);
% % % T = fnc_summary(T,G_polygons.Nodes.Perimeter,'P','Per','none',mm);
T = fnc_summary(T,G_polygons.Nodes.Elongation,'P','Elg','none',1);
% % % T = fnc_summary(T,G_polygons.Nodes.Circularity,'P','Cir','none',1);
% % % T = fnc_summary(T,G_polygons.Nodes.Roughness,'P','Rgh','none',1);
% % % T = fnc_summary(T,G_polygons.Nodes.Orientation,'P','Ori','circ',1);
% % % % add in additional metrics if required
% % % if FullMetrics == 1
% % %     T = fnc_summary(T,G_polygons.Nodes.ConvexArea,'P','CnA','none',mm^2);
% % %     T = fnc_summary(T,G_polygons.Nodes.Solidity,'P','Sld','none',mm);
% % %     T = fnc_summary(T,G_polygons.Nodes.MeanDistance,'P','Dav','none',1);
% % %     T = fnc_summary(T,G_polygons.Nodes.MaxDistance,'P','Dmx','none',1);
% % % end
end

function T = fnc_summary(T,metric,prefix,suffix,transform,units)
% ignore infinite or nan values
metric = metric(~isinf(metric) & ~isnan(metric));
% transform the data if required
switch transform
    case 'log'
        metric = log(metric);
    case 'inverse'
        metric = 1./metric;
end
% calculate summary statistic. Note: uncomment lines to add metrics
switch transform
    case 'circ'
        % use circular stats
        warning off
        T.([prefix 'av' suffix]) = circ_rad2ang(circ_mean(metric));
        %         don't calculate the median as the intermediate array size is Ne x NE
        %         T.([prefix 'md' suffix]) = circ_rad2ang(circ_median(metric)); %
        %         T.([prefix 'mn' suffix]) = circ_rad2ang(min(metric));
        %         T.([prefix 'mx' suffix]) = circ_rad2ang(max(metric));
        % T.([prefix 'sd' suffix]) = circ_rad2ang(circ_std(metric));
        %         T.([prefix 'sk' suffix]) = circ_skewness(metric);
        warning on
    otherwise
        T.([prefix 'av' suffix]) = mean(metric).*units;
        % T.([prefix 'md' suffix]) = median(metric).*units;
        %         T.([prefix 'mo' suffix]) = mode(metric).*units;
        %         T.([prefix 'mn' suffix]) = min(metric).*units;
        %         T.([prefix 'mx' suffix]) = max(metric).*units;
        % T.([prefix 'sd' suffix]) = std(metric).*units;
        %         T.([prefix 'sk' suffix]) = skewness(metric);
end
end

function [G_HLD, parent] = fnc_HLD(G_veins, G_polygons, polygon_stats, areole_stats, polygon_LM, bw_polygons, micron_per_pixel)
% set calibration factors
mm = micron_per_pixel./1000;
% construct a binary polygon CC object
PCC.Connectivity = 4;
PCC.ImageSize = size(polygon_LM);
PCC.NumObjects = 1;
PCC.PixelIdxList = {};
% find all the polygons on the boundary
boundary = polygon_LM==0;
boundary = bwareafilt(boundary,[200 inf]);
boundary = imdilate(boundary,ones(3));
[r,c] = find(boundary);
B_polygons = bwselect(bw_polygons,c,r,4);
Bidx = unique(polygon_LM(B_polygons));
Bidx(Bidx==0) = [];
% add in a boundary flag if touching the boundary
G_polygons.Nodes.Boundary(:,1) = 0;
G_polygons.Nodes.Boundary(Bidx,1) = 1;
% select the largest component of the polygon graph
CC = conncomp(G_polygons);
idx = find(CC==1);
G_polygons = subgraph(G_polygons,idx);
% extract the same component from the stats arrays
polygon_stats = polygon_stats(idx);
% set the weights in the vein graph to length
G_veins.Edges.Weight = G_veins.Edges.Length;
% Keep veins from the vein graph that form part of the polygon_graph. These
% will be the boundary edges and any internal tree-like parts of the
% network, but will exclude edges from incomplete polygons on the boundary
% or disconnected polygons. Edges should have Ai and/or
% Aj corresponding to a polygon node ID.
Vidx = ismember(G_veins.Edges.Ai,idx) | ismember(G_veins.Edges.Aj,idx);
G_veins = rmedge(G_veins,find(~Vidx));
% only keep veins that are still connected to the largest component
CC = conncomp(G_veins);
[N,~] = histcounts(CC,max(CC));
[~,idx] = max(N);
G_veins = subgraph(G_veins,find(CC==idx));
% calculate the initial length and MST ratio for the veins
L = sum(G_veins.Edges.Length);
MST = minspantree(G_veins,'method','sparse');
MSTL = sum(MST.Edges.Weight)/L;
% get the number of nodes and edge in the dual graph
nnP = single(numnodes(G_polygons));
neP = single(numedges(G_polygons));
parent = single(zeros(1,nnP));
width_threshold = zeros((2*nnP)-1,1,'single');
node_Boundary = [G_polygons.Nodes{:,'Boundary'}; zeros(nnP-1,1,'single')];
node_Area = [G_polygons.Nodes{:,'Area'}; zeros(nnP-1,1,'single')];
node_Degree = [ones(nnP,1,'single'); zeros(nnP-1,1,'single')];
degree_Asymmetry = zeros(nnP*2-1,1,'single');
area_Asymmetry = zeros(nnP*2-1,1,'single');
node_HS = [ones(nnP,1,'single'); zeros(nnP-1,1,'single')];
subtree_degree_Asymmetry = zeros((2*nnP)-1,1,'single');
subtree_area_Asymmetry = zeros((2*nnP)-1,1,'single');
VTotLen = [repmat(L,nnP,1); zeros(nnP-1,1,'single')];
VTotVol = [repmat(L,nnP,1); zeros(nnP-1,1,'single')];
MSTRatio = [repmat(MSTL,nnP,1); zeros(nnP-1,1,'single')];
% redimension the stat arrays to accomodate all the fused nodes
areole_stats(nnP.*2-1).Area = 0;
polygon_stats(nnP.*2-1).Area = 0;
% order the edges by width in the polygon graph
[W,idx] = sort(G_polygons.Edges{:,'Width'});
% sort the edge nodes and edge name to match the ordered widths
nodei = G_polygons.Edges{idx,'EndNodes'}(:,1);
nodej = G_polygons.Edges{idx,'EndNodes'}(:,2);
% set up a list of the initial edges sorted by width
ET = single([nodei nodej W]);
% start the index for the new node (Nk) to follow on the number of existing
% nodes (nnP)
Nk = nnP;
Ne = 0;
% get all the current polygon PixelIdxLists at the start as a cell array;
P_PIL = {polygon_stats.PixelIdxList};
PCC.NumObjects = length(P_PIL);
PCC.PixelIdxList  = {polygon_stats.PixelIdxList}';
LM = labelmatrix(PCC);
P_stats = regionprops('table',LM,'Area','Centroid','Perimeter','MajorAxisLength','MinorAxisLength','Eccentricity','Orientation');
% set up the endnodes
EndNodes = zeros(nnP*2-2,2,'single');
% loop through all the edges, calculating the metrics
for iE = 1:neP
    % test each edge in sequence
    Ni = ET(1,1);
    Nj = ET(1,2);
    % the edge to be removed only links two areas if
    % the end nodes are different
    if Ni~=Nj
        % create a new node
        Nk = single(Nk+1);
        % get the width threshold for the HLD for this partition
        width_threshold(Nk,1) = ET(1,3);
        % construct the EndNodes for the two new edges that connect Ni and Nj to Nk
        Ne = Ne+1;
        EndNodes(Ne,:) = [Ni Nk];
        Ne = Ne+1;
        EndNodes(Ne,:) = [Nj Nk];
        % check whether the fused area includes a boundary region
        node_Boundary(Nk) = max(node_Boundary(Ni),node_Boundary(Nj));
        % sum the areas of the nodes to fuse
        node_Area(Nk) = node_Area(Ni)+node_Area(Nj);
        % sum the degree of the subtree at each node
        node_Degree(Nk) = node_Degree(Ni)+node_Degree(Nj);
        % find a measure of the partition asymmetry of the
        % bifurcation vertex
        degree_Asymmetry(Nk,1) = abs(node_Degree(Ni)-node_Degree(Nj))/max(node_Degree(Ni),node_Degree(Nj));
        % calculate the subtree asymmetry
        subtree_degree_Asymmetry(Nk,1) = (1/(node_Degree(Nk)-1))*((degree_Asymmetry(Ni)*(node_Degree(Ni)-1))+(degree_Asymmetry(Nj)*(node_Degree(Nj)-1)));
        area_Asymmetry(Nk,1) = abs(node_Area(Ni)-node_Area(Nj))/max(node_Area(Ni),node_Area(Nj));
        % calculate the subtree asymmetry
        subtree_area_Asymmetry(Nk,1) = (1/(node_Area(Nk)-1))*((area_Asymmetry(Ni)*(node_Area(Ni)-1))+(area_Asymmetry(Nj)*(node_Area(Nj)-1)));
        % Strahler index: keep the larger of the Horton-Strahler indices if they
        % are not equal. If they are equal, increment the HS index by one
        if node_HS(Ni) ~= node_HS(Nj)
            node_HS(Nk,1) = max(node_HS(Ni),node_HS(Nj));
        else
            node_HS(Nk,1) = node_HS(Ni)+1;
        end
        % delete the current edge
        ET(1,:) = [];
        % replace any other occurrences of the nodes that have fused with
        % the new node ID
        idx = ET(:,1) == Ni | ET(:,1) == Nj;
        ET(idx,1) = Nk;
        idx = ET(:,2) == Ni | ET(:,2) == Nj;
        ET(idx,2) = Nk;
        % set the parent of the nodes to be fused to the new node
        parent(Ni) = Nk;
        parent(Nj) = Nk;
        % add the PixelIdxList for the nodes that have fused to the new node
        P_PIL(Nk) = {cat(1,P_PIL{Ni},P_PIL{Nj})};
        % calculate the statistics for the new region and add them to the
        % stats structures
        PCC.NumObjects = 1;
        PCC.PixelIdxList  = P_PIL(Nk);
        P_stats(Nk,:) = regionprops('table',PCC,'Area','Centroid','Perimeter','MajorAxisLength','MinorAxisLength','Eccentricity','Orientation');
        % find edges in the vein graph up to and including this edge width
        Eidx = G_veins.Edges.Width <= width_threshold(Nk,1);
        % remove these edges from the graph
        G_veins = rmedge(G_veins,find(Eidx));
        % only keep veins that are still connected to the largest component
        CC = conncomp(G_veins);
        [N,~] = histcounts(CC,max(CC));
        [~,idx] = max(N);
        G_veins = subgraph(G_veins,find(CC==idx));
        % replace any occurrences of the nodes that have fused with
        % the new node ID
        idx = G_veins.Edges.Ai == Ni | G_veins.Edges.Ai == Nj;
        G_veins.Edges.Ai(idx) = Nk;
        idx = G_veins.Edges.Aj == Ni | G_veins.Edges.Aj == Nj;
        G_veins.Edges.Aj(idx) = Nk;
        % calculate the minimum spanning tree using Kruskal's algorithm
        MST = minspantree(G_veins,'method','sparse');
        VTotLen(Nk,1) = sum(G_veins.Edges.Weight);
        VTotVol(Nk,1) = sum(G_veins.Edges.Volume);
        MSTRatio(Nk,1) = sum(MST.Edges.Weight)/VTotLen(Nk,1);
    else
        % delete the current edge as it lies within areas that are already
        % fused
        ET(1,:) = [];
    end
end
% complete links to the root node
parent(end+1) = Nk+1;
parent = double(fliplr(max(parent(:)) - parent));
% calculate additional metrics
Circularity = (4.*pi.*[P_stats.Area])./([P_stats.Perimeter].^2);
Elongation = [P_stats.MajorAxisLength]./[P_stats.MinorAxisLength];
Roughness = ([P_stats.Perimeter].^2)./[P_stats.Area];
% assemble the HLD graph object
NodeTable = table((1:(2*nnP)-1)', width_threshold.*mm, ...
    node_Area.*(mm^2), ...
    node_Degree, ...
    degree_Asymmetry, ...
    subtree_degree_Asymmetry, ...
    area_Asymmetry, ...
    subtree_area_Asymmetry, ...
    node_HS, ...
    [P_stats.Perimeter].*mm, ...
    [P_stats.MajorAxisLength].*mm, ...
    [P_stats.MinorAxisLength].*mm, ...
    [P_stats.Eccentricity], ...
    [P_stats.Orientation], ...
    Circularity, ...
    Elongation, ...
    Roughness, ...
    VTotLen.*mm, ...
    VTotVol.*mm^3, ...
    MSTRatio, ...
    node_Boundary, ...
    'VariableNames',{'node_ID', 'width_threshold', 'node_Area', 'node_Degree', ...
    'degree_Asymmetry',  'subtree_degree_Asymmetry', 'area_Asymmetry',  'subtree_area_Asymmetry', ...
    'node_HS','Perimeter','MajorAxisLength', 'MinorAxisLength', 'Eccentricity','Orientation','Circularity','Elongation','Roughness','VTotLen','VTotVol','MSTRatio','Boundary'});
idx = NodeTable.node_Area == 0;
NodeTable(idx,:) = [];
EdgeTable = table(EndNodes, ...
    'VariableNames', {'EndNodes'});
idx = EdgeTable.EndNodes(:,1)==0;
EdgeTable(idx,:) = [];
idx = EdgeTable.EndNodes(:,2)==0;
EdgeTable(idx,:) = [];
% combine to form the HLD tree graph
G_HLD = graph(EdgeTable, NodeTable);
end

function hfig = display_PR(PR)
methods = {'cnn';'vesselness';'featuretype';'bowlerhat';'midgrey'};%;'niblack';'bernsen';'sauvola'};
cols = {'r-';'g-';'b-';'c-';'m-';'g:';'b:';'c:'};
pnts = {'ro';'go';'bo';'co';'mo';'g*';'b*';'c*'};
mrks = {'.';'.';'.';'.';'.';'.';'.';'.'};
% plot the precision-recall plots and mark the optimum
hfig = figure('Renderer','painters');
hfig.Color = 'w';
for iP = 1:numel(methods)
    h(iP) = plot(PR.results.(methods{iP}).Recall,PR.results.(methods{iP}).Precision,cols{iP},'Marker',mrks{iP});
    hold on
    plot(PR.results.(methods{iP}).Recall(PR.evaluation{methods{iP},'F1_idx'}),PR.results.(methods{iP}).Precision(PR.evaluation{methods{iP},'F1_idx'}),pnts{iP})
end
xlabel('Recall')
ylabel('Precision')
ax = gca;
ax.FontUnits = 'points';
ax.FontSize = 14;
legend(h,methods,'Location','SouthWest')
drawnow
end

function hfig = display_sk_PR(PR)
methods = {'cnn_sk';'vesselness_sk';'featuretype_sk';'bowlerhat_sk';'midgrey_sk'};%;'niblack_sk';'bernsen_sk';'sauvola_sk'};
cols = {'r-';'g-';'b-';'c-';'m-';'g:';'b:';'c:'};
pnts = {'ro';'go';'bo';'co';'mo:';'go';'bo';'co'};
mrks = {'.';'.';'.';'.';'.';'.';'.';'.'};
% plot the precision-recall plots and mark the optimum
hfig = figure('Renderer','painters');
hfig.Color = 'w';
for iP = 1:numel(methods)
    h(iP) = plot(PR.results.(methods{iP}).Recall,PR.results.(methods{iP}).Precision,cols{iP},'Marker',mrks{iP});
    hold on
    plot(PR.results.(methods{iP}).Recall(PR.evaluation{methods{iP},'F1_idx'}),PR.results.(methods{iP}).Precision(PR.evaluation{methods{iP},'F1_idx'}),pnts{iP})
end
xlabel('Recall')
ylabel('Precision')
ax = gca;
ax.FontUnits = 'points';
ax.FontSize = 14;
legend(h,strrep(methods,'_sk',''),'Location','SouthWest')
drawnow
end

function hfig = display_figure(images,graphs,titles,G,E_width,links,name,ExportFigs)
hfig = figure('Renderer','painters');
hfig.Units = 'normalized';
hfig.Position = [0 0 1 1];
hfig.Color = 'w';
inset_sz = 2/5;
inset_zoom = 6;
for ia = 1:6
    ax(ia) = subplot(2,3,ia);
    pos = ax(ia).OuterPosition;
    pos(3) = 0.25;
    pos(4) = 0.4;
    ax(ia).Position = pos;
    ax(ia+6) = axes('Position', [pos(1)+pos(3)*(1-inset_sz) pos(2)+pos(4)*(1-inset_sz) pos(3)*inset_sz pos(4)*inset_sz]);
end
linkaxes(ax(links),'xy')
for ia = 1:6
    axes(ax(ia))
    if ~isempty(images{ia})
        imshow(images{ia},[],'Border','tight','InitialMagnification','fit');
        h = title(titles{ia},'fontsize',12,'fontweight','normal','interpreter','none');
        h.FontWeight = 'normal';
        axis on
        ax(ia).XTick = [];
        ax(ia).YTick = [];
        ax(ia).XColor = 'k';
        ax(ia).YColor = 'k';
        ax(ia).LineWidth = 1;
        % display the inset
        axes(ax(ia+6))
        imshow(images{ia},[]);
        zoom(inset_zoom)
        axis on
        ax(ia+6).XTick = [];
        ax(ia+6).YTick = [];
        ax(ia+6).XColor = 'k';
        ax(ia+6).YColor = 'k';
        ax(ia+6).LineWidth = 1;
    end
end
for ia = 1:6
    if ~strcmp(graphs{ia},'none')
        % get the edge parameter to display
        E = G.Edges.(graphs{ia});
        E(isinf(E)) = nan;
        % Normalise the metric between min and max and generate an index
        % into the colourmap
        E_min = min(E,[], 'omitnan');
        E_max = max(E,[], 'omitnan');
        % normalise the range between 2 and 256 as an index into the
        % colourmap
        E_idx = ceil(254.*((E-E_min)./(E_max-E_min)))+1;
        % set any nan values to 1 in the index
        E_idx(isnan(E_idx)) = 1;
        % the index should be real numbers normalised between 1 and 256 at
        % this point.
        cmap = jet(256);
        cmap(1,:) = 0;
        E_color =  cmap(E_idx,:);
        % plot the graph
        axes(ax(ia))
        hold on
        plot(G, 'XData',G.Nodes.node_X_pix,'YData',G.Nodes.node_Y_pix, ...
            'NodeColor','g','MarkerSize',1,'Marker', 'none', 'NodeLabel', [], ...
            'EdgeColor',E_color,'EdgeAlpha',1,'EdgeLabel', [],'LineWidth',E_width);
        axes(ax(ia+6))
        hold on
        plot(G, 'XData',G.Nodes.node_X_pix,'YData',G.Nodes.node_Y_pix, ...
            'NodeColor','g','MarkerSize',1,'Marker', 'none', 'NodeLabel', [], ...
            'EdgeColor',E_color,'EdgeAlpha',1,'EdgeLabel', [],'LineWidth',1);
    end
end
drawnow
if ExportFigs
    export_fig(name,'-png','-r300',hfig)
    %     saveas(hfig,name)
end
end

function hfig = display_HLD(G_polygons,im_cnn,G_HLD,FullLeaf,name,ExportFigs)
hfig = figure('Renderer','painters');
hfig.Units = 'normalized';
hfig.Position = [0 0 0.6 1];
hfig.Color = 'w';
% display the CNN image to overlay the subgraphs
ax(1) = subplot(3,3,4);
axes(ax(1))
pos = ax(1).OuterPosition;
ax(1).Position = pos;
imshow(1-im_cnn,[])
hold on
axis off
box on
% display the complete treeplot
subplot(3,3,[1,3])
h = plot(G_HLD,'Layout','layered','Direction','down','sources',numnodes(G_HLD), ...
    'AssignLayers','alap','NodeColor','k','Marker','.','EdgeColor','k','LineWidth',1);
% recolor any nodes connected to the boundary in grey
SP = shortestpathtree(G_HLD,numnodes(G_HLD),G_HLD.Nodes.node_ID(find(G_HLD.Nodes.Boundary)));
highlight(h,SP,'NodeColor',[0.5 0.5 0.5],'EdgeColor',[0.5 0.5 0.5],'Marker','*','LineWidth',1)
ylabel('node level');
xlabel('terminal node');
axis tight
axis on
xlim([0 numnodes(G_polygons)])
y = ylim;
ylim([0 y(2)])
box on
% display the tree width histogram
subplot(3,3,5)
nbins = 50;
vv = G_HLD.Nodes.width_threshold;
ww = G_HLD.Nodes.node_Area;
% get automatic bin limits using the histogram function
[~,edges] = histcounts(vv,nbins);
% calculate a weighted histogram
[histw, ~] = histwv(vv, ww, min(edges), max(edges), nbins);
% plot the histogram against the bin centers
center = edges(2:end)-diff(edges(1:2))./2;
bar(center(2:end),histw(2:end),'FaceColor',[0.5 0.5 0.5])
xlabel('width of edge removed')
ylabel('area weighted freq.')
box on
if FullLeaf == 1
    last = 1;
    g1 = G_HLD;
else
    % extract the subgraph for nodes that are not linked to a boundary node
    g1 = subgraph(G_HLD,find(~G_HLD.Nodes.Boundary));
    % extract the largest five fully connected subgraphs
    last = 5;
end
% order by the largest connected subtree
cc = conncomp(g1,'OutputForm','cell');
l = cellfun(@(x) length(x),cc);
[~,idx] = sort(l,'descend');
% set up plot options
cols = repmat({'r','g','b','c','m','y'},1,50);
width_limits = ([round(min(log(g1.Nodes.width_threshold(g1.Nodes.width_threshold>0)))-0.1,1) round(max(log(g1.Nodes.width_threshold))+0.1,1)]);
area_limits = ([floor(min(log(g1.Nodes.node_Area))-0.25) ceil(max(log(g1.Nodes.node_Area))+0.25)]);
for icc = 1:last
    % extract the HLD subgraph
    g2 = subgraph(g1,cc{idx(icc)});
    % recolor the nodes and edges
    highlight(h,g2.Nodes.node_ID(g2.Edges.EndNodes(:,1)),g2.Nodes.node_ID(g2.Edges.EndNodes(:,2)),'NodeColor',cols{icc},'Marker','o','EdgeColor',cols{icc},'LineWidth',1)
    % get the node-IDs for the subtree to overlay on the image
    Gidx = g1.Nodes.node_ID(cc{idx(icc)});
    % limit the nodes to display to the initial nodes
    Gidx(Gidx>numnodes(G_polygons)) = [];
    % extract the subgraph from the original dual-graph that contain the selected nodes
    g3 = subgraph(G_polygons,Gidx);
    % display the subgraph on the image
    plot(ax(1),g3, 'XData',g3.Nodes.node_X_pix,'YData',g3.Nodes.node_Y_pix, ...
        'NodeColor','g','MarkerSize',1,'Marker', 'none', 'NodeLabel', [], ...
        'EdgeColor',cols{icc},'EdgeAlpha',1,'EdgeLabel', [],'LineWidth',1);
    
    % plot the strahler bifurcation ratio
    maxX = max(g1.Nodes.node_HS);
    Bifurcation_ratio = [];
    for iHS = 1:max(g2.Nodes.node_HS)-1
        Bifurcation_ratio(iHS) = sum(g2.Nodes.node_HS==iHS)/sum(g2.Nodes.node_HS==iHS+1);
    end
    subplot(3,3,6)
    plot(Bifurcation_ratio+icc.*.1,'Color',cols{icc},'Marker','o','Linestyle', '-')
    hold on
    xlim([0 maxX])
    ylim([0 8])
    xlabel('Strahler number')
    ylabel('bifurcation ratio')
    box on
    
    % calculate the cumulative size distribution
    a = sort(g2.Nodes.node_Area,'ascend');
    csN = cumsum(1:numnodes(g2));
    Pa = 1-(csN/max(csN));
    subplot(3,3,7)
    plot(log(a),log(a.*Pa'), 'Color',cols{icc},'Marker','o','Linestyle', '-')
    xlabel('log(area)')
    ylabel('log Pa*area')
    xlim(area_limits)
    %ylim([-6 0])
    hold on
    box on
    
    % plot the MSTRatio
    ax(8) = subplot(3,3,8);
    plot(ax(8),log(g2.Nodes.width_threshold), g2.Nodes.MSTRatio,'LineStyle','none','Marker','.','Color',cols{icc})
    xlabel('log(width)')
    ylabel('MST ratio')
    xlim(width_limits)
    ylim([0.5 1])
    hold on
    box on
    
    % plot the Circularity
    ax(9) = subplot(3,3,9);
    plot(ax(9),log(g2.Nodes.width_threshold), 4*pi*g2.Nodes.node_Area./g2.Nodes.Perimeter.^2,'LineStyle','none','Marker','.','Color',cols{icc})
    xlabel('log(width)')
    ylabel('circularity')
    xlim(width_limits)
    ylim([0.2 1])
    hold on
    box on
    
end
drawnow
if ExportFigs
    export_fig(name,'-png','-r300','-painters',hfig)
    %     saveas(hfig,name)
end
end

function hfig = display_HLD_figure(G_polygons,im_cnn,G_HLD,FullLeaf,name,ExportFigs)
hfig = figure('Renderer','painters');
hfig.Units = 'normalized';
hfig.Position = [0 0 1 0.5];
hfig.Color = 'w';
% set up the subplots
for ia = 1:3
    ax(ia) = subplot(1,3,ia);
    pos = ax(ia).OuterPosition;
    ax(ia).Position = pos.*[1 1 0.95 1];
end
% display the CNN image to overlay the subgraphs
axes(ax(1))
imshow(1-im_cnn,[])
axis on
ax(1).XTick = [];
ax(1).YTick = [];
hold on
% display the complete treeplot
subplot(1,3,[2,3])
h = plot(G_HLD,'Layout','layered','Direction','down','sources',numnodes(G_HLD), ...
    'AssignLayers','alap','NodeColor','k','Marker','.','EdgeColor','k','LineWidth',1);
% recolor any nodes connected to the boundary in grey
SP = shortestpathtree(G_HLD,numnodes(G_HLD),G_HLD.Nodes.node_ID(find(G_HLD.Nodes.Boundary)));
highlight(h,SP,'NodeColor',[0.5 0.5 0.5],'EdgeColor',[0.5 0.5 0.5],'Marker','*','LineWidth',1)
ylabel('node level');
xlabel('terminal node');
axis tight
axis on
xlim([0 numnodes(G_polygons)])
y = ylim;
ylim([0 y(2)])
box on
if FullLeaf == 1
    last = 1;
    g1 = G_HLD;
else
    % extract the subgraph for nodes that are not linked to a boundary node
    g1 = subgraph(G_HLD,find(~G_HLD.Nodes.Boundary));
    % extract the largest five fully connected subgraphs
    last = 5;
end
% order by the largest connected subtree
cc = conncomp(g1,'OutputForm','cell');
l = cellfun(@(x) length(x),cc);
[~,idx] = sort(l,'descend');
% set up plot options
cols = repmat({'r','g','b','c','m','y'},1,50);
area_limits = ([floor(min(log(g1.Nodes.node_Area))-0.25) ceil(max(log(g1.Nodes.node_Area))+0.25)]);
% extract the largest five fully connected subgraphs
for icc = 1:last
    % extract the HLD subgraph
    g2 = subgraph(g1,cc{idx(icc)});
    % recolor the nodes and edges
    highlight(h,g2.Nodes.node_ID(g2.Edges.EndNodes(:,1)),g2.Nodes.node_ID(g2.Edges.EndNodes(:,2)),'NodeColor',cols{icc},'Marker','o','EdgeColor',cols{icc},'LineWidth',1)
    % get the node-IDs for the subtree to overlay on the image
    Gidx = g1.Nodes.node_ID(cc{idx(icc)});
    % limit the nodes to display to the initial nodes
    Gidx(Gidx>numnodes(G_polygons)) = [];
    % extract the subgraph from the original dual-graph that contain the selected nodes
    g3 = subgraph(G_polygons,Gidx);
    % display the subgraph on the image
    plot(ax(1),g3, 'XData',g3.Nodes.node_X_pix,'YData',g3.Nodes.node_Y_pix, ...
        'NodeColor','g','MarkerSize',1,'Marker', 'none', 'NodeLabel', [], ...
        'EdgeColor',cols{icc},'EdgeAlpha',1,'EdgeLabel', [],'LineWidth',1);
end
drawnow
if ExportFigs
    export_fig(name,'-png','-r300','-painters',hfig)
    %     saveas(hfig,name)
end
end

function [histw, histv] = histwv(v, w, min, max, bins)
% code from Brent
%Inputs:
%vv - values
%ww - weights
%minV - minimum value
%maxV - max value
%bins - number of bins (inclusive)

%Outputs:
%histw - weighted histogram
%histv (optional) - histogram of values

delta = (max-min)/(bins-1);
subs = round((v-min)/delta)+1;

histv = accumarray(subs,1,[bins,1]);
histw = accumarray(subs,w,[bins,1]);
end


