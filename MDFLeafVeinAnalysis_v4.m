% %function results = MDFLeafVeinAnalysisv2(FolderName,micron_per_pixel,DownSample,ShowFigs,ExportFigs)
% %% set up directories
% dir_out_images = ['..' filesep 'summary' filesep 'images' filesep];
% dir_out_width = ['..' filesep 'summary' filesep 'width' filesep];
% dir_out_data = ['..' filesep 'summary' filesep 'data' filesep];
% dir_out_HLD = ['..' filesep 'summary' filesep 'HLD' filesep];
% dir_out_PR_results = ['..' filesep 'summary' filesep 'PR' filesep 'results' filesep];
% dir_out_PR_images = ['..' filesep 'summary' filesep 'PR' filesep 'images' filesep];
% dir_out_PR_graphs = ['..' filesep 'summary' filesep 'PR' filesep 'graphs' filesep];
% dir_out_PR_fig = ['..' filesep 'summary' filesep 'PR' filesep 'fig' filesep];
% %% set up parameters
% micron_per_pixel = micron_per_pixel.*DownSample;
% sk_width = 3;
% E_width = 2;
% cmap = jet(256);
% cmap(1,:) = 0;
% step = 0;
% warning off
% %% Load in the images
% step = step+1;
% disp(['Step ' num2str(step) ': Processing ' FolderName])
% [im,im_cnn,bw_mask,bw_vein,bw_roi,bw_GT] = fnc_load_CNN_images(FolderName,DownSample);
% %% test the performance against the internal ground truth
% if any(bw_GT(:))
%     step = step+1;
%     disp(['Step ' num2str(step) ': Precision-Recall analysis'])
%     fout = [dir_out_PR_fig FolderName];
%     PR = fnc_precision_recall(im,im_cnn,bw_mask,bw_vein,bw_roi,bw_GT,DownSample,fout);
%     if ShowFigs == 1
%         % display the PR curves for the full-width binary image and the
%         % skeleton
%         display_PR(PR, FolderName);
%         display_sk_PR(PR, FolderName);
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
%         hfig1 = display_PR(PR, FolderName);
%         saveas(hfig1,[dir_out_PR_graphs FolderName '_PRC.png'])
%         delete(hfig1);
%         % save the PR skeleton curve
%         hfig2 = display_sk_PR(PR, FolderName);
%         saveas(hfig2,[dir_out_PR_graphs FolderName '_sk_PRC.png'])
%         delete(hfig2);
%     end
% else
%     PR = [];
% end
% %% get the skeleton
% step = step+1;
% disp(['Step ' num2str(step) ': Skeleton extraction'])
% if exist('PR','var')
%     % use the threshold value determined from the PR analysis
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
% % % % %% calculate the width using granulometry
% % % % step = step+1;
% % % % disp(['Step ' num2str(step) ': Calculating width using granulometry'])
% % % % %im_width = fnc_granulometry(imcomplement(mat2gray(im)),DownSample);
% % % % im_width = fnc_granulometry(im_cnn,DownSample);
% % % % W_pixels = zeros(size(im_width),'single');
% % % % W_pixels(sk) = single(im_width(sk).*2);
% %% set up the segmentation figure
% if ShowFigs == 1
%     warning off images:initSize:adjustingMag
%     warning off MATLAB:LargeImage
%     images = {im,im_cnn,bw_mask,imdilate(single(cat(3,skLoop,skTree,skLoop)), ones(3)),im_distance,imdilate(W_pixels, ones(3))};
%     graphs = {'none','none','none','none','none','none'};
%     titles = {'original','CNN','Mask','Skeleton','Distance','Width (pixels)'};
%     display_figure(images,graphs,titles,[],E_width,[1:6],'Segmentation',ExportFigs);
% end
% 
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
% [edgelist, edgeim] = fnc_resolve_duplicates(edgelist,edgeim);
% %% construct the weighted graph
% step = step+1;
% disp(['Step ' num2str(step) ': Weighted graph'])
% [G_veins,edgelist] = fnc_weighted_graph(edgelist,W_pixels,skLoop,skTree);
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
% if ShowFigs == 1
%     warning off images:initSize:adjustingMag
%     warning off MATLAB:LargeImage
%     if exist('PR','var')
%         images = {im_cnn,im_cnn,CW_pixels,coded,PR.images.cnn(:,:,:,PR.evaluation{'cnn','F1_idx'}),PR.images.cnn_sk(:,:,:,PR.evaluation{'cnn_sk','F1_idx'})};
%     else
%         images = {im_cnn,im_cnn,CW_pixels,coded,[],[]};
%     end
%     graphs = {'Width_initial','Width','none','none','none','none'};
%     titles = {'Original width','center-weighted width','width (pixels)','coded','precision-recall full width','precision-recall skeleton'};
%     display_figure(images,graphs,titles,G_veins,E_width,[1:4],'Width',ExportFigs);
% end
% if ExportFigs == 1
%     step = step+1;
%     disp(['Step ' num2str(step) ': Saving width images'])
%     [nY,nX,nC] = size(coded);
%     % save the color-coded width image
%     fout = [dir_out_images FolderName '_width.png'];
%     imwrite(coded,fout,'png','Xresolution',nX,'Yresolution',nY)
%     % save the greyscale width array as a matlab file. Note outside the
%     % masked area is now coded as -1
%     save([dir_out_width FolderName '_Width_array'],'im_width')
% %     % save the greyscale width image as a uint8 image
% %     fout = [dir_out_width FolderName '_width_gray.png'];
% %     imwrite(uint8(im_width),fout,'png','Xresolution',nX,'Yresolution',nY)
% end
% % find the areoles
% step = step+1;
% disp(['Step ' num2str(step) ': polygon analysis'])
% % find the polygon and areole areas
% [G_veins, sk_polygon, bw_polygons, bw_areoles, total_area_mask, polygon_LM] = fnc_polygon_find(G_veins,bw_cnn,sk,skLoop,bw_mask);
% [areole_stats,polygon_stats] = fnc_polygon_analysis(bw_polygons,bw_areoles, polygon_LM);
% % construct color-coded images based on log area
% im_polygons_rgb = fnc_polygon_image(polygon_stats, sk_polygon, total_area_mask);
% im_areoles_rgb = fnc_polygon_image(areole_stats, sk_polygon, total_area_mask);
% % convert to an areole graph and a polygon graph
% step = step+1;
% disp(['Step ' num2str(step) ': Dual graph'])
% [G_areoles] = fnc_area_graph(G_veins,areole_stats);
% [G_polygons] = fnc_area_graph(G_veins,polygon_stats);
% % collect summary statistics into a results array
% step = step+1;
% disp(['Step ' num2str(step) ': Summary statistics'])
% total_area = sum(bw_mask(:));
% polygon_area = sum(G_polygons.Nodes.Area);
% veins = fnc_summary_veins(G_veins,total_area,polygon_area,micron_per_pixel);
% areoles = fnc_summary_areoles(G_areoles,polygon_area,micron_per_pixel);
% polygons = fnc_summary_polygons(G_polygons,micron_per_pixel);
% results = [veins areoles polygons];
% %% add in results for the ROC analysis (if present)
% if exist('PR','var')
%     results.PR_threshold = PR.evaluation{'cnn_sk','F1_threshold'};
%     results.PR_Precision = PR.results.cnn.Precision(PR.evaluation{'cnn_sk','F1_idx'});
%     results.PR_Recall = PR.results.cnn.Recall(PR.evaluation{'cnn_sk','F1_idx'});
%     results.PR_F = PR.evaluation{'cnn_sk','F1'};
%     results.PR_FBeta2 = PR.evaluation{'cnn_sk','FBeta2'};
% end
% %% add in file information
% results.File = FolderName;
% results.TimeStamp = datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss Z');
% results.MicronPerPixel = micron_per_pixel;
% results.DownSample = DownSample;
% % reorder the table to get the file info first
% results = results(:, [end-3:end 1:end-4]);
% %% save the graphs
% step = step+1;
% disp(['Step ' num2str(step) ': Saving graphs data'])
% % save the results
% save([dir_out_data FolderName '_Graphs.mat'],'G_veins','G_areoles','G_polygons','results')
% %% set up the results figure
% if ShowFigs == 1
%     warning off images:initSize:adjustingMag
%     warning off MATLAB:LargeImage
%     images = {im_polygons_rgb,im_areoles_rgb,im_cnn,[],[],[]};
%     graphs = {'none','none','Width','none','none','none'};
%     titles = {'polygons','areoles','dual graph','none','none','none'};
%     display_figure(images,graphs,titles,G_polygons,E_width,[1:3],'Polygons',ExportFigs);
% end
% % set up the figure for paper
% if ShowFigs == 1
%     warning off images:initSize:adjustingMag
%     warning off MATLAB:LargeImage
%     images = {im,im_cnn,imdilate(single(cat(3,skLoop,skTree,skLoop)), ones(3)),coded,im_areoles_rgb,im_cnn};
%     graphs = {'none','none','none','none','none','Width'};
%     titles = {'original','CNN','Skeleton','width','areoles','dual graph'};
%     display_figure(images,graphs,titles,G_polygons,E_width,[1:6],'Figure',ExportFigs);
% end
% 
%% Hierarchical loop decomposition
step = step+1;
disp(['Step ' num2str(step) ': Hierarchical loop decomposition'])
[HLD_levels, G_HLD, parent, HLD_metrics, im_HLD_order] = fnc_HLD(G_veins, G_polygons, G_areoles, polygon_stats, areole_stats, bw_polygons, bw_areoles, total_area, polygon_area, micron_per_pixel);
% HLD display
if ShowFigs == 1
    display_HLD(G_polygons,im_cnn,HLD_levels,im_HLD_order,G_HLD,parent);
end
if ExportFigs == 1
    hfig = display_HLD(G_polygons,im_cnn,HLD_levels,im_HLD_order,G_HLD,parent);
    saveas(hfig,[dir_out_HLD FolderName '_HLD.png'])
    delete(hfig);
    save([dir_out_HLD FolderName '_HLD_results.mat'],'G_HLD','parent','HLD_metrics')
end
% % % %% HLD slices
% % % step = step+1;
% % % disp(['Step ' num2str(step) ': HLD slices'])
% % % [HLD_results,HLD_slices] = fnc_HLD_slices(G_veins,bw_cnn,bw_mask,total_area_mask,im_width,HLD_levels,micron_per_pixel);
% % % %% Save the HLD data
% % % if ExportFigs == 1
% % %     step = step+1;
% % %     disp(['Step ' num2str(step) ': saving HLD data'])
% % %     % save the slice images as a matlab file
% % %     save([dir_out_HLD FolderName '_HLD_slices.mat'],'HLD_slices')
% % %     % save the slice data
% % %     writetable(HLD_results,[dir_out_HLD FolderName '_HLD_results.xlsx'],'FileType','Spreadsheet','Range', 'A1','WriteVariableNames',1)
% % %     if isunix
% % %         fileattrib([dir_out_HLD FolderName '_HLD_results.xlsx'], '+w','a')
% % %     end
% % % end
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

function PR = fnc_precision_recall(im,im_cnn,bw_mask,bw_vein,bw_roi,bw_GT,DownSample,fout)
% subsample images for comparison with ground truth
stats = regionprops(bw_roi,'BoundingBox');
BB = round(stats.BoundingBox);
roi = bw_roi(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
roi_im = im(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
roi_cnn = im_cnn(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
roi_mask = bw_mask(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
roi_vein = bw_vein(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
roi_GT = bw_GT(BB(2):BB(2)+BB(4)-1,BB(1):BB(1)+BB(3)-1);
% keep the largest connected component
roi_GT = bwareafilt(roi_GT,1);
% convert to [0 1] and enhance images
roi_im  = fnc_enhance_im(roi_im,DownSample,'im');
roi_cnn = fnc_enhance_im(roi_cnn,DownSample,'CNN');
roi_vesselness = fnc_enhance_im(roi_im,DownSample,'Vesselness');
roi_featuretype = fnc_enhance_im(roi_im,DownSample,'FeatureType');
roi_bowlerhat = fnc_enhance_im(roi_im,DownSample,'BowlerHat');
% convert to a full-width binary image
Tmin = 0;
Tmax = 1;
nT = 20;
Tint = (Tmax-Tmin)/(nT);
T = Tmin:Tint:Tmax-Tint;
bw_cnn_roi = fnc_im_to_bw(roi_cnn,roi,roi_vein,roi_mask,T,'hysteresis');
bw_niblack = fnc_im_to_bw(roi_im,roi,roi_vein,roi_mask,T,'Niblack');
bw_midgrey = fnc_im_to_bw(roi_im,roi,roi_vein,roi_mask,T,'midgrey');
bw_bernsen = fnc_im_to_bw(roi_im,roi,roi_vein,roi_mask,T,'Bernsen');
bw_sauvola = fnc_im_to_bw(roi_im,roi,roi_vein,roi_mask,T,'Sauvola');
bw_vesselness = fnc_im_to_bw(roi_vesselness,roi,roi_vein,roi_mask,T,'hysteresis');
bw_featuretype = fnc_im_to_bw(roi_featuretype,roi,roi_vein,roi_mask,T,'hysteresis');
bw_bowlerhat = fnc_im_to_bw(roi_bowlerhat,roi,roi_vein,roi_mask,T,'hysteresis');
% evaluate performance using Precision-Recall analysis
[PR.results.cnn, PR.images.cnn] = fnc_PRC_bw(roi_GT,bw_cnn_roi,T);
[PR.results.niblack, PR.images.niblack] = fnc_PRC_bw(roi_GT,bw_niblack,T);
[PR.results.midgrey, PR.images.midgrey] = fnc_PRC_bw(roi_GT,bw_midgrey,T);
[PR.results.bernsen, PR.images.bernsen] = fnc_PRC_bw(roi_GT,bw_bernsen,T);
[PR.results.sauvola, PR.images.sauvola] = fnc_PRC_bw(roi_GT,bw_sauvola,T);
[PR.results.vesselness, PR.images.vesselness] = fnc_PRC_bw(roi_GT,bw_vesselness,T);
[PR.results.featuretype, PR.images.featuretype] = fnc_PRC_bw(roi_GT,bw_featuretype,T);
[PR.results.bowlerhat, PR.images.bowlerhat] = fnc_PRC_bw(roi_GT,bw_bowlerhat,T);
% convert to a skeleton
sk_GT = fnc_im_to_sk(roi_GT,roi_vein,roi,T,'GT');
sk_cnn = fnc_im_to_sk(roi_cnn,roi_vein,roi,T,'cnn');
sk_niblack = fnc_im_to_sk(bw_niblack,roi_vein,roi,T,'niblack');
sk_midgrey = fnc_im_to_sk(bw_midgrey,roi_vein,roi,T,'midgrey');
sk_bernsen = fnc_im_to_sk(bw_bernsen,roi_vein,roi,T,'bernsen');
sk_sauvola = fnc_im_to_sk(bw_sauvola,roi_vein,roi,T,'sauvola');
sk_vesselness = fnc_im_to_sk(roi_vesselness,roi_vein,roi,T,'vesselness');
sk_featuretype = fnc_im_to_sk(roi_featuretype,roi_vein,roi,T,'featuretype');
sk_bowlerhat = fnc_im_to_sk(roi_bowlerhat,roi_vein,roi,T,'bowlerhat');
% get the skeleton using triangulation
% contourThreshold = 0.5;
% [xcc, ycc] = sk_by_triangulation(roi_cnn, sk_cnn, contourThreshold);
% compare the skeletons with the skeleton ground truth within a given tolerance (in pixels)
tolerance = 3;
[PR.results.cnn_sk,PR.images.cnn_sk] = fnc_PRC_sk(sk_GT,sk_cnn,tolerance,T);
[PR.results.vesselness_sk,PR.images.vesselness_sk] = fnc_PRC_sk(sk_GT,sk_vesselness,tolerance,T);
[PR.results.featuretype_sk,PR.images.featuretype_sk] = fnc_PRC_sk(sk_GT,sk_featuretype,tolerance,T);
[PR.results.bowlerhat_sk,PR.images.bowlerhat_sk] = fnc_PRC_sk(sk_GT,sk_bowlerhat,tolerance,T);
[PR.results.midgrey_sk,PR.images.midgrey_sk] = fnc_PRC_sk(sk_GT,sk_midgrey,tolerance,T);
[PR.results.niblack_sk,PR.images.niblack_sk] = fnc_PRC_sk(sk_GT,sk_niblack,tolerance,T);
[PR.results.bernsen_sk,PR.images.bernsen_sk] = fnc_PRC_sk(sk_GT,sk_bernsen,tolerance,T);
[PR.results.sauvola_sk,PR.images.sauvola_sk] = fnc_PRC_sk(sk_GT,sk_sauvola,tolerance,T);
% get the threshold for the best performance
PR = fnc_PRC_evaluation(PR,T);
% display the figure
hfig = figure;
for ia = 1:18
    ax(ia) = subplot(3,6,ia);
    axes(ax(ia))
    pos = ax(ia).OuterPosition;
    ax(ia).Position = pos;
    ax(ia).XTick = [];
    ax(ia).YTick = [];
end
hfig.Units = 'normalized';
hfig.Position = [0 0 0.8 1];
hfig.Color = 'w';

methods = {'cnn';'vesselness';'featuretype';'bowlerhat';'midgrey'};
% subplot(3,6,1)
axes(ax(1))
if size(roi_im,1) > size(roi_im,2)
    rotate_angle = 90;
else
    rotate_angle = 0;
end

imshow(imrotate(roi_im,rotate_angle),[])
title('original')
axis off
export_fig('roi_im','-png',ax(1))
for iP = 1:4
    % subplot(3,6,iP+1)
    axes(ax(iP+1))
    imshow(imrotate(eval(['roi_' methods{iP}]),rotate_angle),[])
    title(methods{iP})
    axis off
    export_fig(methods{iP},'-png',ax(iP+1))
end
% subplot(3,6,6)
axes(ax(6))
imshow(imrotate(bw_midgrey(:,:,PR.evaluation{'midgrey','F1_idx'}),rotate_angle),[])
title('midgrey')
axis off
export_fig('midgrey','-png',ax(6))

% subplot(3,6,7)
axes(ax(7))
imshow(imrotate(roi_GT,rotate_angle),[])
%title('ground truth')
axis off
export_fig('GT','-png',ax(7))
for iP = 1:5
    % subplot(3,6,iP+7)
    axes(ax(iP+7))
    imshow(imrotate(PR.images.cnn(:,:,:,PR.evaluation{methods{iP},'F1_idx'}),rotate_angle),[])
    %title(round(PR.evaluation{methods{iP},'F1'},2,'significant'))
    axis off
    export_fig([methods{iP} '_2'],'-png',ax(iP+7))
end
methods = {'cnn_sk';'vesselness_sk';'featuretype_sk';'bowlerhat_sk';'midgrey_sk'};
% subplot(3,6,13)
axes(ax(13))
imshow(imdilate(imrotate(sk_GT,rotate_angle), ones(3)),[])
axis off
export_fig(sk_GT,'-png',ax(13))

for iP = 1:5
    % subplot(3,6,iP+13)
    axes(ax(iP+13))
    imshow(imerode(imrotate(PR.images.cnn_sk(:,:,:,PR.evaluation{methods{iP},'F1_idx'}),rotate_angle),ones(3)),[])
    %title(round(PR.evaluation{methods{iP},'F1'},2,'significant'))
    axis off
    export_fig(methods{iP},'-png',ax(iP+13))
end

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

export_fig('all','-native','-png')
linkaxes
drawnow
saveas(hfig,[fout '_PRall.fig'])
delete(hfig)

end

function im_out = fnc_enhance_im(im_in,DownSample,method)
switch method
    case {'CNN'}
        im_out = mat2gray(im_in);
    case {'im';'midgrey';'Niblack';'Bernsen';'Sauvola';'Singh'}
        im_out = imcomplement(mat2gray(im_in));
    case {'Vesselness';'FeatureType';'BowlerHat'} % use an image pyramid to span scales
        % find the scales for the filter
        minW = floor(9/DownSample);
        maxW = ceil(150/DownSample);
        % set up an image pyramid
        nLevels = 5;
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

function sk_out = fnc_im_to_sk(im_in,roi_vein,roi,T,method)
% calculate the skeleton
nT = length(T);
sk_out = false([size(im_in,1) size(im_in,2) nT]);
im_in = mat2gray(im_in);
if ~isempty(roi_vein)
    im_in = max(im_in,roi_vein);
end
switch method
    case {'cnn';'vesselness';'featuretype';'bowlerhat'}
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
    case {'cnn';'vesselness';'featuretype';'bowlerhat'}
        % apply the threshold during the skeletonisation of the enhanced
        % image
        for iT = 1:nT
            [~, sk_out(:,:,iT),~,~] = fnc_skeleton(im_in,roi_vein,T(iT));
        end
    case {'niblack';'midgrey';'bernsen';'sauvola'}
        % use the binary image calculated at different threshold values
        for iT = 1:nT
            [~, sk_out(:,:,iT),~,~] = fnc_skeleton(im_in(:,:,iT),roi_vein,T(iT));
        end
end
end

function [xcc, ycc] = sk_by_triangulation(im_in, sk_in, contourThreshold)
% Get an isocontour
[Lines,Vertices,Objects] = isocontour(im_in,contourThreshold);
Vertices = fliplr(Vertices); % Get it back in XY from IJ
% reduce the number of vertices by setting the tolerance to 0.2 degree arc
[x,y] = reducem(Vertices(:,1),Vertices(:,2),0.2);
% Triangulate all pts in the isocontour and check which trias are in/out
dt = delaunayTriangulation(x,y);
%dt = delaunayTriangulation(x,y);
fc = dt.incenter;
inside = interp2(im_in, fc(:,1), fc(:,2))>=contourThreshold;
% Construct a triangulation to represent the domain triangles.
tr = triangulation(dt(inside, :), dt.Points);
% Construct a set of edges that join the circumcenters of neighboring
% triangles; the additional logic constructs a unique set of such edges.
numt = size(tr,1);
T = (1:numt)';
neigh = tr.neighbors();
cc = tr.circumcenter();
xcc = cc(:,1);
ycc = cc(:,2);
idx1 = T < neigh(:,1);
idx2 = T < neigh(:,2);
idx3 = T < neigh(:,3);
neigh = [T(idx1) neigh(idx1,1); T(idx2) neigh(idx2,2); T(idx3) neigh(idx3,3)]';
figure
imshow(imoverlay(im_in,sk_in(:,:,10),'b'),[])
hold on
%clf;
triplot(tr, 'g');
hold on
plot(xcc(neigh), ycc(neigh), '-r', 'LineWidth', 1);
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
    %     % calculate true positives present in GT and binary image
    %     TP = sk_in(:,:,iT) & sk_GT;
    %     % false positives are in the binary image; but not GT
    %     FP = sk_in(:,:,iT) & ~sk_GT;
    %     % false negatives are in the ground truth, but not the
    % %     % binary image
    %     FN = sk_GT & ~sk_in(:,:,iT);
    %     % true negatives are all the pixels that are not in either
    %     TN = ~sk_GT & ~sk_in(:,:,iT);
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

function PR = fnc_PRC_evaluation(PR,thresholds)
methods = fieldnames(PR.results);
nM = numel(methods);
PR.evaluation = array2table(zeros(nM,6,'single'));
PR.evaluation.Properties.VariableNames = { ...
    'F1' 'F1_idx' 'F1_threshold' ...
    'FBeta2' 'FBeta2_idx' 'FBeta2_threshold'};
PR.evaluation.Properties.RowNames = methods;
for iM = 1:nM
    method = methods{iM};
    switch method
        case {'Niblack';'midgrey';'Bernsen';'Sauvola'}
            T = thresholds-0.5; % allow the offset to run from -0.5 to 0.5
        otherwise
            T = thresholds;
    end
    [PR.evaluation{iM,'F1'}, PR.evaluation{iM,'F1_idx'}] = max(PR.results.(methods{iM}).F1);
    PR.evaluation{iM,'F1_threshold'} = T(PR.evaluation{iM,'F1_idx'});
    [PR.evaluation{iM,'FBeta2'}, PR.evaluation{iM,'FBeta2_idx'}] = max(PR.results.(methods{iM}).FBeta2);
    PR.evaluation{iM,'FBeta2_threshold'} = T(PR.evaluation{iM,'FBeta2_idx'});
end
end

function [bw_cnn, skfinal, skLoop, skTree] = fnc_skeleton(im_in,bw_vein,threshold)
warning off
if islogical(im_in)
    % the input image is alread a binary image
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

function [width] = fnc_granulometry(im_cnn,DownSample)
s = 0:round(90/DownSample);
imo = zeros([size(im_cnn),length(s)], 'single');
for i=1:length(s)
    imo(:,:,i) = imopen(mat2gray(im_cnn),strel('disk',s(i)));
end
width = sum(imo,3);
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

function [G_veins,edgelist] = fnc_weighted_graph(edgelist,W_pixels,skLoop,skTree)
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
% % %             % get the minimum edge weight for nodei and nodej
% % %             NMinI = G_veins.Nodes{G_veins.Edges.EndNodes(:,1),'node_Min'};
% % %             NMinJ = G_veins.Nodes{G_veins.Edges.EndNodes(:,2),'node_Min'};
% get the maximum edge weight for nodei and nodej
NMajI = G_veins.Nodes{G_veins.Edges.EndNodes(:,1),'node_Maj'};
NMajJ = G_veins.Nodes{G_veins.Edges.EndNodes(:,2),'node_Maj'};
% Get the penultimate edge weight for nodei and nodej
NMidI = G_veins.Nodes{G_veins.Edges.EndNodes(:,1),'node_Mid'};
NMidJ = G_veins.Nodes{G_veins.Edges.EndNodes(:,2),'node_Mid'};
% % %             % Get the average edge weight for nodei and nodej
% % %             NAveI = G_veins.Nodes{G_veins.Edges.EndNodes(:,1),'node_Average'};
% % %             NAveJ = G_veins.Nodes{G_veins.Edges.EndNodes(:,2),'node_Average'};
% % %             % test whether nodei and nodej are perimeter nodes
% % %             NPerimI = contains(G_veins.Nodes{G_veins.Edges.EndNodes(:,1),'node_Type'},'P');
% % %             NPerimJ = contains(G_veins.Nodes{G_veins.Edges.EndNodes(:,2),'node_Type'},'P');
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
% % calculate an edge width skeleton based on the central width
% % for the selected edges
% V_int = num2cell(E);
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
[nY,nX] = size(bw_cnn);
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

function [areole_stats,polygon_stats] = fnc_polygon_analysis(bw_polygons,bw_areoles, polygon_LM)
% measure the stats for the polygonal regions using the label matrix
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
% calculate additional parameters
% areole_stats = A_stats;
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
% sort by the edge width and keep the smallest
edges = sortrows(edges,3);
[~,idx] = unique(edges(:,1:2),'rows');
edges = double(edges(idx,:));
idx = max(edges(:,1:2)==0, [],2);
edges(idx,:) = [];
% remove edges connected to themselves
idx = diff(edges(:,1:2),[],2)==0;
edges(idx,:) = [];
% remove edges not present in the polygons
EdgeTable = table([edges(:,1) edges(:,2)],edges(:,3),edges(:,4), 'VariableNames', names);
G_areas = graph(EdgeTable,NodeTable,'OmitSelfLoops');
% check the number of components and only keep the largest
CC = conncomp(G_areas);
G_areas = rmnode(G_areas,find(CC>1));
end

function T = fnc_summary_veins(G_veins,total_area,polygon_area,micron_per_pixel)
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
% get the total volume of the veins
T.VTotV = sum(G_veins.Edges.Volume(E_idx)).*(mm.^3); %
T.VLoopV = sum(G_veins.Edges.Volume(L_idx)).*(mm.^3); %
T.VTreeV = sum(G_veins.Edges.Volume(T_idx)).*(mm.^3); %
% get the weighted vein width
T.VTotW = (1/T.VTotL).*(sum(G_veins.Edges.Length(E_idx).*G_veins.Edges.Width(E_idx))).*mm;
T.VLoopW = (1/T.VLoopL).*(sum(G_veins.Edges.Length(L_idx).*G_veins.Edges.Width(L_idx))).*mm;
T.VTreeW = (1/T.VTreeL).*(sum(G_veins.Edges.Length(T_idx).*G_veins.Edges.Width(T_idx))).*mm;
% set the graph weight to be the length (in microns);
G_veins.Edges.Weight = G_veins.Edges.Length.*mm;
% % calculate the minimum spanning tree using Kruskal's algorithm
% MST = minspantree(G_veins,'method','sparse');
% T.VMSTratio = sum(MST.Edges.Weight)/T.VTotL;
% total areas analysed
T.TotA = total_area.*(mm.^2);
T.TotPA = polygon_area.*(mm.^2);
% get the density measurements
T.VTotLD = T.VTotL / T.TotA;
T.VLoopLD = T.VLoopL / T.TotA;
T.VTreeLD = T.VTreeL / T.TotA;
T.VTotVD = T.VTotV / T.TotA;
T.VLoopVD = T.VLoopV / T.TotA;
T.VTreeVD = T.VTreeV / T.TotA;
T.VNND = T.VNN / T.TotA;
% get the summary statistics for all relevant edge metrics in the table
% T = fnc_summary(T,metric,prefix,suffix,transform,units)
T = fnc_summary(T,G_veins.Edges.Length(E_idx),'VTot','Len','none',mm);
T = fnc_summary(T,G_veins.Edges.Length(L_idx),'VLoop','Len','none',mm);
T = fnc_summary(T,G_veins.Edges.Length(T_idx),'VTree','Len','none',mm);
T = fnc_summary(T,G_veins.Edges.Width(E_idx),'VTot','Wid','none',mm);
T = fnc_summary(T,G_veins.Edges.Width(L_idx),'VLoop','Wid','none',mm);
T = fnc_summary(T,G_veins.Edges.Width(T_idx),'VTree','Wid','none',mm);
T = fnc_summary(T,G_veins.Edges.SurfaceArea(E_idx),'VTot','SAr','none',mm);
T = fnc_summary(T,G_veins.Edges.SurfaceArea(L_idx),'VLoop','SAr','none',mm);
T = fnc_summary(T,G_veins.Edges.SurfaceArea(T_idx),'VTree','SAr','none',mm);
T = fnc_summary(T,G_veins.Edges.Volume(E_idx),'VTot','Vol','none',mm^3);
T = fnc_summary(T,G_veins.Edges.Volume(L_idx),'VLoop','Vol','none',mm^3);
T = fnc_summary(T,G_veins.Edges.Volume(T_idx),'VTree','Vol','none',mm^3);
T = fnc_summary(T,G_veins.Edges.Tortuosity(E_idx),'V','Tor','none',1);
T = fnc_summary(T,G_veins.Edges.Or_ij(E_idx),'V','Ori','circ',1);
T = fnc_summary(T,G_veins.Nodes.node_Min,'N','Bmn','none',1);
T = fnc_summary(T,G_veins.Nodes.node_Mid,'N','Bmd','none',1);
T = fnc_summary(T,G_veins.Nodes.node_Maj,'N','Bmx','none',1);
T = fnc_summary(T,G_veins.Nodes.node_Min_Mid,'N','Rnd','none',1);
T = fnc_summary(T,G_veins.Nodes.node_Min_Maj,'N','Rnx','none',1);
T = fnc_summary(T,G_veins.Nodes.node_Mid_Maj,'N','Rdx','none',1);
T = fnc_summary(T,G_veins.Nodes.node_Omin_Omid,'N','And','circ',1);
T = fnc_summary(T,G_veins.Nodes.node_Omin_Omaj,'N','Anx','circ',1);
T = fnc_summary(T,G_veins.Nodes.node_Omid_Omaj,'N','Adx','circ',1);
end


function T = fnc_summary_veins_HLD(G_veins,total_area,polygon_area,micron_per_pixel)
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
%Deg = degree(G_veins); % dimensionless
%T.VFE = sum(Deg==1); % freely ending veins have degree one
%T.VFEratio = T.VFE / T.VNE; % Number of FEV divided by number of edges
%T.Valpha = (T.VNE - T.VNN + 1)/((2*T.VNN)-5); % dimensionless
% get the total length of veins (in mm)
T.VTotL = sum(G_veins.Edges.Length(E_idx)).*mm; %
%T.VLoopL = sum(G_veins.Edges.Length(L_idx)).*mm; %
%T.VTreeL = sum(G_veins.Edges.Length(T_idx)).*mm; %
% get the total volume of the veins
T.VTotV = sum(G_veins.Edges.Volume(E_idx)).*(mm.^3); %
%T.VLoopV = sum(G_veins.Edges.Volume(L_idx)).*(mm.^3); %
%T.VTreeV = sum(G_veins.Edges.Volume(T_idx)).*(mm.^3); %
% get the weighted vein width
T.VTotW = (1/T.VTotL).*(sum(G_veins.Edges.Length(E_idx).*G_veins.Edges.Width(E_idx))).*mm;
%T.VLoopW = (1/T.VLoopL).*(sum(G_veins.Edges.Length(L_idx).*G_veins.Edges.Width(L_idx))).*mm;
%T.VTreeW = (1/T.VTreeL).*(sum(G_veins.Edges.Length(T_idx).*G_veins.Edges.Width(T_idx))).*mm;
% set the graph weight to be the length (in microns);
G_veins.Edges.Weight = G_veins.Edges.Length.*mm;
% % calculate the minimum spanning tree using Kruskal's algorithm
% MST = minspantree(G_veins,'method','sparse');
% T.VMSTratio = sum(MST.Edges.Weight)/T.VTotL;
% total areas analysed
T.TotA = total_area.*(mm.^2);
T.TotPA = polygon_area.*(mm.^2);
% get the density measurements
T.VTotLD = T.VTotL / T.TotA;
%T.VLoopLD = T.VLoopL / T.TotA;
%T.VTreeLD = T.VTreeL / T.TotA;
T.VTotVD = T.VTotV / T.TotA;
%T.VLoopVD = T.VLoopV / T.TotA;
%T.VTreeVD = T.VTreeV / T.TotA;
T.VNND = T.VNN / T.TotA;
% get the summary statistics for all relevant edge metrics in the table
% T = fnc_summary(T,metric,prefix,suffix,transform,units)
T = fnc_summary(T,G_veins.Edges.Length(E_idx),'VTot','Len','none',mm);
%T = fnc_summary(T,G_veins.Edges.Length(L_idx),'VLoop','Len','none',mm);
%T = fnc_summary(T,G_veins.Edges.Length(T_idx),'VTree','Len','none',mm);
T = fnc_summary(T,G_veins.Edges.Width(E_idx),'VTot','Wid','none',mm);
%T = fnc_summary(T,G_veins.Edges.Width(L_idx),'VLoop','Wid','none',mm);
%T = fnc_summary(T,G_veins.Edges.Width(T_idx),'VTree','Wid','none',mm);
T = fnc_summary(T,G_veins.Edges.SurfaceArea(E_idx),'VTot','SAr','none',mm);
%T = fnc_summary(T,G_veins.Edges.SurfaceArea(L_idx),'VLoop','SAr','none',mm);
%T = fnc_summary(T,G_veins.Edges.SurfaceArea(T_idx),'VTree','SAr','none',mm);
T = fnc_summary(T,G_veins.Edges.Volume(E_idx),'VTot','Vol','none',mm^3);
%T = fnc_summary(T,G_veins.Edges.Volume(L_idx),'VLoop','Vol','none',mm^3);
%T = fnc_summary(T,G_veins.Edges.Volume(T_idx),'VTree','Vol','none',mm^3);
T = fnc_summary(T,G_veins.Edges.Tortuosity(E_idx),'V','Tor','none',1);
T = fnc_summary(T,G_veins.Edges.Or_ij(E_idx),'V','Ori','circ',1);
T = fnc_summary(T,G_veins.Nodes.node_Min,'N','Bmn','none',1);
T = fnc_summary(T,G_veins.Nodes.node_Mid,'N','Bmd','none',1);
T = fnc_summary(T,G_veins.Nodes.node_Maj,'N','Bmx','none',1);
T = fnc_summary(T,G_veins.Nodes.node_Min_Mid,'N','Rnd','none',1);
T = fnc_summary(T,G_veins.Nodes.node_Min_Maj,'N','Rnx','none',1);
T = fnc_summary(T,G_veins.Nodes.node_Mid_Maj,'N','Rdx','none',1);
T = fnc_summary(T,G_veins.Nodes.node_Omin_Omid,'N','And','circ',1);
T = fnc_summary(T,G_veins.Nodes.node_Omin_Omaj,'N','Anx','circ',1);
T = fnc_summary(T,G_veins.Nodes.node_Omid_Omaj,'N','Adx','circ',1);
end



function T = fnc_summary_areoles(G_areoles,polygon_area,micron_per_pixel)
% set calibration factors
mm = micron_per_pixel./1000;
T = table;
T.ATA = sum(G_areoles.Nodes.Area).*(mm.^2);
T.ANN = numnodes(G_areoles);
PTA = polygon_area.*(mm.^2);
T.Aloop = T.ANN / PTA; % should be the same as T.Ploop
% get areole statistics
T = fnc_summary(T,G_areoles.Nodes.Area,'A','Are','none',mm^2);
T = fnc_summary(T,G_areoles.Nodes.ConvexArea,'A','CnA','none',mm^2);
T = fnc_summary(T,G_areoles.Nodes.Eccentricity,'A','Ecc','none',1);
T = fnc_summary(T,G_areoles.Nodes.MajorAxisLength,'A','Maj','none',mm);
T = fnc_summary(T,G_areoles.Nodes.MinorAxisLength,'A','Min','none',mm);
T = fnc_summary(T,G_areoles.Nodes.EquivDiameter,'A','EqD','none',mm);
T = fnc_summary(T,G_areoles.Nodes.Perimeter,'A','Per','none',mm);
T = fnc_summary(T,G_areoles.Nodes.Solidity,'A','Sld','none',mm);
T = fnc_summary(T,G_areoles.Nodes.Elongation,'A','Elg','none',1);
T = fnc_summary(T,G_areoles.Nodes.Circularity,'A','Cir','none',1);
T = fnc_summary(T,G_areoles.Nodes.Roughness,'A','Rgh','none',1);
T = fnc_summary(T,G_areoles.Nodes.Orientation,'A','Ori','circ',1);
T = fnc_summary(T,G_areoles.Nodes.MeanDistance,'A','Dav','none',1);
T = fnc_summary(T,G_areoles.Nodes.MaxDistance,'A','Dmx','none',1);
end

function T = fnc_summary_polygons(G_polygons,micron_per_pixel)
% set calibration factors
mm = micron_per_pixel./1000;
T = table;
T.PTA = sum(G_polygons.Nodes.Area).*(mm.^2);
T.PNN = numnodes(G_polygons);
T.Ploop = T.PNN / T.PTA; % should be the same as T.Aloop
% get polgonal area statistics
T = fnc_summary(T,G_polygons.Nodes.Area,'P','Are','none',mm^2);
T = fnc_summary(T,G_polygons.Nodes.ConvexArea,'P','CnA','none',mm^2);
T = fnc_summary(T,G_polygons.Nodes.Eccentricity,'P','Ecc','none',1);
T = fnc_summary(T,G_polygons.Nodes.MajorAxisLength,'P','Maj','none',mm);
T = fnc_summary(T,G_polygons.Nodes.MinorAxisLength,'P','Min','none',mm);
T = fnc_summary(T,G_polygons.Nodes.EquivDiameter,'P','EqD','none',mm);
T = fnc_summary(T,G_polygons.Nodes.Perimeter,'P','Per','none',mm);
T = fnc_summary(T,G_polygons.Nodes.Solidity,'P','Sld','none',mm);
T = fnc_summary(T,G_polygons.Nodes.Elongation,'P','Elg','none',1);
T = fnc_summary(T,G_polygons.Nodes.Circularity,'P','Cir','none',1);
T = fnc_summary(T,G_polygons.Nodes.Roughness,'P','Rgh','none',1);
T = fnc_summary(T,G_polygons.Nodes.Orientation,'P','Ori','circ',1);
T = fnc_summary(T,G_polygons.Nodes.MeanDistance,'P','Dav','none',1);
T = fnc_summary(T,G_polygons.Nodes.MaxDistance,'P','Dmx','none',1);
end

function T = fnc_summary_polygon_stats(polygon_stats,micron_per_pixel)
% set calibration factors
mm = micron_per_pixel./1000;
T = table;
T.PTA = sum([polygon_stats.Area]).*(mm.^2);
T.PNN = size(polygon_stats,1);
T.Ploop = T.PNN / T.PTA; % should be the same as T.Aloop
% get polgonal area statistics
T = fnc_summary(T,[polygon_stats.Area],'P','Are','none',mm^2);
T = fnc_summary(T,[polygon_stats.ConvexArea],'P','CnA','none',mm^2);
T = fnc_summary(T,[polygon_stats.Eccentricity],'P','Ecc','none',1);
T = fnc_summary(T,[polygon_stats.MajorAxisLength],'P','Maj','none',mm);
T = fnc_summary(T,[polygon_stats.MinorAxisLength],'P','Min','none',mm);
T = fnc_summary(T,[polygon_stats.EquivDiameter],'P','EqD','none',mm);
T = fnc_summary(T,[polygon_stats.Perimeter],'P','Per','none',mm);
T = fnc_summary(T,[polygon_stats.Solidity],'P','Sld','none',mm);
T = fnc_summary(T,[polygon_stats.Elongation],'P','Elg','none',1);
T = fnc_summary(T,[polygon_stats.Circularity],'P','Cir','none',1);
T = fnc_summary(T,[polygon_stats.Roughness],'P','Rgh','none',1);
T = fnc_summary(T,[polygon_stats.Orientation]','P','Ori','circ',1);
T = fnc_summary(T,[polygon_stats.MeanDistance],'P','Dav','none',1);
T = fnc_summary(T,[polygon_stats.MaxDistance],'P','Dmx','none',1);
end

function T = fnc_summary_areole_stats(areole_stats,polygon_area,micron_per_pixel)
% set calibration factors
mm = micron_per_pixel./1000;
T = table;
T.ATA = sum([areole_stats.Area]).*(mm.^2);
T.ANN = size(areole_stats,1);
T.Aloop = T.ANN / polygon_area; % should be the same as T.Ploop
% get areole statistics
T = fnc_summary(T,[areole_stats.Area],'A','Are','none',mm^2);
T = fnc_summary(T,[areole_stats.ConvexArea],'A','CnA','none',mm^2);
T = fnc_summary(T,[areole_stats.Eccentricity],'A','Ecc','none',1);
T = fnc_summary(T,[areole_stats.MajorAxisLength],'A','Maj','none',mm);
T = fnc_summary(T,[areole_stats.MinorAxisLength],'A','Min','none',mm);
T = fnc_summary(T,[areole_stats.EquivDiameter],'A','EqD','none',mm);
T = fnc_summary(T,[areole_stats.Perimeter],'A','Per','none',mm);
T = fnc_summary(T,[areole_stats.Solidity],'A','Sld','none',mm);
T = fnc_summary(T,[areole_stats.Elongation],'A','Elg','none',1);
T = fnc_summary(T,[areole_stats.Circularity],'A','Cir','none',1);
T = fnc_summary(T,[areole_stats.Roughness],'A','Rgh','none',1);
T = fnc_summary(T,[areole_stats.Orientation]','A','Ori','circ',1);
T = fnc_summary(T,[areole_stats.MeanDistance],'A','Dav','none',1);
T = fnc_summary(T,[areole_stats.MaxDistance],'A','Dmx','none',1);
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
% use circular stats
switch transform
    case 'circ'
        warning off
        T.([prefix 'av' suffix]) = circ_rad2ang(circ_mean(metric));
        %T.([prefix 'md' suffix]) = circ_rad2ang(circ_median(metric)); %
        % don't calculate the median as the intermediate array size is Ne x NE
        %         T.([prefix 'mn' suffix]) = circ_rad2ang(min(metric));
        %         T.([prefix 'mx' suffix]) = circ_rad2ang(max(metric));
        T.([prefix 'sd' suffix]) = circ_rad2ang(circ_std(metric));
        %         T.([prefix 'sk' suffix]) = circ_skewness(metric);
        warning on
    otherwise
        T.([prefix 'av' suffix]) = mean(metric).*units;
        T.([prefix 'md' suffix]) = median(metric).*units;
        T.([prefix 'mo' suffix]) = mode(metric).*units;
        %         T.([prefix 'mn' suffix]) = min(metric).*units;
        %         T.([prefix 'mx' suffix]) = max(metric).*units;
        T.([prefix 'sd' suffix]) = std(metric).*units;
        %         T.([prefix 'sk' suffix]) = skewness(metric);
end
end

function [HLD_levels, G_HLD, parent, HLD_metrics, im_HLD_order] = fnc_HLD(G_veins, G_polygons, G_areoles, polygon_stats, areole_stats, bw_polygons, bw_areoles, total_area,polygon_area,micron_per_pixel)
% construct a full binary polygon CC object
PCC.Connectivity = 4;
PCC.ImageSize = size(bw_polygons);
PCC.NumObjects = 1;
PCC.PixelIdxList = {};
% set up a slice CC object
SlCC.Connectivity = 4;
SlCC.ImageSize = size(bw_polygons);
SlCC.NumObjects = 1;
SlCC.PixelIdxList = {};
% select the largest component of the polygon graph
CC = conncomp(G_polygons);
idx = find(CC==1);
G_polygons = subgraph(G_polygons,idx);
% % extract the same component from the areole graph (which has the same node
% % IDs as the polygon graph).
% G_areoles = subgraph(G_areoles,idx);
% extract the same component from the stats arrays
polygon_stats = polygon_stats(idx);
areole_stats = areole_stats(idx);
% get the ID of the nodes remaining
ID = G_polygons.Nodes.ID;
% Keep veins from the vein graph that form part of the polygon_graph. These
% will be the boundary edges and any internal tree-like parts of the
% network, but will exclude edges from incomplete polygons on the boundary
% or disconnected polygons. Edges should have Ai and/or
% Aj corresponding to a polygon node ID.
% hfig = figure;
% hfig.Units = 'normalized';
% hfig.Position = [0 0 0.8 1];
% hfig.Color = 'w';
% gplot(adjacency(G_veins),[G_veins.Nodes.node_X_pix,G_veins.Nodes.node_Y_pix],'k:')
% axis off
% axis ij
% axis image
% axis square
% box on
Eidx = ismember(G_veins.Edges.Ai,ID) | ismember(G_veins.Edges.Aj,ID);
G_veins = rmedge(G_veins,find(~Eidx));
% hold on
% gplot(adjacency(G_veins),[G_veins.Nodes.node_X_pix,G_veins.Nodes.node_Y_pix],'k-')
% drawnow
% get the number of nodes and edge in the dual graph
nnP = single(numnodes(G_polygons));
neP = single(numedges(G_polygons));
parent = single(zeros(1,nnP));
width_threshold = zeros((2*nnP)-1,1,'single');
node_Area = [G_polygons.Nodes{:,'Area'}; zeros(nnP-1,1,'single')];
node_Degree = [ones(nnP,1,'single'); zeros(nnP-1,1,'single')];
node_Asymmetry = zeros(nnP*2-1,1,'single');
node_HS = [ones(nnP,1,'single'); zeros(nnP-1,1,'single')];
subtree_Asymmetry = zeros((2*nnP)-1,1,'single');
% % dimension the graph metric arrays
% wEdges = zeros(nnP,1,'single');
% nNodes = zeros(nnP,1,'single');
% nEdges = zeros(nnP,1,'single');
% total_length = zeros(nnP,1,'single');
% total_volume = zeros(nnP,1,'single');
% FEV = zeros(nnP,1,'single');
% set the first entry to the full graph
% set the counter for the number of fusions
Nf = 1;
% wEdges(Nf) = 0;
% nEdges(Nf) = single(numedges(G_veins));
% nNodes(Nf) = single(numnodes(G_veins));
% total_length(Nf) = single(sum(G_veins.Edges.Length));
% total_volume(Nf) = single(sum(G_veins.Edges.Volume));
% % calculate the number of FEV created
% D = degree(G_veins);
% FEV(Nf) = single(sum(D==1));
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
Nt(1) = nnP;
Ne = 0;
nLevels = 50;
HLD_level_criteria = 'width';
switch HLD_level_criteria
    case 'area'
        % set the first level for output in the HLD_movie to the nearest power of 2
        % within the number of nodes. The level will count down from this
        % number of areas until the final value is 1.
        mx = 2*(nextpow2(nnP)-1);
        criteria = single(round(sqrt(2).^(mx:-1:1)));
    case 'number'
        inc = neP/nLevels;
        criteria = inc:inc:neP;
    case 'width'
        % set up the cumulative widths
        csW = cumsum(W);
        % use the discretize function to divide into equal number of bins
        [~,BinEdges] = discretize(csW,min(neP,nLevels+1));
        bins = BinEdges(2:end-1);
        % find the nearest edge in csW that exceeds the bin edge
        [~,criteria] = min(single(abs(csW-bins)));
    case 'sq width'
        % set up the cumulative widths
        csW = (cumsum(W).^2);
        % use the discretize function to divide into equal number of bins
        [~,BinEdges] = discretize(csW,min(neP,nLevels+1));
        bins = BinEdges(1:end);
        % find the nearest edge in csW that exceeds the bin edge
        [~,criteria] = min(single(abs(csW-bins)));
    case 'sqrt width'
        % set up the cumulative widths on a log scale
        csW = sqrt(cumsum(W));
        % use the discretize function to divide into equal number of bins
        [~,BinEdges] = discretize(csW,min(neP,nLevels+1));
        bins = BinEdges(2:end-1);
        % find the index of the nearest edge in csW that exceeds the bin edge
        [~,criteria] = min(single(abs(csW-bins)));
        criteria = unique(criteria);
end
HLD_levels = zeros(numel(criteria),2,'single');
nL = 1;
% calculate the full set of summary statistics for the initial network 
P = fnc_summary_polygon_stats(polygon_stats,micron_per_pixel);
A = fnc_summary_areole_stats(areole_stats,polygon_area,micron_per_pixel);
V = fnc_summary_veins_HLD(G_veins,total_area,polygon_area,micron_per_pixel);
% As summary reduces the distribution for each variable for network object
% to a single values, the summary statistics can be combined into a single
% row vector
HLD_metrics(1,:) = [V A P];
Sl_HLD_metrics(1,:) = [V A P];

% get all the current polygon PixelIdxList at the start as a cell array;
P_PIL = {polygon_stats.PixelIdxList};
% set up the endnodes
EndNodes = zeros(nnP*2-2,2,'single');
% set a colormap with nnP entries
cols = jet(nnP);
exclude = [];
G_initial = G_veins;
G_current = G_veins;
% loop through all the edges, calculating the metrics
for iE = 1:neP
    % test each edge in sequence
    Ni = ET(1,1);
    Nj = ET(1,2);
    % the edge to be removed only links two areas if
    % the end nodes are different
    if Ni~=Nj
        Nf = single(Nf+1);
        % create a new node
        Nk = single(Nk+1);
        % construct the EndNodes for the two new edges that connect Ni and Nj to Nk
        Ne = Ne+1;
        EndNodes(Ne,:) = [Ni Nk];
        Ne = Ne+1;
        EndNodes(Ne,:) = [Nj Nk];
        % sum the areas of the nodes to fuse
        node_Area(Nk) = node_Area(Ni)+node_Area(Nj);
        % sum the degree 
        node_Degree(Nk) = node_Degree(Ni)+node_Degree(Nj);
        % get the width threshold for the HLD
        width_threshold(Nk,1) = ET(1,3);
        % find a measure of the partition asymmetry of the
        % bifurcation vertex
        node_Asymmetry(Nk,1) = abs(node_Degree(Ni)-node_Degree(Nj))/max(node_Degree(Ni),node_Degree(Nj));
        % calculate the subtree asymmetry
        subtree_Asymmetry(Nk,1) = (1/(node_Degree(Nk)-1))*((node_Asymmetry(Ni)*(node_Degree(Ni)-1))+(node_Asymmetry(Nj)*(node_Degree(Nj)-1)));
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
        exclude = [exclude; Ni; Nj];
        % find edges in the vein graph up to and including this edge width
        Eidx = G_veins.Edges.Width <= width_threshold(Nk,1);
        % remove these edges from the graph
        G_veins = rmedge(G_veins,find(Eidx));
        % only keep veins that are still connected
        CC = conncomp(G_veins);
        [N,~] = histcounts(CC,max(CC));
        [~,idx] = max(N);
        G_veins = subgraph(G_veins,find(CC==idx));
        % replace any other occurrences of the nodes that have fused with
        % the new node ID
        idx = G_veins.Edges.Ai == Ni | G_veins.Edges.Ai == Nj;
        G_veins.Edges.Ai(idx) = Nk;
        idx = G_veins.Edges.Aj == Ni | G_veins.Edges.Aj == Nj;
        G_veins.Edges.Aj(idx) = Nk;
        % plot the graph and re-colour the edges remaining with the index
        % into the custom colormap
%         [X,Y] = gplot(adjacency(G_veins),[G_veins.Nodes.node_X_pix,G_veins.Nodes.node_Y_pix]);
%         plot(X, Y, '-', 'Color', cols(Nk-nnP,:));
%         drawnow
        % add the PixelIdxList for the nodes that have fused to the new node
        P_PIL(Nk) = {cat(1,P_PIL{Ni},P_PIL{Nj})};
        switch HLD_level_criteria
            case 'area'
                % count the number of nodes (areas) now present
                nP = (2*nnP)-Nk;
                % check whether the areas remaining are a power of sqrt(2).
                % If so then record the current edge width to use as a
                % threshold in the HLD movie
                if nP <= criteria(nL)
                    HLD_levels(nL,:) = [ET(1,3),nP];
                    % increment the level counter
                    nL = nL + 1;
                end
            case {'width';'log width';'sqrt width';'sq width';'number'}
                % check whether the current edge is above the threshold for
                % the next edge width interval
                if iE >= criteria(nL)
                    % store the current width as a threshold to segment the
                    % weighted skeleton array
                    HLD_levels(nL,:) = [ET(1,3),iE];
                    % increment the level counter
                    nL = nL + 1;
                    % record the node at this threshold
                    Nt(nL) = Nk;
                    include = setdiff(1:Nk,exclude);
                    % NOTE THIS CALCULATES THE STATS ON THE COMPLETE
                    % REGION IMAGE NOT JUST THE NEW REGION WHICH WOULD BE
                    % QUICKER
                    PCC.NumObjects = length(include);
                    PCC.PixelIdxList  = P_PIL(include);
                    LM = labelmatrix(PCC);
                    % calculate all the metrics for the complete new image
                    % including the new fused region
                    [A_stats,P_stats] = fnc_polygon_analysis(bw_polygons,bw_areoles, LM);
                    P = fnc_summary_polygon_stats(P_stats,micron_per_pixel);
                    A = fnc_summary_areole_stats(A_stats,polygon_area,micron_per_pixel);
                    V = fnc_summary_veins_HLD(G_veins,total_area,polygon_area,micron_per_pixel);
                    HLD_metrics(nL,:) = [V A P];
                    % extract the veins that have been removed to calculate their stats
                    % separately. Start by finding the nodes that have been removed
                    idx = setdiff(G_current.Nodes.node_ID,G_veins.Nodes.node_ID);
                    G_cut = subgraph(G_current,idx);
%                     C(nL,:) = fnc_summary_veins(G_cut,total_area,polygon_area,micron_per_pixel)
                    % get the polygons and areoles that have fused
                                        SlCC.NumObjects = length(Nt(nL-1):Nt(nL));
                    SlCC.PixelIdxList  = P_PIL(Nt(nL-1):Nt(nL));
                    SlLM = labelmatrix(SlCC);
                    figure
                    imshow(SlLM,[])
                    % calculate all the metrics for the complete image
                    % including the new fused regions. There may be some
                    % areas that are set to zero as they have been subsimed
                    % into a larger area within the selected bin intervals.
                    [Sl_A_stats,Sl_P_stats] = fnc_polygon_analysis(bw_polygons,bw_areoles, SlLM);
                    idxA = find([Sl_A_stats.Area]>0);
                    SlP = fnc_summary_polygon_stats(Sl_P_stats(idxA),micron_per_pixel);
                    SlA = fnc_summary_areole_stats(Sl_A_stats(idxA),polygon_area,micron_per_pixel);
                    SlV = fnc_summary_veins_HLD(G_cut,total_area,polygon_area,micron_per_pixel);
                    Sl_HLD_metrics(nL,:) = [SlV SlA SlP]
                    hold on

                    [X,Y] = gplot(adjacency(G_initial),[G_initial.Nodes.node_X_pix,G_initial.Nodes.node_Y_pix]);
                    plot(X, Y, '-', 'Color', 'w');
                     [X,Y] = gplot(adjacency(G_cut),[G_cut.Nodes.node_X_pix,G_cut.Nodes.node_Y_pix]);
                    plot(X, Y, '-', 'Color', cols(Nk-nnP,:));
                    drawnow                   
                    [X,Y] = gplot(adjacency(G_veins),[G_veins.Nodes.node_X_pix,G_veins.Nodes.node_Y_pix]);
                    plot(X, Y, ':', 'Color', 'r');
                    
                    drawnow
                            % take a copy of the current vein graph
        G_current = G_veins;
                end
        end
    else
        % delete the current edge as it lies within areas that are already
        % fused
        ET(1,:) = [];
    end
end
% complete links to the root node
parent(end+1) = Nk+1;
parent = double(fliplr(max(parent(:)) - parent));
% assemble the HLD graph object
NodeTable = table((1:(2*nnP)-1)', width_threshold, node_Area, node_Degree, node_Asymmetry, node_HS,subtree_Asymmetry, ...
    'VariableNames',{'node_ID' 'width_threshold' 'node_Area' 'node_Degree' 'node_Asymmetry' 'node_HS' 'subtree_Asymmetry'});
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
% % % % create a table for the HLD metrics for the network decomposition
% % % HLD_G_metrics = table((0:length(nNodes)-1)', wEdges, nNodes, nEdges, total_length, total_volume, FEV, ...
% % %      'VariableNames',{'fusion_number' 'width_threshold' 'vein_nodes' 'vein_edges' 'total_length' 'total_volume' 'FEV'});
% im_HLD_order = export_fig('HLD_coded','-png','-m4',hfig);
%delete(hfig);
assignin('base','Sl_HLD_metrics',Sl_HLD_metrics)
end

function [HLD_results,HLD_slices] = fnc_HLD_slices(G_veins,bw_cnn,bw_mask,total_area_mask,width,HLD_levels,micron_per_pixel)
% remove the negative mask pixels from the width image
width(width<0) = 0;
% remove any rows that were not set during the threshold
% selection
HLD_levels(HLD_levels(:,1)==0,:) = [];
% start the levels at 0
HLD_levels = [0 0; HLD_levels];
nLevels = size(HLD_levels,1);
% set up a movie array
sz = size(imresize(total_area_mask, 0.25));
HLD_slices = zeros(sz(1),sz(2),3,1,nLevels,'uint8');
% get the border
border = bwperim(total_area_mask);
% clear the results array
HLD_results = table;
for iW = 1:nLevels
    % Get the skeleton for pixels above the width
    sk = width > HLD_levels(iW,1);
    % only keep the connected skeleton and the border
    sk = bwareafilt(sk,1) | border;
    % run the polygon analysis and extact the graphs
    [G_veins,areole_stats,polygon_stats,~,im_polygons_rgb,~] = fnc_polygon_analysis(G_veins,bw_cnn,sk,bw_mask);
    [G_areoles] = fnc_area_graph(G_veins,areole_stats);
    [G_polygons] = fnc_area_graph(G_veins,polygon_stats);
    % get the ID of the nodes remaining
    ID = G_polygons.Nodes.ID;
    % Keep veins from the vein graph that form part of the polygon_graph. These
    % will be the boundary edges and any internal tree-like parts of the
    % network, but will exclude edges from incomplete polygons on the boundary
    % or disconnected polygons. Edges should have Ai and/or
    % Aj within ID.
    Eidx = ismember(G_veins.Edges.Ai,ID) | ismember(G_veins.Edges.Aj,ID);
    G_veins = rmedge(G_veins,find(~Eidx));
    % calculate the summary statistics for the areoles and polygons
    veins = fnc_summary_veins(total_area_mask,G_veins,G_polygons,micron_per_pixel);
    areoles = fnc_summary_areoles(G_areoles,G_polygons,micron_per_pixel);
    polygons = fnc_summary_polygons(G_polygons,micron_per_pixel);
    % update the results table
    HLD_results(iW,:) = [veins areoles polygons];
    % construct the movie
    HLD_slices(:,:,:,1,iW) = imresize(im_polygons_rgb, [sz(1) sz(2)]);
end
end

function hfig = display_PR(PR, FolderName)
methods = {'cnn';'vesselness';'featuretype';'bowlerhat';'midgrey'};%;'niblack';'bernsen';'sauvola'};
cols = {'r-';'g-';'b-';'c-';'m-';'g:';'b:';'c:'};
pnts = {'ro';'go';'bo';'co';'mo';'g*';'b*';'c*'};
mrks = {'.';'.';'.';'.';'.';'.';'.';'.'};

% plot the precision-recall plots and mark the optimum
hfig = figure;
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
legend(h,methods,'Location','SouthEast')
%title([FolderName ' : Precision-Recall  full-width images'])
drawnow
end

function hfig = display_sk_PR(PR, FolderName)
methods = {'cnn_sk';'vesselness_sk';'featuretype_sk';'bowlerhat_sk';'midgrey_sk'};%;'niblack_sk';'bernsen_sk';'sauvola_sk'};
cols = {'r-';'g-';'b-';'c-';'m-';'g:';'b:';'c:'};
pnts = {'ro';'go';'bo';'co';'mo:';'go';'bo';'co'};
mrks = {'.';'.';'.';'.';'.';'.';'.';'.'};
% plot the precision-recall plots and mark the optimum
hfig = figure;
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
legend(h,strrep(methods,'_sk',''),'Location','SouthEast')

% title([FolderName ' : Precision-Recall skeleton'])
drawnow
end

function hfig = display_figure(images,graphs,titles,G,E_width,links,name,ExportFigs)
hfig = figure;
for ia = 1:6
    ax(ia) = subplot(2,3,ia);
    axes(ax(ia))
    pos = ax(ia).OuterPosition;
    ax(ia).Position = pos;
end
hfig.Units = 'normalized';
hfig.Position = [0 0 1 1];
hfig.Color = 'w';
linkaxes(ax(links),'xy')
set(gcf,'renderer','opengl')
for ia = 1:6
    axes(ax(ia))
    if ~isempty(images{ia})
        imshow(images{ia},[])
        h = title(titles{ia},'fontsize',12,'fontweight','normal','interpreter','none');
        h.FontWeight = 'normal';
    end
    ax(ia).XTick = [];
    ax(ia).YTick = [];
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
    end
end
drawnow
if ExportFigs
    export_fig(name,'-png','-r300',hfig)
    saveas(hfig,name)
end
end

function hfig = display_HLD(G_polygons,im_cnn,HLD_levels,im_HLD_order,G_HLD,parent)
hfig = figure;
hfig.Units = 'normalized';
hfig.Position = [0 0 0.6 1];
hfig.Color = 'w';
% display the dual graph
ax(1) = subplot(3,3,1);
    axes(ax(1))
    pos = ax(1).OuterPosition;
    ax(1).Position = pos;

imshow(im_cnn,[])
% get edge width
E = G_polygons.Edges.Width;
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

hold on
plot(G_polygons, 'XData',G_polygons.Nodes.node_X_pix,'YData',G_polygons.Nodes.node_Y_pix, ...
    'NodeColor','g','MarkerSize',1,'Marker', 'none', 'NodeLabel', [], ...
    'EdgeColor',E_color,'EdgeAlpha',1,'EdgeLabel', [],'LineWidth',1);
axis off
box on
% display the treeplot
subplot(3,3,[2,3])
treeplot(parent,'','r')
ylabel('node level');
xlabel('terminal nodes');
box on
% calculate the average asymmetry
subplot(3,3,4)
scatter(log(G_HLD.Nodes.node_Degree), G_HLD.Nodes.subtree_Asymmetry)
xlabel('log degree')
ylabel('subtree asymmetry')
box on
% calculate the cumulative size distribution
a = sort(G_HLD.Nodes.node_Area,'ascend');
csN = cumsum(1:numnodes(G_HLD));
Pa = 1-(csN/max(csN));
subplot(3,3,5)
plot(log(a),log(a.*Pa'), 'go-')
xlabel('log area')
ylabel('log Pa*area')
box on
% calculate the strahler bifurcation ratio
for iHS = 1:max(G_HLD.Nodes.node_HS)-1
    Bifurcation_ratio(iHS) = sum(G_HLD.Nodes.node_HS==iHS)/sum(G_HLD.Nodes.node_HS==iHS+1);
end
subplot(3,3,6)
plot(Bifurcation_ratio,'ro-')
xlabel('Strahler number')
ylabel('bifurcation ratio')
box on

% display the cumulative width pdf
W = sort(G_polygons.Edges{:,'Width'});
subplot(3,3,7)
plot(cumsum(W),'k-')
hold on
plot([HLD_levels(:,2) HLD_levels(:,2)],ylim,'r:')
xlabel('edge number')
ylabel('cumulative width')

box on
% display the removal order image
% subplot(3,3,8)
ax(8) = subplot(3,3,8);
    axes(ax(8))
    pos = ax(8).OuterPosition;
    ax(8).Position = pos;

imshow(im_HLD_order)
axis off
box on
% display the metric scaling relationship
subplot(3,3,9);
histogram
end


