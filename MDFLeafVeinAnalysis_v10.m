function results = MDFLeafVeinAnalysis_v10(FolderName,MicronPerPixel,DownSample,threshold,ShowFigs,ExportFigs,FullLeaf,FullMetrics)
%% set up directories
dir_out_images = ['..' filesep 'summary' filesep 'images' filesep];
dir_out_width = ['..' filesep 'summary' filesep 'width' filesep];
dir_out_data = ['..' filesep 'summary' filesep 'data' filesep];
dir_out_HLD = ['..' filesep 'summary' filesep 'HLD' filesep];
%% set up parameters
Calibration = MicronPerPixel.*DownSample;
sk_width = 5;
E_width = 1;
%% set up default colour map
%cmap = jet(256);
% use a perceptually uniform colormap from Peter Kovesi
cmap = colorcet('R3');
cmap(1,:) = 0;
%% load in the image files
step = 0;
warning off
% Load in the images and downsample them. All results are calibrated in
% microns taking this into account.
step = step+1;
disp(['Step ' num2str(step) ': Processing ' FolderName])
[im,imCNN,bwMask,bwVein,bwROI,bwGT] = fnc_load_CNN_images(FolderName,DownSample);
%% get the skeleton
step = step+1;
disp(['Step ' num2str(step) ': Skeleton extraction using threshold ' num2str(threshold)])
[bwCNN, sk, skLoop, skTree, skRing, skMask] = fnc_skeleton(imCNN,bwVein,bwMask,threshold);
%% calculate the width
step = step+1;
disp(['Step ' num2str(step) ': Calculating width'])
[im_width, ~] = bwdist(~bwCNN,'Euclidean');
% extract the initial width along the skeleton
W_microns = zeros(size(im_width),'single');
W_microns(sk) = single(im_width(sk).*2.*Calibration);
% % % %% calculate the intensity weighted width using granulometry
% % % [im_granulometry] = fnc_granulometry(im, bw_cnn, bw_mask);
% % % W_granulometry = zeros(size(im_granulometry),'single');
% % % W_granulometry(sk) = single(im_granulometry(sk).*2);
%% extract network using the thinned skeleton
step = step+1;
disp(['Step ' num2str(step) ': Extracting the network'])
% collect a cell array of connected edge pixels
[edgelist, edgeim] = edgelink(sk);
%% find any self loops and split them
step = step+1;
disp(['Step ' num2str(step) ': Resolving self loops'])
[edgelist, edgeim] = fnc_resolve_self_loops(edgelist,edgeim);
%% resolve duplicate edges by splitting one edge into two
step = step+1;
disp(['Step ' num2str(step) ': Resolving duplicates'])
[edgelist, ~] = fnc_resolve_duplicates(edgelist,edgeim);
%% construct the weighted graph
step = step+1;
disp(['Step ' num2str(step) ': Weighted graph'])
[G_veins,edgelist] = fnc_weighted_graph(edgelist,W_microns,skTree,Calibration);
%% Refine the width
step = step+1;
disp(['Step ' num2str(step) ': Refining width'])
[G_veins,edgelist_center] = fnc_refine_width(G_veins,edgelist,im,imCNN,W_microns,Calibration);
%% calculate a pixel skeleton for the center weighted edges
step = step+1;
disp(['Step ' num2str(step) ': Colour-coded skeleton'])
[CW_microns,im_width,coded_CW,coded_FW] = fnc_coded_skeleton(im,sk,skMask,G_veins,edgelist,edgelist_center,sk_width,cmap);
%% display the weighted network
step = step+1;
disp(['Step ' num2str(step) ': Image display'])
if ExportFigs == 1
    step = step+1;
    disp(['Step ' num2str(step) ': Saving width images'])
    [nY,nX,~] = size(coded_CW);
    % save the color-coded width images
    %     fout = [dir_out_images FolderName '_centerwidth.png'];
    %     imwrite(coded_CW,fout,'png','Xresolution',nX,'Yresolution',nY)
    fout = [dir_out_images FolderName '_fullwidth.png'];
    imwrite(coded_FW,fout,'png','Xresolution',nX,'Yresolution',nY)
    % save the greyscale width array as a matlab file. Note outside the
    % masked area is now coded as -1
    save([dir_out_width FolderName '_Width_array'],'im_width')
end
%% find the areoles
step = step+1;
disp(['Step ' num2str(step) ': polygon analysis'])
% find the polygon and areole areas
[G_veins, sk_polygon, bw_polygons, bw_areoles, polygon_LM] = fnc_polygon_find(G_veins,bwCNN,sk,skLoop,skRing,skMask);
[areole_stats,polygon_stats] = fnc_polygon_analysis(bw_polygons,bw_areoles, polygon_LM,FullMetrics,Calibration);
% construct color-coded image based on log area for display
im_areoles_rgb = fnc_polygon_image(areole_stats, sk_polygon, skMask);
%im_polygons_rgb = fnc_polygon_image(polygon_stats, sk_polygon, total_area_mask);
%% convert to an areole graph and a polygon graph
step = step+1;
disp(['Step ' num2str(step) ': Areole and Polygon dual graph conversion'])
[G_polygons,polygon_LM2] = fnc_area_graph(G_veins,polygon_stats,polygon_LM,'polygons',Calibration);
[G_areoles,~] = fnc_area_graph(G_veins,areole_stats,polygon_LM,'areoles',Calibration);
%% collect summary statistics into a results array
step = step+1;
disp(['Step ' num2str(step) ': Summary statistics'])
%total_area = sum(bw_mask(:))*Calibration^2;
total_area = sum(skMask(:))*Calibration^2;
polygon_area = sum(G_polygons.Nodes.Area);
veins = fnc_summary_veins(G_veins,total_area,polygon_area);
areoles = fnc_summary_areoles(G_areoles,polygon_area,FullMetrics);
polygons = fnc_summary_polygons(G_polygons,FullMetrics);
results = [veins areoles polygons];
%% add in file information
results.File = FolderName;
results.TimeStamp = datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss Z');
results.MicronPerPixel = Calibration;
results.DownSample = DownSample;
results.Threshold = threshold;
% reorder the table to get the file info first
results = results(:, [end-4:end 1:end-5]);
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
    images = {im,max(imCNN(:))-imCNN,skel,coded_FW,im_areoles_rgb,max(imCNN(:))-imCNN};
    graphs = {'none','none','none','none','none','Width'};
    titles = {'original','CNN','Skeleton','width','areoles','dual graph'};
    display_figure(images,graphs,titles,G_polygons,E_width,[1:6],[dir_out_images FolderName '_Figure'],ExportFigs);
end
%% Hierarchical loop decomposition
step = step+1;
disp(['Step ' num2str(step) ': Hierarchical loop decomposition'])
[G_HLD, HLD_image,  parent] = fnc_HLD(G_veins, G_polygons, polygon_stats, areole_stats, polygon_LM2, bw_polygons, Calibration);
save([dir_out_HLD FolderName '_HLD_results.mat'],'G_HLD','parent')
%% HLD display
if ShowFigs == 1 && ExportFigs == 0
    display_HLD_v1(G_polygons,imCNN,G_HLD,FullLeaf,[dir_out_HLD FolderName '_HLD_1'],ExportFigs);
    display_HLD_v2(G_polygons,imCNN,G_HLD,FullLeaf,[dir_out_HLD FolderName '_HLD_2'],ExportFigs);
    display_HLD_figure(G_polygons,imCNN,G_HLD,FullLeaf,[dir_out_HLD FolderName '_HLD_3'],ExportFigs);
end
if ExportFigs == 1
    display_HLD_v1(G_polygons,imCNN,G_HLD,FullLeaf,[dir_out_HLD FolderName '_HLD_1'],ExportFigs);
    display_HLD_v2(G_polygons,imCNN,G_HLD,FullLeaf,[dir_out_HLD FolderName '_HLD_2'],ExportFigs);
    display_HLD_figure(G_polygons,imCNN,G_HLD,FullLeaf,[dir_out_HLD FolderName '_HLD_3'],ExportFigs);
end
%%
% % % if ExportFigs == 1
% % %     step = step+1;
% % %     disp(['Step ' num2str(step) ': Saving HLD image'])
% % %     [nY,nX,~] = size(HLD_image);
% % %     % save the color-coded width images
% % %     fout = [dir_out_images FolderName '_HLD_level.png'];
% % %     mx = double(max(HLD_image(:)));
% % %     cmap = jet(mx+1);
% % %     cmap(1,:) = 0;
% % %     %HLD_image = colfilt(HLD_image,[3 3],'sliding',@max);
% % %     HLD_image(bw_polygons) = 0;
% % %     hfig = figure;
% % %     imshow(ind2rgb(HLD_image,cmap))
% % %     imwrite(ind2rgb(HLD_image,cmap),fout,'png','Xresolution',nX,'Yresolution',nY)
% % %     delete(hfig)
% % % end
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
% dir_current = pwd;
% cd(dir_out_data);
% dir_in = pwd;
% %xls_delete_sheets([dir_in filesep FolderName '_results.xlsx'],{'Sheet1','Sheet2','Sheet3'})
% cd(dir_current);
end

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
    disp('        No original image')
    im = zeros(nY,nX, 'uint8');
end
% load in the mask images
if exist(mask_name,'file') == 2
    bw_mask = imresize(logical(imread(mask_name)),[nY,nX]);
else
    disp('        No mask image')
    bw_mask = true(nY,nX);
end
if exist(cnn_mask_name,'file') == 2
    bw_cnn_mask = imresize(logical(imread(cnn_mask_name)),[nY,nX]);
else
    disp('        No cnn mask image')
    bw_cnn_mask = true(nY,nX);
end
% load in the big vein image if present
if exist(vein_name,'file') == 2
    bw_vein = imresize(logical(imread(vein_name)),[nY,nX]);
else
    disp('        No manual vein image')
    bw_vein = false(nY,nX);
end
% load in the manual roi and ground truth images
if exist(roi_name,'file') == 2
    bw_roi = imresize(logical(imread(roi_name)>0),[nY,nX]);
else
    disp('        No roi image')
    bw_roi = true(nY,nX);
end
if exist(GT_name,'file') == 2
    bw_GT = imresize(logical(imread(GT_name)>0),[nY,nX]);
else
    disp('        No GT image')
    bw_GT = false(nY,nX);
end
% apply the masks
bw_mask = bw_mask & bw_cnn_mask;
im_cnn(~bw_mask) = 0;
end

function [bwCNN, skFinal, skLoop, skTree, skRing, skMask] = fnc_skeleton(im_in,bwVein,bwMask,threshold)
%warning off
if islogical(im_in)
    % the input image is already a binary image
    bwCNN = im_in;
else
    % impose local minima to smooth out background noise using a dip of 26.5%
    % approximating the raleigh criterion
    exmin = imextendedmin(mat2gray(im_in),0.265);
    im = imimposemin(mat2gray(im_in),exmin);
    % convert to a binary image
    bwCNN = imbinarize(im,threshold);
    %bw_cnn = imbinarize(mat2gray(im_in),threshold);
end
% add in the big vein image if present
if ~isempty(bwVein)
    bwCNN = bwCNN | bwVein;
end
% smooth the binary image
%bw_cnn = medfilt2(bw_cnn,[3 3]);
% fill in any small holes - suggest ~5x5 um = 25 um2,
% 100/(Calibration^2) in pixels after downsampling
% % % imHoles = bwareafilt(imcomplement(bwCNN),[0 round(25/Calibration^2)]);
% % % disp(['         Check for small holes: ' num2str(any(any(imHoles)))])
% % % bwCNN = bwCNN | imHoles;
if ~isempty(bwMask)
    %bwCNN = bwCNN | ~bwMask 
    bwCNN = bwCNN & bwMask;
else
    bwMask = true(size(bwCNN));
end
% keep the connected component
bwCNN = bwareafilt(bwCNN,1);
% pad the array to ensure veins touching the edge form part of a dummy loop so that they are segmented to the midline
%bwCNN = padarray(bwCNN,[1 1],1,'both');
% calculate the distance transform as the input to a watershed segmentation
D = bwdist(~bwCNN,'Euclidean');
%W = watershed(D,4);
% use 8 connectivity to avoid slicing off single border pixels on a
% diagonal. Note this gives a thicker watershed, but is thinned back to a
% single pixel skeleton.
W = watershed(D,8);
% get the watershed skeleton comprising only loops
skLoopInitial = W == 0;
% fill in any single pixel holes
skLoopInitial = bwmorph(skLoopInitial,'fill');
% thin to a single pixel skeleton
skLoopInitial = bwmorph(skLoopInitial,'thin',Inf);
% skLoop = bwmorph(skLoop,'spur',inf); way too slow!
% % % % remove regions outside the mask and the border
% % % skLoop = skLoopInitial & padarray(bwMask,[1 1],0,'both');
skLoop = skLoopInitial;
% prune any external truncated loops using the dilated region inside the loopy skeleton as
% a mask
skMask = imdilate(imfill(skLoop,'holes') & ~skLoop, [0 1 0; 1 1 1; 0 1 0]);
% find any internal damaged regions that need to be excluded
[r,c] = find(imclearborder(~bwMask));
DamageMask = bwselect(~skLoop,c,r,4);
% keep the largest component as a modified mask
%skLoopMask = bwareafilt(skLoopMask & padarray(bwMask,[1 1],0,'both'),1);
skMask = bwareafilt(skMask & ~DamageMask,1);
% prune the loop skeleton to match
skLoop = skLoop & skMask;
% find any isolated loops not connected to the GCC within the skLoopMask
% area
skRing = skLoop & ~bwareafilt(skLoop,1);
% exclude rings touching the border as these are incomplete
skRing = imclearborder(skRing);
if any(skRing(:))
    % fill the loops
    skLoopFill = imfill(skRing,'holes');
    % punch the loop skeleton back out. This ensure two touching loops are
    % treated separately
    skLoopFill = skLoopFill & ~skRing;
    % erode to a single point in the middle of the loop
    skLoopPoints = bwulterode(skLoopFill);
    % use these points to fill the loop in the binary image
    bwCNNFill = imfill(bwCNN,find(skLoopPoints));
    % thin the binary image to a single pixel skeleton to ensure that there
    % is a connected skeleton within the isolated loops. Later on this will
    % be relaced by the loop skeleton but is needed to maintain
    % connecectivity of the trees
    sk = bwmorph(bwCNNFill,'thin',Inf);
else
    sk =  bwmorph(bwCNN,'thin',Inf);
end
% % only keep the connected component within the loop mask
% sk = sk & skLoopMask;
% sk = bwareafilt(sk,1);
% get the loops from the thinned skeleton using the watershed with 4
% connectivity
W1 = watershed(sk,4);
skLoop2 = W1 == 0;
skLoop2 = bwmorph(skLoop2,'thin',Inf);
% find tree regions in the thinned skeleton as the regions not part of a loop
skTree = xor(sk,skLoop2);
% also punch out the original loop skeleton. This may disconnect short segments
% from the base of the tree skeleton. The main trunk will be re-connected
% later.
skTree(skLoop) = 0;
% make sure trees connected to isolated loops are retained
if any(skRing(:))
    % punch out the skeleton within the area of the isolated rings
    skTree = skTree & ~skLoopFill;
    % add back in the rings
    skTree = skTree | skRing;
end
% get the endpoints of the thinned skeleton, representing the free end
% veins
epsk = bwmorph(sk,'endpoints');
% mask out endpoints and skeleton outside the loopy region and mask
epsk = epsk & skMask;
% only keep branches that originate from an endpoint
[r,c] = find(epsk);
skTree = bwselect(skTree,c,r);
% now find the endpoints at the root of the tree
epskTree = bwmorph(skTree,'endpoints');
epskTree = xor(epsk,epskTree);
% only work with endpoints that are not already connected to
% the watershed skeleton
connected = bwareafilt(epskTree | skLoop,1);
epskTree = epskTree & ~connected;
% get the feature map from the watershed skeleton to find the nearest pixel
% to connect to in the original full loop skeleton
[~,skW_idx] = bwdist(skLoopInitial);
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
% % remove any tree components outside the mask
% skTree = skTree & padarray(bwMask,[1 1],0,'both');
if any(skRing(:))
    % remove any parts of the skeleton overlapping the loops
    skTree(skLoopFill) = 0;
    % remove the rings from the loop skeleton as they will be included in
    % the tree skeleton
    skLoop(skRing) = 0;
end
% add the watershed loop skeleton back in
skFinal = skTree | skLoop;
% % remove the padding
% skFinal = skFinal(2:end-1,2:end-1);
% skLoop = skLoop(2:end-1,2:end-1);
% skTree = skTree(2:end-1,2:end-1);
% bwCNN = bwCNN(2:end-1,2:end-1);
% remove edges touching the mask
bp = bwmorph(skFinal,'branchpoints');
sk = skFinal;
sk(imdilate(bp,ones(3))) = 0;
B = bwperim(true(size(bwMask)));
[r,c] = find(~bwMask | B);
touching = bwselect(sk|~bwMask,c,r);
skFinal(touching) = 0;
% make sure pixels are allocated correctly
overlaps = skLoop & skTree;
skTree(overlaps) = 0;
% keep the largest connected component
skFinal = bwareafilt(skFinal,1);
skLoop = skLoop & skFinal;
skTree = skTree & skFinal;
end

function [width] = fnc_granulometry(im, bw_cnn, bw_mask)
method = 'gradient';
im = imcomplement(single(mat2gray(im)));
% constrain the CLAHE image to the thresholded width
%im(~bw_cnn) = 0;
im(~bw_mask) = 0;
s = 0:60;
imo = zeros([size(im),length(s)], 'single');
im = imtophat(im,strel('disk',61));
for i=1:length(s)
    imo(:,:,i) = imopen(im,strel('disk',s(i)));
end
switch method
    case 'integral'
        % integral granulometry
        width = sum(imo,3);
    case 'gradient'
        % max neg gradient
        imgcd = diff(imo,1,3);
        [imgcm,imgcmi] = min(movmean(imgcd,3,3),[],3);
        imgcmi(imgcm==0)=0;
        width = medfilt2(imgcmi,[5 5]);
end
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

function [G_veins,edgelist] = fnc_weighted_graph(edgelist,W_microns,skTree,MicronPerPixel)
[nY,nX] = size(W_microns);
% calculate the node indices to account for the edges added and removed
% I_idx is the linear index to the first pixel in each edge
I_idx = cellfun(@(x) sub2ind([nY nX],x(1,1),x(1,2)),edgelist);
% J_idx is the linear index to the last pixel in each edge array
J_idx = cellfun(@(x) sub2ind([nY nX],x(end,1),x(end,2)),edgelist);
% P_idx is the linear index of each pixel in the cell array of edges
P_idx = cellfun(@(x) sub2ind([nY nX],x(:,1),x(:,2)),edgelist,'UniformOutput',0);
% N_pix is the number of pixels in the edge list
N_pix = cellfun(@(x) numel(x(:,1)), edgelist);
% M_idx is the mid pixel index
M_idx = cellfun(@(x,y) x(round(y/2)), P_idx, num2cell(N_pix));
% Get various metrics associated with each edge from the various processing
% stages. Initially the values are returned as cell arrays with individual
% values for each pixel. These are then summarised to give a single value
% for each edge for that metric.
% Get the length as the calibrated difference in euclidean distance between pixels
L_val = cellfun(@(x) MicronPerPixel.*hypot(diff(x(:,1)),diff(x(:,2))),edgelist,'UniformOutput',0);
L_sum = cellfun(@sum, L_val);
% get the orientation using atan2
O_val = cellfun(@(x) atan2(x(1,1)-x(end,1),x(1,2)-x(end,2)),edgelist,'UniformOutput',1);
% Get the width
W_val = cellfun(@(x) W_microns(x),P_idx,'UniformOutput',0);
W_mean = cellfun(@mean, W_val);
% Set the edge ID
nEdges = length(I_idx);
EName = (1:nEdges)';
% set all edges ('E') to belong to a loop ('L') initially
EType = repmat({'EL'},nEdges,1);
% Set any edges that belong to a tree with 'T'. Use M_idx to sample the
% middle point in the skeleton
EType(skTree(M_idx)) = {'ET'};
% set the edge weight to the average width
E_weight = W_mean;
% add in the calibration factor
MPP = repmat(MicronPerPixel,nEdges,1);
% initially the nodes are empty
nodei = zeros(nEdges,1);
nodej = zeros(nEdges,1);
node_idx = unique([I_idx'; J_idx']);
% combine the edge metrics into an EdgeTable
names = {'EndNodes', 'node_Idx', 'Name', 'Type','Calibration', 'Weight', ...
    'Width_initial',  ...
    'Length_initial', ...
    'Orientation_initial', ...
    'N_pix', 'M_idx'};

EdgeTable = table([nodei, nodej], [I_idx', J_idx'], EName, EType, MPP, E_weight', ...
    W_mean', ...
    L_sum', ...
    O_val', ...
    N_pix', M_idx', ...
    'VariableNames', names);
% now collect all the node indices to construct a NodeTable. Get the index
% of all the nodes present in i or j
node_ID = (1:length(node_idx))';
node_type = repmat({'E'},length(node_idx),1);
% get the coordinates of each node
[y_pix,x_pix] = ind2sub([nY nX],node_idx);
% add in the calibration factor
MPP = repmat(MicronPerPixel,length(node_idx),1);
% Construct the full NodeTable (note at this point the co-ordinates are all
% in pixels
NodeTable = table(node_ID, node_idx, node_type, x_pix, y_pix, MPP, ...
    'VariableNames',{'node_ID' 'node_Idx' 'node_Type' 'node_X_pix' 'node_Y_pix' 'Calibration'});
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

function [G_veins,edgelist_center] = fnc_refine_width(G_veins,edgelist,im,im_cnn,W_microns,MicronPerPixel)
[nY,nX] = size(W_microns);
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
[edge_D0, max_idx] = max(A,[],2); % column format to match the table later
edge_D0 = full(edge_D0);
G_veins.Nodes.node_D0 = edge_D0;
% Calculate the width of the penultimate edge width by removing the max and
% recalculating the max for the remainder
rows = (1:size(A,1))';
mx_idx = sub2ind(size(A),rows,max_idx);
AD1 = A;
AD1(mx_idx) = 0;
G_veins.Nodes.node_D1 = full(max(AD1,[],2));
% Calculate the minimum edge width. To calculate the initial min of a
% sparse matrix, take the negative and then add back the maximum
[nnzr, nnzc] = find(A);
B = -A + sparse(nnzr,nnzc,max(edge_D0),size(A,1),size(A,2));
mn = max(B,[],2);
G_veins.Nodes.node_D2 = -(mn-max(edge_D0));
% get the degree for nodei and nodej
NDegI = G_veins.Nodes{G_veins.Edges.EndNodes(:,1),'node_Degree'};
NDegJ = G_veins.Nodes{G_veins.Edges.EndNodes(:,2),'node_Degree'};
% get the maximum edge weight for nodei and nodej
ND0I = G_veins.Nodes{G_veins.Edges.EndNodes(:,1),'node_D0'};
ND0J = G_veins.Nodes{G_veins.Edges.EndNodes(:,2),'node_D0'};
% Get the penultimate edge weight for nodei and nodej
ND1I = G_veins.Nodes{G_veins.Edges.EndNodes(:,1),'node_D1'};
ND1J = G_veins.Nodes{G_veins.Edges.EndNodes(:,2),'node_D1'};
% Get the number of pixels in the edge (after downSampling if used)
N_pix = G_veins.Edges.N_pix;
% Get the length as the calibrated difference in euclidean distance between pixels
L_val = cellfun(@(x) MicronPerPixel.*hypot(diff(x(:,1)),diff(x(:,2))),edgelist,'UniformOutput',0);
% L_sum is the cumulative length of the edge starting from nodei
L_sum = cellfun(@(x) cumsum(x),L_val,'UniformOutput',0);
% Initially use half the average (NAveI) or maximum (ND0I) at
% the node as the overlap distance to avoid. Limit the index to
% 50% of the number of pixels minus 1 (to ensure that there is
% a minimum length of edge remaining). Find where the index
% where cumulative length equals or exceeeds half the overlap
% width
L_idx = cellfun(@(x,y) find(x>=y/2,1),L_sum,num2cell(ND0I)','UniformOutput',0);
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
idx =  width >= ND1I | NDegI <= 2;
%idx =  NDegI <= 2 & NPerimI == 0;
first(idx) = 1;
% Repeat for node j, using half the largest edge width as an
% index into the array from a minimum of some fraction of the
% number of pixels + 1 (to ensure that there is a minimum edge
% length remaining). Switch the order to set nodej at the start
% for every edge
L_val = cellfun(@(x) flipud(x),L_val,'UniformOutput',0);
L_sum = cellfun(@(x) cumsum(x),L_val,'UniformOutput',0);
% use half of the average (NAveJ) or maximum (ND0J) at the
% node as the overlap distance to avoid. L_idx is the index
% into the flipped edgelist.
L_idx = cellfun(@(x,y) find(x>=y/2,1),L_sum,num2cell(ND0J)','UniformOutput',0);
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
idx =  width >= ND1J | NDegJ <= 2;
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
CL_val = cellfun(@(x) MicronPerPixel.*hypot(diff(x(:,1)),diff(x(:,2))),edgelist_center,'UniformOutput',0);
CL_sum = cellfun(@sum, CL_val);
% Get the width from the distance transform
CW_val = cellfun(@(x) W_microns(x),P_idx,'UniformOutput',0);
CW_mean = cellfun(@mean, CW_val);
CW_std = cellfun(@std, CW_val);
CW_cv = CW_std./CW_mean;
% % % % Get the width from the integral granulometry
% % % CWG_val = cellfun(@(x) W_granulometry(x),P_idx,'UniformOutput',0);
% % % CWG_mean = cellfun(@mean, CWG_val);
% % % CWG_std = cellfun(@std, CWG_val);
% % % CWG_cv = CWG_std./CWG_mean;
% get the average orientation of the center-width region. These
% will tend to be in the range pi/2 to -pi/2 because nodei for all
% edges is likely to be to the left of node j because of the
% re-ordering in the graph.
[midy, midx] = ind2sub([nY,nX],G_veins.Edges.M_idx);
mid = mat2cell([midy midx],ones(length(midx),1),2)';
% edgelist and mid co-ordinates are in [r,c] format i.e. [y,x]
% mid - i for ij and mid - j for ji
% flip y-axis so positive is counterclockwise from horizontal
CO_ij = cellfun(@(m,i) atan2(-(m(1,1)-i(1,1)), m(1,2)-i(1,2)),mid,edgelist,'UniformOutput',1);
CO_ji = cellfun(@(m,j) atan2(-(m(1,1)-j(end,1)), m(1,2)-j(end,2)),mid,edgelist,'UniformOutput',1);
% update the edgetable
G_veins.Edges.Length = CL_sum';
G_veins.Edges.Width = CW_mean';
G_veins.Edges.Width_cv = CW_cv';
% G_veins.Edges.Granulometry = CWG_mean';
% G_veins.Edges.Granulometry_cv = CWG_cv';
G_veins.Edges.Or_ij = rad2deg(CO_ij');
G_veins.Edges.Or_ji = rad2deg(CO_ji');
% calculate the tortuosity as the ratio of the distance between the end
% points (chord length) and the total length
CL_chord = cellfun(@(x) MicronPerPixel.*hypot(x(1,1)-x(end,1),x(1,2)-x(end,2)),edgelist_center);
G_veins.Edges.Tortuosity = CL_sum'./CL_chord';
G_veins.Edges.Tortuosity(isinf(G_veins.Edges.Tortuosity)) = 1;
% calculate the center-weighted radius
radius = G_veins.Edges.Width/2;
% calculate the center-weighted area
G_veins.Edges.Area = pi.*(radius.^2);
% calculate the center-weighted surface area
G_veins.Edges.SurfaceArea = pi.*2.*radius.*G_veins.Edges.Length ;
% calculate the center-weighted volume
G_veins.Edges.Volume = G_veins.Edges.Length.*G_veins.Edges.Area;
% calculate the center-weighted resistance to flow for area or Poiseuille flow
G_veins.Edges.R2 = G_veins.Edges.Length./(radius.^2);
G_veins.Edges.R4 = G_veins.Edges.Length./(radius.^4);
% find the feature edges and set them to the max
F_idx = find(strcmp(G_veins.Edges.Type,'F'));
G_veins.Edges.Width(F_idx) = max(CW_mean);
G_veins.Edges.Width_cv(F_idx) = 0;
% recalculate the node parameters from the center-width estimate
rows = (1:size(A,1))';
% calculate a weighted adjacency matrix for the center width
A = sparse(i,j,double(G_veins.Edges.Width),nN,nN);
A = A + A.' - diag(diag(A));
G_veins.Nodes.node_Strength = full(sum(A,2));
G_veins.Nodes.node_Average = G_veins.Nodes.node_Strength./G_veins.Nodes.node_Degree;
% calculate a weighted adjacency matrix for orientation for
% edges ij
Oij = sparse(i,j,double(G_veins.Edges.Or_ij),nN,nN);
Oji = sparse(j,i,double(G_veins.Edges.Or_ji),nN,nN);
O = Oij + Oji - diag(diag(Oij));
% Calculate the initial max edge width incident at the node (initially
% in double precision)
[edge_D0, max_idx] = max(A,[],2);
edge_D0 = full(edge_D0);
G_veins.Nodes.node_D0 = edge_D0;
mx_idx = sub2ind(size(A),rows,max_idx);
% get the orientation of the largest edge from the orientation adjacency
% matrix using the max_idx index
G_veins.Nodes.node_OD0 = full(O(mx_idx));
% Calculate the minimum edge width. To calculate the min of a
% sparse matrix, take the negative and then add back the maximum. This
% avoids finding a full array of zeros!
MaxA = sparse(i,j,max(edge_D0),nN,nN);
MaxA = MaxA + MaxA.' - diag(diag(MaxA));
B = -A + MaxA;
[edge_D2, D2_idx] = max(B,[],2);
mn_idx = sub2ind(size(A),rows,D2_idx);
% Calculate the width of the penultimate edge width by removing the max
% values from the adjacency matrix and recalculating max for the remainder
AD1 = A;
AD1(mx_idx) = 0;
[edge_D1, D1_idx] = max(AD1,[],2);
pn_idx = sub2ind(size(A),rows,D1_idx);
G_veins.Nodes.node_D1 = full(edge_D1);
% get the orientation of the penultimate edge from the orientation
% adjacency matrix using the pn_idx
G_veins.Nodes.node_OD1 = full(O(pn_idx));
% now add the D2 results to the table - convert back to absolute positive value by negating and adding back the max
G_veins.Nodes.node_D2 = -full(edge_D2)+max(edge_D0);
% get the orientation of the weakest edge from the orientation adjacency
% matric and the D2_idx
G_veins.Nodes.node_OD2 = full(O(mn_idx));% convert back to absolute positive value by negating and adding back the max
% tidy up results for k=1 nodes
idx = G_veins.Nodes.node_Degree == 1;
G_veins.Nodes.node_D1(idx) = nan;
G_veins.Nodes.node_D2(idx) = nan;
G_veins.Nodes.node_OD1(idx) = nan;
G_veins.Nodes.node_OD2(idx) = nan;
% % % % calculate the ratios for the radii of smallest to largest and
% % % % intermediate to largest
% % % G_veins.Nodes.node_D1_D0 = G_veins.Nodes.node_D1./G_veins.Nodes.node_D0;
% % % G_veins.Nodes.node_D2_D0 = G_veins.Nodes.node_D2./G_veins.Nodes.node_D0;
% % % G_veins.Nodes.node_D2_D1 = G_veins.Nodes.node_D2./G_veins.Nodes.node_D1;
G_veins.Nodes.node_AreaRatio = (G_veins.Nodes.node_D2+G_veins.Nodes.node_D1)./G_veins.Nodes.node_D0;
G_veins.Nodes.node_Symmetry = G_veins.Nodes.node_D2./G_veins.Nodes.node_D1;
% calculate the absolute angles around the branch
G_veins.Nodes.theta_D1_D0 = 180 - abs(180 - abs(G_veins.Nodes.node_OD1-G_veins.Nodes.node_OD0));
G_veins.Nodes.theta_D2_D0 = 180 - abs(180 - abs(G_veins.Nodes.node_OD2-G_veins.Nodes.node_OD0));
G_veins.Nodes.theta_D2_D1 = 180 - abs(180 - abs(G_veins.Nodes.node_OD2-G_veins.Nodes.node_OD1));
end

function [gray_CW,width,coded_CW,coded_FW] = fnc_coded_skeleton(im,sk,bw_mask,G_veins,edgelist,edgelist_center,sk_width,cmap)
[nY,nX] = size(sk);
% set up a blank image
gray_CW = zeros([nY,nX], 'single');
CW_center = zeros([nY,nX], 'single');
CW_full = zeros([nY,nX], 'single');
width = zeros([nY,nX], 'single');
% get the value of the edges from the graph. These are sorted
% automatically when the graph is set up in order of node i
E = G_veins.Edges.Width;
% get just the edges (not any features)
E_idx = contains(G_veins.Edges.Type,'E');
E = E(E_idx);
% get the linear pixel index for the center width each edge excluding the features
CW_idx = cellfun(@(x) sub2ind([nY nX],x(:,1),x(:,2)),edgelist_center(E_idx),'UniformOutput',0);
% concatenate all the pixels indices to give a single vector
P_CW = cat(1,CW_idx{:});
% calculate an edge width skeleton based on the central width
% for the selected edges
V_CW = num2cell(E);
% duplicate the value for the number of pixels in the center part of each edge
V_coded_idx = cellfun(@(x,y) repmat(y,length(x),1), CW_idx, V_CW','UniformOutput',0);
% concatenate all the values into a single vector
V_CW_all = cat(1,V_coded_idx{:});
% set the edge values to the central width for the center part of each
% edge. The CW_microns image is a grayscale image with the intensity
% of the skeleton making up each edge set to the width in pixels. The
% overlap region is set to zero.
gray_CW(P_CW) = single(V_CW_all);
% To construct the color-coded skeleton, normalise the range between 2 and
% 256 as an index into the colourmap
Emin = min(E);
Emax = max(E);
cmap_idx = ceil(254.*((E-Emin)./(Emax-Emin)))+2;
% set any nan values to 1
cmap_idx(isnan(cmap_idx)) = 1;
% convert to a cell array
V_coded = num2cell(cmap_idx);
% duplicate the value for the number of pixels in each edge
V_coded_idx = cellfun(@(x,y) repmat(y,length(x),1), CW_idx, V_coded','UniformOutput',0);
% concatenate all the values into a single vector
V_coded_all = cat(1,V_coded_idx{:});
% set the edge values to the central width for each edge in the image
% scaled between the min and the max
CW_center(P_CW) = single(V_coded_all);
if sk_width > 1
    sk_D = imdilate(sk, ones(sk_width));
    CW_center = imdilate(CW_center, ones(sk_width));
else
    sk_D = sk;
end
im2 = im;%uint8(255.*im);
im2(sk_D) = 0;
im_rgb = uint8(255.*ind2rgb(im2,gray(256)));
sk_rgb = uint8(255.*ind2rgb(uint8(CW_center),cmap));
coded_CW = imadd(im_rgb,sk_rgb);
% add the boundaries of the mask
coded_CW = imoverlay(coded_CW,bwperim(bw_mask),'m');
% Repeat the process but this time calculate the full pixel skeleton
% (including the overlap regions) and coded by the center weight for
% export. Get the linear pixel index for the full width each edge excluding the features
FW_idx = cellfun(@(x) sub2ind([nY nX],x(:,1),x(:,2)),edgelist(E_idx),'UniformOutput',0);
% concatenate all the pixels indices to give a single vector
P_FW = cat(1,FW_idx{:});
% calculate an edge width skeleton based on the central width
% for the selected edges
V_FW = num2cell(E);
% duplicate the value for the number of pixels in each edge
V_FW_idx = cellfun(@(x,y) repmat(y,length(x),1), FW_idx, V_FW','UniformOutput',0);
% concatenate all the values into a single vector
V_FW_all = cat(1,V_FW_idx{:});
% set the edge values to the central width for each edge in the image
width(P_FW) = single(V_FW_all);
% make sure that the junctions are set to the local maximum, so that
% junctions are preserved as part of the strongest edge
temp = colfilt(width,[3 3],'sliding',@max);
bp = bwmorph(sk,'branchpoints');
% The width image is a complete skeleton with no breaks in the overlap
% region coded by the center-weighted thickness of the edge
width(bp) = temp(bp);
% % set the masked regions to -1
width(~bw_mask) = -1;
%
%Calculate complete coded skeleton image using the center-weighted width
% duplicate the coded width value for the number of pixels in each full edge
V_FW_idx = cellfun(@(x,y) repmat(y,length(x),1), FW_idx, V_coded','UniformOutput',0);
% concatenate all the values into a single vector
V_FW_all = cat(1,V_FW_idx{:});
% set the edge values to the central width for each edge in the image
CW_full(P_FW) = single(V_FW_all);
% set the edge values to the central width for each edge in the image
% scaled between the min and the max
temp = colfilt(CW_full,[3 3],'sliding',@max);
% The width image is a complete skeleton with no breaks in the overlap
% region coded by the center-weighted thickness of the edge
CW_full(bp) = temp(bp);
if sk_width > 1
    CW_full = imdilate(CW_full, ones(sk_width));
end
sk_rgb = uint8(255.*ind2rgb(uint8(CW_full),cmap));
coded_FW = imadd(im_rgb,sk_rgb);
% add the boundaries of the mask
coded_FW = imoverlay(coded_FW,bwperim(bw_mask),'m');
end

function [G_veins, sk_polygon, bw_polygons, bw_areoles, polygon_LM] = fnc_polygon_find(G_veins,bw_cnn,sk,skLoop,skRing,skMask)
% % remove any partial areas that are not fully bounded and therefore contact
% % the edge
% area = imclearborder(~sk & skMask,4);
% % remove any partial areoles that are touching any masked regions
% [r,c] = find(~skMask);
% area_mask = ~(bwselect(area | ~skMask,c,r,4));
% % get rid of any vestigial skeleton lines within the mask
% area_mask = bwmorph(area_mask, 'open');
% % fill the skeleton lines on the total area
% total_area_mask = imdilate(area, [0 1 0; 1 1 1; 0 1 0]) & area_mask;
% % make sure the full boundary is present
% total_area_mask = total_area_mask | skLoop;
% % remove isolated polygons
% total_area_mask = bwareafilt(total_area_mask, 1,'largest');
% The polygons include the areoles and the vein itself.
bw_polygons = ~(skLoop | skRing) & skMask;
% % % % find areas that are too small (note bwareafilt does not work as it is not
% % % % possible to set connectivity to 4)
% % % CC = bwconncomp(bw_polygons,4);
% % % stats = regionprops(CC, 'Area');
% % % % arbitary threshold of ~3x3 pixels
% % % idx = find([stats.Area] > 9);
% % % bw_polygons  = ismember(labelmatrix(CC), idx);
% The areoles exclude the full width of the veins
bw_areoles = ~bw_cnn & skMask & bw_polygons;
% remove any orphan pixels
%bw_areoles = bwmorph(bw_areoles,'clean');
% trim the skeleton to match
sk_polygon = sk & skMask;
% construct a label matrix of areas including the veins
LM = bwlabel(bw_polygons,4);
% remove the skeleton in the calculation of the neighbouring
% area
LM(sk_polygon) = NaN;
% find the neighbours of each edge
Neighbour1 = colfilt(LM,[3 3],'sliding',@max);
Neighbour2 = colfilt(LM,[3 3],'sliding',@(x) min(x,[],'Omitnan'));
% Note some neighbours may link to the excluded areas and have a value of 0
% add the neighbours to the graph
G_veins.Edges.Ai = Neighbour1(G_veins.Edges.M_idx);
G_veins.Edges.Aj = Neighbour2(G_veins.Edges.M_idx);
% replace the skeleton pixels with zero again
LM(sk_polygon) = 0;
% dilate the label matrix to include the pixel skeleton
%polygon_LM = imdilate(LM,ones(3));
polygon_LM = imdilate(LM,[0 1 0; 1 1 1; 0 1 0]);
% trim off any extension beyond the boundary
polygon_LM = polygon_LM.*skMask;
% fill in any remaining pixels
polygon_LM = imfill(polygon_LM);
end

function im_polygons_rgb = fnc_polygon_image(polygon_stats, sk_polygon, skMask)
[nY,nX] = size(skMask);
% construct a colour-coded area image from the log area
idx = cat(1,polygon_stats.PixelIdxList{:});
polygon_areas = cellfun(@(x) length(x),polygon_stats.PixelIdxList,'UniformOutput',1);
%
logA = log10(polygon_areas);
logA(isinf(logA)) = nan;
Amin = 1;
Amax = log10(sum(skMask(:)));
% normalise the range between 2 and 256 as an index into the
% colourmap
A_idx = ceil(254.*((logA-Amin)./(Amax-Amin)))+1;
% set any nan values to 1
A_idx(isnan(A_idx)) = 1;
% duplicate the area value for the number of pixels in each edge
A_val = cellfun(@(x,y) repmat(x,y,1), num2cell(A_idx),num2cell(polygon_areas),'UniformOutput',0);
% concatenate all the values into a single vector
A_all = cat(1,A_val{:});
% set the edge values to the edge ID for each edge in the image
%im_polygons = zeros(nY,nX);
im_polygons = zeros(nY,nX, 'single');
im_polygons(idx) = single(A_all);
cmap = cool(256);
cmap(1,:) = 1;
% punch out the skeleton to make the polygons distinct
im_polygons(sk_polygon) = 0;
im_polygons = uint8(im_polygons);
im_polygons_rgb = uint8(255.*ind2rgb(im_polygons,cmap));
end

function [areole_stats,polygon_stats] = fnc_polygon_analysis(bw_polygons,bw_areoles, polygon_LM,FullMetrics,MicronPerPixel)
% measure the stats for the polygonal regions using the label matrix
if FullMetrics == 0
    % don't calculate the most time consuming metrics (ConvexArea,
    % Solidity) and the internal distance metrics.
    polygon_stats = regionprops('Table',polygon_LM, ...
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
    P_stats = regionprops('Table',polygon_LM, ...
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
        % calibrate the additional fields
    P_stats.ConvexArea = P_stats.ConvexArea.*MicronPerPixel^2;
    % get the maximum distance to the skeleton for each area
    D_stats = regionprops('Table',polygon_LM,bwdist(~bw_polygons)+0.5,'MaxIntensity','MeanIntensity');
%     % check whether format has switched to cell rather than number because
%     % values are not set)
%     D_stats.MeanIntensity(isnan(D_stats.MeanIntensity)) = 0.5;
%     if iscell(D_stats.MaxIntensity)
%         test4empty = cellfun(@isempty,D_stats.MaxIntensity);
%         if any(test4empty)
%             % replace the empty cell with the minimum distance of 0.5 pixel
%             [D_stats.MaxIntensity(test4empty)] = {[0.5]};
%             D_stats.MaxDistance = cell2mat(D_stats.MaxIntensity);
%         end
%     end
    % rename the fields
    D_stats.Properties.VariableNames([1 2]) = {'MaxDistance','MeanDistance'};
    % calibrate the distances
    D_stats.MaxDistance = D_stats.MaxDistance.*MicronPerPixel;
    D_stats.MeanDistance = D_stats.MeanDistance.*MicronPerPixel;
    % 2020a onwards D_stats = renamevars(D_stats,["MaxIntensity","MeanIntensity"],["MaxDistance","MeanDistance"]);
    polygon_stats = [P_stats D_stats];
end
% calibrate the remaining fields
polygon_stats.Area = polygon_stats.Area.*MicronPerPixel^2;
polygon_stats.EquivDiameter = polygon_stats.EquivDiameter.*MicronPerPixel;
polygon_stats.MajorAxisLength = polygon_stats.MajorAxisLength.*MicronPerPixel;
polygon_stats.MinorAxisLength = polygon_stats.MinorAxisLength.*MicronPerPixel;
polygon_stats.Perimeter = polygon_stats.Perimeter.*MicronPerPixel;
% calculate additional parameters
polygon_stats.polygon_ID = (1:height(polygon_stats))';
polygon_stats.Circularity = (4.*pi.*polygon_stats.Area)./(polygon_stats.Perimeter.^2);
polygon_stats.Elongation = polygon_stats.MajorAxisLength./polygon_stats.MinorAxisLength;
polygon_stats.Roughness = (polygon_stats.Perimeter.^2)./polygon_stats.Area;
% modify the label matrix to only include the areoles, but with the same ID
% as the polygons
areole_LM = polygon_LM;
areole_LM(~bw_areoles) = 0;
% run all the stats for the areoles
if FullMetrics == 0
    % don't calculate the most time consuming metrics (ConvexArea,
    % Solidity) and don't calculate the internal distance measures
    areole_stats = regionprops('Table',areole_LM, ...
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
    A_stats = regionprops('Table',areole_LM, ...
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
    % calibrate the additional fields
    A_stats.ConvexArea = A_stats.ConvexArea.*MicronPerPixel^2;
    % get the maximum distance to the skeleton for each area
    D_stats = regionprops('Table',areole_LM,bwdist(~bw_areoles)+0.5,'MaxIntensity','MeanIntensity');
    % check whether format has switched to cell rather than number because
    % values are not set)
%    idx = ismissing(D_stats);
%    D_stats(idx) = 0.5
%     if iscell(D_stats.MeanIntensity)
%     test4empty = cellfun(@isempty,D_stats.MeanIntensity);
%     if any(test4empty)
%         % replace the empty ell with the minimum distance of 0.5 pixel
%         D_stats.MeanIntensity(test4empty) = deal(0.5);
%         D_stats.MeanIntensity = cell2mat(D_stats.MeanIntensity);
%     end
%     
%     end
    % rename the fields and calibrate
    % 2020a D_stats = renamevars(D_stats,["MaxIntensity","MeanIntensity"],["MaxDistance","MeanDistance"]);
    D_stats.Properties.VariableNames([1 2]) = {'MaxDistance','MeanDistance'};
    D_stats.MaxDistance = D_stats.MaxDistance.*MicronPerPixel;
    D_stats.MeanDistance = D_stats.MeanDistance.*MicronPerPixel;
    areole_stats = [A_stats D_stats];
end
% calibrate the remaining fields
areole_stats.Area = areole_stats.Area.*MicronPerPixel^2;
areole_stats.EquivDiameter = areole_stats.EquivDiameter.*MicronPerPixel;
areole_stats.MajorAxisLength = areole_stats.MajorAxisLength.*MicronPerPixel;
areole_stats.MinorAxisLength = areole_stats.MinorAxisLength.*MicronPerPixel;
areole_stats.Perimeter = areole_stats.Perimeter.*MicronPerPixel;
% calculate additional parameters
areole_stats.areole_ID = (1:height(areole_stats))';
areole_stats.Circularity = (4.*pi.*areole_stats.Area)./(areole_stats.Perimeter.^2);
areole_stats.Elongation = areole_stats.MajorAxisLength./areole_stats.MinorAxisLength;
areole_stats.Roughness = (areole_stats.Perimeter.^2)./areole_stats.Area;
end

function [G_areas,LM] = fnc_area_graph(G_veins,NodeTable,LM,type,MicronPerPixel)
% Construct a NodeTable with the node for each area, along with the
% corresponding metrics
% extract the centroid values
%polygon_Centroid = cat(1,NodeTable.Centroid);
NodeTable.node_X_pix = NodeTable.Centroid(:,1);
NodeTable.node_Y_pix = NodeTable.Centroid(:,2);
% remove the centroid and PixelIdxList fields
% 2020a NodeTable = removevars(NodeTable,{'Centroid';'PixelIdxList'});
NodeTable.Centroid = [];
NodeTable.PixelIdxList = [];
NodeTable.Calibration = repmat(MicronPerPixel,height(NodeTable),1);
% reorder the table to get the ID and co-ordinates first
NodeTable = NodeTable(:,[end-3:end, 1:end-4]);
% Construct an EdgeTable with the width of the pixel skeleton edge that it
% crosses
names = {'EndNodes' 'Width' 'Name' 'Calibration'};
% If an area has a k=1 edge within it (i.e. a terminal vein) there could be
% two possible edges to an adjacent area on either side with the same ID.
% Therefore duplicate edges are resolved to the minimum. Reorder nodes to
% be minimum first
i = min(G_veins.Edges.Ai, G_veins.Edges.Aj);
j = max(G_veins.Edges.Ai, G_veins.Edges.Aj);
edges = [i j G_veins.Edges.Width G_veins.Edges.Name];
% if any edge links to an excluded region Ai or Aj will have a value of
% zero. Therefore exclude these edges from analysis.
edges(i==0 | j==0,:) = [];
% sort by the edge width and keep the smallest
edges = sortrows(edges,3);
[~,idx] = unique(edges(:,1:2),'rows');
edges = double(edges(idx,:));
% remove edges connected to the background
idx = max(edges(:,1:2)==0, [],2);
edges(idx,:) = [];
% remove edges connected to themselves
idx = diff(edges(:,1:2),[],2)==0;
edges(idx,:) = [];
% calibration factor
Calibration = repmat(MicronPerPixel,size(edges,1),1);
% create the edgetable
EdgeTable = table([edges(:,1) edges(:,2)],edges(:,3),edges(:,4),Calibration, 'VariableNames', names);
G_areas = graph(EdgeTable,NodeTable,'OmitSelfLoops');
% check the number of components and only keep the largest
% [CC, binsizes] = conncomp(G_areas);
% [~,GCC] = max(binsizes);
CC = conncomp(G_areas);
[N,~] = histcounts(CC,max(CC));
[~,GCC] = max(N);
% only keep the connected areas in the label matrix
switch type
    case 'areoles'
        LM(~ismember(LM,G_areas.Nodes.areole_ID(CC==GCC))) = 0;
    case 'polygons'
        LM(~ismember(LM,G_areas.Nodes.polygon_ID(CC==GCC))) = 0;
end
G_areas = rmnode(G_areas,find(CC~=GCC));
end

function T = fnc_summary_veins(G_veins,total_area,polygon_area)
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
% get the total length of veins (in microns)
T.VTotL = sum(G_veins.Edges.Length(E_idx)); %
T.VLoopL = sum(G_veins.Edges.Length(L_idx)); %
T.VTreeL = sum(G_veins.Edges.Length(T_idx)); %
% get the total volume of the veins
T.VTotV = sum(G_veins.Edges.Volume(E_idx)); %
T.VLoopV = sum(G_veins.Edges.Volume(L_idx)); %
T.VTreeV = sum(G_veins.Edges.Volume(T_idx)); %
% get the weighted vein width
T.VTotW = (1/T.VTotL).*(sum(G_veins.Edges.Length(E_idx).*G_veins.Edges.Width(E_idx)));
T.VLoopW = (1/T.VLoopL).*(sum(G_veins.Edges.Length(L_idx).*G_veins.Edges.Width(L_idx)));
T.VTreeW = (1/T.VTreeL).*(sum(G_veins.Edges.Length(T_idx).*G_veins.Edges.Width(T_idx)));
% set the graph weight to be the length;
G_veins.Edges.Weight = G_veins.Edges.Length;
% total areas analysed
T.TotA = total_area;
T.TotPA = polygon_area;
% get the density measurements
T.VTotLD = T.VTotL / T.TotA;
T.VLoopLD = T.VLoopL / T.TotA;
T.VTreeLD = T.VTreeL / T.TotA;
T.VTotVD = T.VTotV / T.TotA;
T.VLoopVD = T.VLoopV / T.TotA;
T.VTreeVD = T.VTreeV / T.TotA;
T.VNND = T.VNN / T.TotA;
% get the summary statistics for all relevant edge metrics in the table
% usage: T = fnc_summary(T,metric,prefix,suffix,transform)
T = fnc_summary(T,G_veins.Edges.Length(E_idx),'VTot','Len','none');
T = fnc_summary(T,G_veins.Edges.Length(L_idx),'VLoop','Len','none');
T = fnc_summary(T,G_veins.Edges.Length(T_idx),'VTree','Len','none');
T = fnc_summary(T,G_veins.Edges.Width(E_idx),'VTot','Wid','none');
T = fnc_summary(T,G_veins.Edges.Width(L_idx),'VLoop','Wid','none');
T = fnc_summary(T,G_veins.Edges.Width(T_idx),'VTree','Wid','none');
T = fnc_summary(T,G_veins.Edges.SurfaceArea(E_idx),'VTot','SAr','none');
T = fnc_summary(T,G_veins.Edges.SurfaceArea(L_idx),'VLoop','SAr','none');
T = fnc_summary(T,G_veins.Edges.SurfaceArea(T_idx),'VTree','SAr','none');
T = fnc_summary(T,G_veins.Edges.Volume(E_idx),'VTot','Vol','none');
T = fnc_summary(T,G_veins.Edges.Volume(L_idx),'VLoop','Vol','none');
T = fnc_summary(T,G_veins.Edges.Volume(T_idx),'VTree','Vol','none');
T = fnc_summary(T,G_veins.Edges.Tortuosity(E_idx),'V','Tor','none');
T = fnc_summary(T,G_veins.Edges.Or_ij(E_idx),'V','Ori','circ');
T = fnc_summary(T,G_veins.Nodes.node_D2,'N','BD2','none');
T = fnc_summary(T,G_veins.Nodes.node_D1,'N','BD1','none');
T = fnc_summary(T,G_veins.Nodes.node_D0,'N','BD0','none');
% T = fnc_summary(T,G_veins.Nodes.node_D2_D1,'N','Rnd','none');
% T = fnc_summary(T,G_veins.Nodes.node_D2_D0,'N','Rnx','none');
% T = fnc_summary(T,G_veins.Nodes.node_D1_D0,'N','Rdx','none');
T = fnc_summary(T,G_veins.Nodes.theta_D2_D1,'N','A21','circ');
T = fnc_summary(T,G_veins.Nodes.theta_D2_D0,'N','A20','circ');
T = fnc_summary(T,G_veins.Nodes.theta_D1_D0,'N','A10','circ');
end

function T = fnc_summary_areoles(G_areoles,polygon_area,FullMetrics)
T = table;
T.ATA = sum(G_areoles.Nodes.Area);
T.ANN = numnodes(G_areoles);
PTA = polygon_area;
T.Aloop = T.ANN / PTA; % should be the same as T.Ploop
% get areole statistics
T = fnc_summary(T,G_areoles.Nodes.Area,'A','Are','none');
T = fnc_summary(T,G_areoles.Nodes.Eccentricity,'A','Ecc','none');
T = fnc_summary(T,G_areoles.Nodes.MajorAxisLength,'A','Maj','none');
T = fnc_summary(T,G_areoles.Nodes.MinorAxisLength,'A','Min','none');
T = fnc_summary(T,G_areoles.Nodes.EquivDiameter,'A','EqD','none');
T = fnc_summary(T,G_areoles.Nodes.Perimeter,'A','Per','none');
T = fnc_summary(T,G_areoles.Nodes.Elongation,'A','Elg','none');
T = fnc_summary(T,G_areoles.Nodes.Circularity,'A','Cir','none');
T = fnc_summary(T,G_areoles.Nodes.Roughness,'A','Rgh','none');
T = fnc_summary(T,G_areoles.Nodes.Orientation,'A','Ori','circ');
% add in additional metrics if required
if FullMetrics == 1
    T = fnc_summary(T,G_areoles.Nodes.ConvexArea,'A','CnA','none');
    T = fnc_summary(T,G_areoles.Nodes.Solidity,'A','Sld','none');
    T = fnc_summary(T,G_areoles.Nodes.MeanDistance,'A','Dav','none');
    T = fnc_summary(T,G_areoles.Nodes.MaxDistance,'A','Dmx','none');
end
end

function T = fnc_summary_polygons(G_polygons,FullMetrics)
T = table;
T.PTA = sum(G_polygons.Nodes.Area);
T.PNN = numnodes(G_polygons);
T.Ploop = T.PNN / T.PTA; % should be the same as T.Aloop
% get polgonal area statistics
T = fnc_summary(T,G_polygons.Nodes.Area,'P','Are','none');
T = fnc_summary(T,G_polygons.Nodes.Eccentricity,'P','Ecc','none');
T = fnc_summary(T,G_polygons.Nodes.MajorAxisLength,'P','Maj','none');
T = fnc_summary(T,G_polygons.Nodes.MinorAxisLength,'P','Min','none');
T = fnc_summary(T,G_polygons.Nodes.EquivDiameter,'P','EqD','none');
T = fnc_summary(T,G_polygons.Nodes.Perimeter,'P','Per','none');
T = fnc_summary(T,G_polygons.Nodes.Elongation,'P','Elg','none');
T = fnc_summary(T,G_polygons.Nodes.Circularity,'P','Cir','none');
T = fnc_summary(T,G_polygons.Nodes.Roughness,'P','Rgh','none');
T = fnc_summary(T,G_polygons.Nodes.Orientation,'P','Ori','circ');
% add in additional metrics if required
if FullMetrics == 1
    T = fnc_summary(T,G_polygons.Nodes.ConvexArea,'P','CnA','none');
    T = fnc_summary(T,G_polygons.Nodes.Solidity,'P','Sld','none');
    T = fnc_summary(T,G_polygons.Nodes.MeanDistance,'P','Dav','none');
    T = fnc_summary(T,G_polygons.Nodes.MaxDistance,'P','Dmx','none');
end
end

function T = fnc_summary(T,metric,prefix,suffix,transform)
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
        T.([prefix 'av' suffix]) = circ_rad2ang(circ_mean(deg2rad(metric)));
        %         don't calculate the median as the intermediate array size is Ne x NE
        %         T.([prefix 'md' suffix]) = circ_rad2ang(circ_median(deg2rad(metric))); %
        %         T.([prefix 'mn' suffix]) = circ_rad2ang(min(deg2rad(metric)));
        %         T.([prefix 'mx' suffix]) = circ_rad2ang(max(deg2rad(metric)));
        T.([prefix 'sd' suffix]) = circ_rad2ang(circ_std(deg2rad(metric)));
        %         T.([prefix 'sk' suffix]) = circ_skewness(deg2rad(metric)));
        warning on
    otherwise
        T.([prefix 'av' suffix]) = mean(metric);
        T.([prefix 'md' suffix]) = median(metric);
        %         T.([prefix 'mo' suffix]) = mode(metric);
        %         T.([prefix 'mn' suffix]) = min(metric);
        %         T.([prefix 'mx' suffix]) = max(metric);
        T.([prefix 'sd' suffix]) = std(metric);
        %         T.([prefix 'sk' suffix]) = skewness(metric);
end
end

function [G_HLD, HLD_image, parent] = fnc_HLD(G_veins, G_polygons, polygon_stats, areole_stats, polygon_LM, bw_polygons, MicronPerPixel)
% construct a binary polygon CC object
PCC.Connectivity = 4;
PCC.ImageSize = size(polygon_LM);
PCC.NumObjects = 1;
PCC.PixelIdxList = {};
% find all the polygons on the boundary. The label matrix has already been
% dilated to remove the internal skeleton lines, so the only background
% values will be at the edges or internal masked regions. If these are dilated using
% a 3x3 kernel, they should overlap any polygons adjacent to the boundary.
% These can then be extracted using the dilated boundary region as seed points.
% The set of polygons touching the boundary can then be used to set a
% boundary flag in the G_polygons graph.
boundary = polygon_LM==0;
boundary = imdilate(boundary,ones(3));
[r,c] = find(boundary);
B_polygons = bwselect(bw_polygons,c,r,4);
B_polygons = imerode(B_polygons, ones(3));
% get the ID of the boundary polygons from the label matrix
Bidx = unique(polygon_LM(B_polygons));
Bidx(Bidx==0) = [];
%
% % % % get the pixelidxlist from the polygon_stats array to check the correct
% % % % polygons have been identified
% % % Bpix = cat(1,polygon_stats(Bidx).PixelIdxList);
% % % Bim = zeros(size(B_polygons));
% % % Bim(Bpix) = 1;
% % % figure
% % % imshow(single(cat(3,B_polygons,bw_polygons,Bim)))
% % % %
% add in a boundary flag if the node is a boundary polygon
G_polygons.Nodes.Boundary = ismember(G_polygons.Nodes.polygon_ID,Bidx);
% select the largest component of the sub-graph
% [CC, binsizes] = conncomp(G_polygons);
% [~,GCC] = max(binsizes);
% SG_GCC = subgraph(G_polygons,find(CC==GCC));
CC = conncomp(G_polygons);
[N,~] = histcounts(CC,max(CC));
[~,GCC] = max(N);
SG_GCC = subgraph(G_polygons,find(CC==GCC));
% % % %
% % % hold on
% % % plot(SG_GCC, 'XData',SG_GCC.Nodes.node_X_pix,'YData',SG_GCC.Nodes.node_Y_pix, ...
% % %     'NodeColor','r','MarkerSize',1,'Marker', 'none', 'NodeLabel', [], ...
% % %     'EdgeColor','r','EdgeAlpha',1,'EdgeLabel', [],'LineWidth',1);
% % % scatter(SG_GCC.Nodes.node_X_pix(~SG_GCC.Nodes.Boundary),SG_GCC.Nodes.node_Y_pix(~SG_GCC.Nodes.Boundary),48,'y','o')
% % % scatter(SG_GCC.Nodes.node_X_pix(SG_GCC.Nodes.Boundary),SG_GCC.Nodes.node_Y_pix(SG_GCC.Nodes.Boundary),48,'m','o')
% % % %
%
% extract the same component from the stats arrays. These arrays contain
% data for all polygons orginally identified and given a unique number (ID)
polygon_stats = polygon_stats(SG_GCC.Nodes.polygon_ID,:);
% Keep veins from the vein graph that form part of the polygon_graph. These
% will be the boundary edges and any internal tree-like parts of the
% network, but will exclude edges from incomplete polygons on the boundary
% or disconnected polygons. Edges should have Ai and/or
% Aj corresponding to a polygon node ID.
%Vidx = ismember(G_veins.Edges.Ai,idx) | ismember(G_veins.Edges.Aj,idx);
Vidx = ismember(G_veins.Edges.Ai,SG_GCC.Nodes.polygon_ID) | ismember(G_veins.Edges.Aj,SG_GCC.Nodes.polygon_ID);
G_veins = rmedge(G_veins,find(~Vidx));
% only keep veins that are still connected to the largest component
% [CC, binsizes] = conncomp(G_veins);
% [~,GCC] = max(binsizes);
% G_veins = subgraph(G_veins,find(CC==GCC));
CC = conncomp(G_veins);
[N,~] = histcounts(CC,max(CC));
[~,GCC] = max(N);
G_veins = subgraph(G_veins,find(CC==GCC));
% % % %
% % % plot(G_veins, 'XData',G_veins.Nodes.node_X_pix,'YData',G_veins.Nodes.node_Y_pix, ...
% % %     'NodeColor','b','MarkerSize',1,'Marker', 'none', 'NodeLabel', [], ...
% % %     'EdgeColor','b','EdgeAlpha',1,'EdgeLabel', [],'LineWidth',1);
% % % %
% calculate the initial length and MST ratio for the veins
L = sum(G_veins.Edges.Length);
% set the weights in the vein graph to length
G_veins.Edges.Weight = G_veins.Edges.Length;
MST = minspantree(G_veins,'method','sparse');
MSTL = sum(MST.Edges.Weight)/L;
% get the number of nodes and edge in the dual graph
nnP = numnodes(SG_GCC);
neP = numedges(SG_GCC);
parent = zeros(1,nnP);
width_threshold = zeros((2*nnP)-1,1);
node_Boundary = [SG_GCC.Nodes{:,'Boundary'}; zeros(nnP-1,1)];
node_Area = [SG_GCC.Nodes{:,'Area'}; zeros(nnP-1,1)];
node_Degree = [ones(nnP,1); zeros(nnP-1,1)];
degree_Asymmetry = zeros(nnP*2-1,1);
area_Asymmetry = zeros(nnP*2-1,1);
node_HS = [ones(nnP,1); zeros(nnP-1,1)];
subtree_degree_Asymmetry = zeros((2*nnP)-1,1);
subtree_area_Asymmetry = zeros((2*nnP)-1,1);
VTotLen = [repmat(L,nnP,1); zeros(nnP-1,1)];
VTotVol = [repmat(L,nnP,1); zeros(nnP-1,1)];
MSTRatio = [repmat(MSTL,nnP,1); zeros(nnP-1,1)];
% redimension the stat arrays to accomodate all the fused nodes
warning off
areole_stats.Area((nnP.*2-1),1) = 0;
polygon_stats.Area((nnP.*2-1),1) = 0;
warning on
% set up a list of the initial edges sorted by width
% ET = [nodei nodej W];
ET = [SG_GCC.Edges.EndNodes(:,1) SG_GCC.Edges.EndNodes(:,2) SG_GCC.Edges{:,'Width'}];
% sort by width
ET = sortrows(ET,3,'ascend');
% start the index for the new node (Nk) to follow on the number of existing
% nodes (nnP)
Nk = nnP;
Ne = 0;
% get all the current polygon PixelIdxLists at the start as a cell array;
P_PIL = polygon_stats.PixelIdxList;
PCC.NumObjects = length(P_PIL);
PCC.PixelIdxList  = polygon_stats.PixelIdxList;
LM = labelmatrix(PCC);
% start the HLD_image with the pixel skeleton
HLD_image = uint8(~bw_polygons);
% % % %
% % % visboundaries(LM>0);drawnow
% % % %
P_stats = regionprops('Table',LM,'Area','Centroid','Perimeter','MajorAxisLength','MinorAxisLength','Eccentricity','Orientation','Image','BoundingBox','PixelList');
% set up the endnodes
EndNodes = zeros(nnP*2-2,2);
% loop through all the edges, calculating the metrics
for iE = 1:neP
    % test each edge in sequence
    Ni = ET(1,1);
    Nj = ET(1,2);
    % the edge to be removed only links two areas if
    % the end nodes are different
    if Ni~=Nj
        % create a new node
        Nk = Nk+1;
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
        P_stats(Nk,:) = regionprops('table',PCC,'Area','Centroid','Perimeter','MajorAxisLength','MinorAxisLength','Eccentricity','Orientation','Image','BoundingBox','PixelList');
        % construct an image of the HLD level
        B = bwboundaries(P_stats.Image{Nk,1});
        Ridx = sub2ind(size(HLD_image),floor(B{1}(:,1)+P_stats.BoundingBox(Nk,2)),floor(B{1}(:,2)+P_stats.BoundingBox(Nk,1)));
        HLD_image(Ridx) = HLD_image(Ridx)+1;
        % find edges in the vein graph up to and including this edge width
        Eidx = G_veins.Edges.Width <= width_threshold(Nk,1);
        % remove these edges from the graph
        G_veins = rmedge(G_veins,find(Eidx));
        % only keep veins that are still connected to the largest component
        CC = conncomp(G_veins);
        [N,~] = histcounts(CC,max(CC));
        [~,GCC] = max(N);
        G_veins = subgraph(G_veins,find(CC==GCC));
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
Circularity = (4.*pi.*P_stats.Area)./(P_stats.Perimeter.^2);
Elongation = P_stats.MajorAxisLength./P_stats.MinorAxisLength;
Roughness = (P_stats.Perimeter.^2)./P_stats.Area;
% assemble the HLD graph object
NodeTable = table((1:(2*nnP)-1)', width_threshold, ...
    node_Area, ...
    node_Degree, ...
    degree_Asymmetry, ...
    subtree_degree_Asymmetry, ...
    area_Asymmetry, ...
    subtree_area_Asymmetry, ...
    node_HS, ...
    P_stats.Perimeter.*MicronPerPixel, ...
    P_stats.MajorAxisLength.*MicronPerPixel, ...
    P_stats.MinorAxisLength.*MicronPerPixel, ...
    P_stats.Eccentricity, ...
    P_stats.Orientation, ...
    Circularity, ...
    Elongation, ...
    Roughness, ...
    VTotLen, ...
    VTotVol, ...
    MSTRatio, ...
    node_Boundary, ...
    'VariableNames',{'node_ID', 'width_threshold', 'node_Area', 'node_Degree', ...
    'degree_Asymmetry',  'subtree_degree_Asymmetry', 'area_Asymmetry',  'subtree_area_Asymmetry', ...
    'node_HS','Perimeter','MajorAxisLength', 'MinorAxisLength', 'Eccentricity','Orientation','Circularity','Elongation','Roughness','VTotLen','VTotVol','MSTRatio','Boundary'});
EdgeTable = table(EndNodes, ...
    'VariableNames', {'EndNodes'});
idx = EdgeTable.EndNodes(:,1)==0;
EdgeTable(idx,:) = [];
idx = EdgeTable.EndNodes(:,2)==0;
EdgeTable(idx,:) = [];
% combine to form the HLD tree graph
G_HLD = graph(EdgeTable, NodeTable);
end

function hfig = display_figure(images,graphs,titles,G,E_width,links,name,ExportFigs)
%hfig = figure('Renderer','painters');
hfig = figure; % remote only uses software opengl
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
        h = title(titles{ia},'fontsize',18,'fontweight','normal','interpreter','none');
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
    warning('off','MATLAB:prnRenderer:opengl');
    export_fig(name,'-png','-pdf','-r300',hfig)
    %     saveas(hfig,name)
end
delete(hfig);
end

function hfig = display_HLD_v1(G_polygons,im_cnn,G_HLD,FullLeaf,name,ExportFigs)
%hfig = figure('Renderer','painters');
hfig = figure; % remote only uses software opengl
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
    warning('off','MATLAB:prnRenderer:opengl');
    %export_fig(name,'-png','-pdf','-r300','-painters',hfig)
    export_fig(name,'-png','-pdf','-r300',hfig)
    %     saveas(hfig,name)
end
delete(hfig);
end

function hfig = display_HLD_v2(G_polygons,im_cnn,G_HLD,FullLeaf,name,ExportFigs)
%hfig = figure('Renderer','painters');
hfig = figure; % remote only uses software opengl
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
% if FullLeaf == 1
last = 1;
g1 = G_HLD;
% else
%     % extract the subgraph for nodes that are not linked to a boundary node
%     g1 = subgraph(G_HLD,find(~G_HLD.Nodes.Boundary));
%     % extract the largest five fully connected subgraphs
%     last = 5;
% end
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
    %highlight(h,g2.Nodes.node_ID(g2.Edges.EndNodes(:,1)),g2.Nodes.node_ID(g2.Edges.EndNodes(:,2)),'NodeColor',cols{icc},'Marker','o','EdgeColor',cols{icc},'LineWidth',1)
    % get the node-IDs for the subtree to overlay on the image
    Gidx = g1.Nodes.node_ID(cc{idx(icc)});
    % limit the nodes to display to the initial nodes
    Gidx(Gidx>numnodes(G_polygons)) = [];
    % extract the subgraph from the original dual-graph that contain the selected nodes
    g3 = subgraph(G_polygons,Gidx);
    % display the subgraph on the image
    plot(ax(1),g3, 'XData',g3.Nodes.node_X_pix,'YData',g3.Nodes.node_Y_pix, ...
        'NodeColor','g','MarkerSize',1,'Marker', 'none', 'NodeLabel', [], ...
        'EdgeColor','g','EdgeAlpha',1,'EdgeLabel', [],'LineWidth',1);
    
    ax(6) = subplot(3,3,6);
    plot(ax(6),log(g2.Nodes.width_threshold), movmedian(log(g2.Nodes.VTotLen),15),'LineStyle','-','Marker','.','Color','k')
    hold on
    xlim(width_limits)
    xlabel('log(width)')
    ylabel('Vein density')
    box on
    
    % plot the area
    ax(7) = subplot(3,3,7);
    plot(ax(7),log(g2.Nodes.width_threshold), movmedian(log(g2.Nodes.node_Area),15),'LineStyle','-','Marker','.','Color','k')
    xlabel('log(width)')
    ylabel('log(area)')
    xlim(width_limits)
    %ylim([-6 0])
    hold on
    box on
    
    % plot the MSTRatio
    ax(8) = subplot(3,3,8);
    plot(ax(8),log(g2.Nodes.width_threshold), movmedian(g2.Nodes.MSTRatio,15),'LineStyle','-','Marker','.','Color','k')
    xlabel('log(width)')
    ylabel('MST ratio')
    xlim(width_limits)
    ylim([0.5 1])
    hold on
    box on
    
    % plot the Circularity
    ax(9) = subplot(3,3,9);
    plot(ax(9),log(g2.Nodes.width_threshold), movmedian(g2.Nodes.Circularity,15),'LineStyle','-','Marker','.','Color','k')
    xlabel('log(width)')
    ylabel('circularity')
    xlim(width_limits)
    ylim([0.2 1])
    hold on
    box on
    
end
drawnow
if ExportFigs
    warning('off','MATLAB:prnRenderer:opengl');
    %export_fig(name,'-png','-pdf','-r300','-painters',hfig)
    export_fig(name,'-png','-pdf','-r300',hfig)
    %     saveas(hfig,name)
end
delete(hfig);
end

function hfig = display_HLD_figure(G_polygons,im_cnn,G_HLD,FullLeaf,name,ExportFigs)
%hfig = figure('Renderer','painters');
hfig = figure; % remote only uses software opengl
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
    warning('off','MATLAB:prnRenderer:opengl');
    %export_fig(name,'-png','-r600','-painters',hfig)
    export_fig(name,'-png','-pdf','-r300',hfig)
    %     saveas(hfig,name)
end
delete(hfig);
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


