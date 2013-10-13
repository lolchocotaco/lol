%% Loading things

base = imread('./img/thresh1.png');

BL = base(end/4:end,1:end/4,:);

imdir = './Champions/';
imf = dir(imdir);
imf = imf(3:end);
champs = cell([numel(imf) 1]);
for i = 1:numel(imf)
%     disp(imf(i).name);
    champs{i} = imread(strcat(imdir,imf(i).name));
end

champ_thresh = champs{93};

%% FIND THE SQUARE

I = rgb2gray(BL);
th = graythresh(I);
I_th = im2bw(I,th);

Ifill = imfill(I_th,'holes');
Iarea = bwareaopen(Ifill,100);
Ifinal = bwlabel(Iarea);
stat = regionprops(Ifinal,'boundingbox');
imshow(I); hold on;
bb = zeros([numel(stat) 4]);
for cnt = 1 : numel(stat)
    bb(cnt,:) = stat(cnt).BoundingBox;
    rectangle('position',bb(cnt,:),'edgecolor','r','linewidth',2);
end
% [x y w h]
% sqness 
sq = (bb(:,3).*bb(:,4)) - (max(bb(:,3:4),[],2)-min(bb(:,3:4),[],2));
[~,ind] = max(sq);
charBox = round(bb(ind,:));
rectangle('position',bb(ind,:),'edgecolor','g','linewidth',2);

charImg = BL(charBox(2):charBox(2)+charBox(4),charBox(1):charBox(1)+charBox(3),:);

figure
imshow(charImg)

%% FIND THE CHAMP







%%




%%

% Demo to use normxcorr2 to find a template (a white onion)
% in a larger image (of a pile of vegetables)
% clc;    % Clear the command window.
% close all;  % Close all figures (except those of imtool.)
% imtool close all;  % Close all imtool figures.
% % clear;  % Erase all existing variables.
% workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;
fontSize = 20;
% Check that user has the Image Processing Toolbox installed.
hasIPT = license('test', 'image_toolbox');
if ~hasIPT
	% User does not have the toolbox installed.
	message = sprintf('Sorry, but you do not seem to have the Image Processing Toolbox.\nDo you want to try to continue anyway?');
	reply = questdlg(message, 'Toolbox missing', 'Yes', 'No', 'Yes');
	if strcmpi(reply, 'No')
		% User said No, so exit.
		return;
	end
end
% Read in a standard MATLAB color demo image.
folder = fullfile(matlabroot, '\toolbox\images\imdemos');
baseFileName = 'peppers.png';
% Get the full filename, with path prepended.
fullFileName = fullfile(folder, baseFileName);
if ~exist(fullFileName, 'file')
	% Didn't find it there.  Check the search path for it.
	fullFileName = baseFileName; % No path this time.
	if ~exist(fullFileName, 'file')
		% Still didn't find it.  Alert user.
		errorMessage = sprintf('Error: %s does not exist.', fullFileName);
		uiwait(warndlg(errorMessage));
		return;
	end
end

% rgbImage = imread(fullFileName);
rgbImage = BL;


% Get the dimensions of the image.  numberOfColorBands should be = 3.
[rows columns numberOfColorBands] = size(rgbImage);
% Display the original color image.
subplot(2, 2, 1);
imshow(rgbImage, []);
axis on;
title('Original Color Image', 'FontSize', fontSize);
% Enlarge figure to full screen.
set(gcf, 'units','normalized','outerposition',[0, 0, 1, 1]);
% Let's get our template by extracting a small portion of the original image.
templateWidth = 71
templateHeight = 49
smallSubImage = imcrop(rgbImage, [192, 82, templateWidth, templateHeight]);

smallSubImage = champ_thresh;


subplot(2, 2, 2);
imshow(smallSubImage, []);
axis on;
title('Template Image to Search For', 'FontSize', fontSize);
% Ask user which channel to search for a match.
% channelToCorrelate = menu('Correlate which color channel?', 'Red', 'Green', 'Blue');
% It actually finds the same location no matter what channel you pick, 
% for this image anyway, so let's just go with red (channel #1).
channelToCorrelate = 1;  
correlationOutput = normxcorr2(smallSubImage(:,:,1), rgbImage(:,:, channelToCorrelate));
subplot(2, 2, 3);
imshow(correlationOutput, []);
axis on;
title('Normalized Cross Correlation Output', 'FontSize', fontSize);
% Find out where the normalized cross correlation image is brightest.
[maxCorrValue, maxIndex] = max(abs(correlationOutput(:)));
[yPeak, xPeak] = ind2sub(size(correlationOutput),maxIndex(1))
% Because cross correlation increases the size of the image, 
% we need to shift back to find out where it would be in the original image.
corr_offset = [(xPeak-size(smallSubImage,2)) (yPeak-size(smallSubImage,1))]
% Plot it over the original image.
subplot(2, 2, 4); % Re-display image in lower right.
imshow(rgbImage);
axis on; % Show tick marks giving pixels
hold on; % Don't allow rectangle to blow away image.
% Calculate the rectangle for the template box.  Rect = [xLeft, yTop, widthInColumns, heightInRows]
boxRect = [corr_offset(1) corr_offset(2) templateWidth, templateHeight]
% Plot the box over the image.
rectangle('position', boxRect, 'edgecolor', 'g', 'linewidth',2);
% Give a caption above the image.
title('Template Image Found in Original Image', 'FontSize', fontSize);
uiwait(helpdlg('Done with demo!'));



%%
% 
% % Demo to use normxcorr2 to find a template (a white onion)
% % in a larger image (of a pile of vegetables)
% clc;    % Clear the command window.
% close all;  % Close all figures (except those of imtool.)
% imtool close all;  % Close all imtool figures.
% clear;  % Erase all existing variables.
% workspace;  % Make sure the workspace panel is showing.
% format long g;
% format compact;
% fontSize = 20;
% % Check that user has the Image Processing Toolbox installed.
% hasIPT = license('test', 'image_toolbox');
% if ~hasIPT
% 	% User does not have the toolbox installed.
% 	message = sprintf('Sorry, but you do not seem to have the Image Processing Toolbox.\nDo you want to try to continue anyway?');
% 	reply = questdlg(message, 'Toolbox missing', 'Yes', 'No', 'Yes');
% 	if strcmpi(reply, 'No')
% 		% User said No, so exit.
% 		return;
% 	end
% end
% % Read in a standard MATLAB color demo image.
% folder = fullfile(matlabroot, '\toolbox\images\imdemos');
% baseFileName = 'peppers.png';
% % Get the full filename, with path prepended.
% fullFileName = fullfile(folder, baseFileName);
% if ~exist(fullFileName, 'file')
% 	% Didn't find it there.  Check the search path for it.
% 	fullFileName = baseFileName; % No path this time.
% 	if ~exist(fullFileName, 'file')
% 		% Still didn't find it.  Alert user.
% 		errorMessage = sprintf('Error: %s does not exist.', fullFileName);
% 		uiwait(warndlg(errorMessage));
% 		return;
% 	end
% end
% rgbImage = imread(fullFileName);
% % Get the dimensions of the image.  numberOfColorBands should be = 3.
% [rows columns numberOfColorBands] = size(rgbImage);
% % Display the original color image.
% subplot(2, 2, 1);
% imshow(rgbImage, []);
% axis on;
% title('Original Color Image', 'FontSize', fontSize);
% % Enlarge figure to full screen.
% set(gcf, 'units','normalized','outerposition',[0, 0, 1, 1]);
% % Let's get our template by extracting a small portion of the original image.
% templateWidth = 71
% templateHeight = 49
% smallSubImage = imcrop(rgbImage, [192, 82, templateWidth, templateHeight]);
% subplot(2, 2, 2);
% imshow(smallSubImage, []);
% axis on;
% title('Template Image to Search For', 'FontSize', fontSize);
% % Ask user which channel to search for a match.
% % channelToCorrelate = menu('Correlate which color channel?', 'Red', 'Green', 'Blue');
% % It actually finds the same location no matter what channel you pick, 
% % for this image anyway, so let's just go with red (channel #1).
% channelToCorrelate = 1;  
% correlationOutput = normxcorr2(smallSubImage(:,:,1), rgbImage(:,:, channelToCorrelate));
% subplot(2, 2, 3);
% imshow(correlationOutput, []);
% axis on;
% title('Normalized Cross Correlation Output', 'FontSize', fontSize);
% % Find out where the normalized cross correlation image is brightest.
% [maxCorrValue, maxIndex] = max(abs(correlationOutput(:)));
% [yPeak, xPeak] = ind2sub(size(correlationOutput),maxIndex(1))
% % Because cross correlation increases the size of the image, 
% % we need to shift back to find out where it would be in the original image.
% corr_offset = [(xPeak-size(smallSubImage,2)) (yPeak-size(smallSubImage,1))]
% % Plot it over the original image.
% subplot(2, 2, 4); % Re-display image in lower right.
% imshow(rgbImage);
% axis on; % Show tick marks giving pixels
% hold on; % Don't allow rectangle to blow away image.
% % Calculate the rectangle for the template box.  Rect = [xLeft, yTop, widthInColumns, heightInRows]
% boxRect = [corr_offset(1) corr_offset(2) templateWidth, templateHeight]
% % Plot the box over the image.
% rectangle('position', boxRect, 'edgecolor', 'g', 'linewidth',2);
% % Give a caption above the image.
% title('Template Image Found in Original Image', 'FontSize', fontSize);
% uiwait(helpdlg('Done with demo!'));

