%% Loading things
tic
%% Load in the Champion Data
if ~exist('champs','var')
    imdir = './Champions/';
    imf = dir(imdir);
    imf = imf(3:end);
    champs = cell([numel(imf) 1]);
    for i = 1:numel(imf)
        champs{i}.name = strrep(imf(i).name,'.png','');
        champs{i}.image = imread(strcat(imdir,imf(i).name));
        champs{i}.r_ch = champs{i}.image(:,:,1);
        champs{i}.g_ch = champs{i}.image(:,:,2);
        champs{i}.b_ch = champs{i}.image(:,:,3);

        champs{i}.r_hist = imhist(champs{i}.r_ch);
        champs{i}.g_hist = imhist(champs{i}.g_ch);
        champs{i}.b_hist = imhist(champs{i}.b_ch);
    end
end

%% Load in Picture in question 

% base = imread('./img/thresh1.png'); qq = 93;
% base = imread('./img/vi.png'); qq = 104;
% base = imread('./img/nid_dead.png'); qq = 64;
base = imread('./img/nid.png'); qq = 64;

BL = base(end/4:end,1:end/4,:);


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
% [x y w h] rectangle

sq_size = (bb(:,3).*bb(:,4));
sq_sqness = abs(1-max(bb(:,3:4),[],2)./min(bb(:,3:4),[],2));
loc_bias = bb(:,1)./size(Ifinal,2)+(size(Ifinal,1)-bb(:,2))./size(Ifinal,1);
sq = (sq_size.*(sq_sqness<0.2))./loc_bias;

[~,ind] = max(sq);
charBox = round(bb(ind,:));
rectangle('position',bb(ind,:),'edgecolor','g','linewidth',2);

charImg = BL(charBox(2):charBox(2)+charBox(4),charBox(1):charBox(1)+charBox(3),:);

figure
imshow(charImg)

%% FIND THE CHAMP

% Resize to prepare to MSE
sz = size(champs{1}.image);
mm = min(size(charImg(:,:,1)));
sq_charImg = charImg(1:mm,1:mm,:);

scale = sz(1)/mm;
res_charImg = im2double(imresize(sq_charImg,scale));
dist = zeros([1 numel(imf)]);
for i = 1:numel(imf)
    dist(i) =  sum(sum(sum(abs(res_charImg - im2double(champs{i}.image)))));
end
[~, winner] = min(dist);

figure
imshow(champs{winner}.image)

subplot(3,2,1)
imshow(abs(im2double(res_charImg) - im2double(champs{winner}.image)));

subplot(3,2,2)
imshow(abs(im2double(res_charImg) - im2double(champs{qq}.image)));

subplot(3,2,3)
imshow(champs{winner}.image);

subplot(3,2,4)
imshow(champs{qq}.image);

subplot(3,2,5)
imshow(res_charImg);

subplot(3,2,6)
imshow(res_charImg);

toc