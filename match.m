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

%%


