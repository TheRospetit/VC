%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% VC i PSIV                                                      %%%
%%% Lab 0 (basat en les pràctiques de Gemma Rotger)                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 
% Hello! Welcome to the computer vision LAB. This is a section, and 
% you can execute it using the run section button on the top panel. If 
% you prefer, you can run all the code using the run button. Run this 
% section when you need to clear your data, figures and console 
% messages.
clearvars,
close all,
clc,

% With addpath you are adding the image path to your main path
% addpath('img')


%% PROBLEM 1 (+0.5) --------------------------------------------------
% TODO. READ THE CAMERAMAN IMAGE. 
img_cameraman = imread('.\img\cameraman.jpg');


%% PROBLEM 2 (+0.5) --------------------------------------------------
% TODO: SHOW THE CAMERAMAN IMAGE
imshow(img_cameraman)


%% PROBELM 3 (+2.0) --------------------------------------------------
% TODO. Negative efect using a double for instruction

tic,
% Your code goes here
im_neg = img_cameraman;
[height, width, chanels] = size(im_neg);
figure(1)
for i = 1:height
    for j = 1:width
        im_neg(i,j) = 255 - im_neg(i,j);
    end
end
imshow(im_neg);
toc

% TODO. Negative efect using a vectorial instruction

tic,
im_neg = 255 - img_cameraman;
figure(2)
imshow(im_neg);
toc,

tic,
im_neg = - img_cameraman; % si hacemos esto tenemos que toda la imagen sale en negro
figure(3)
imshow(im_neg);
toc,

% You sould see that results in figures 1 and 2 are the same but times
% are much different.

%% PROBLEM 4 (+2.0) --------------------------------------------------

% TODO. Give some color (red, green or blue)
% r = ...
% g = ...
% b = ...

% im_col = zeros...
% im_col(:,:,1)=...
% ...
% figure(1)
% imshow(im_col);

% im_col = cat...
% figure(2)
% imshow(im_col);


%% PROBLEM 5 (+1.0) --------------------------------------------------

% imwrite ...
% imwrite ...
% imwrite ...

%% PROBLEM 6 (+1.0) --------------------------------------------------

% lin128=im...
% figure(1)
% plot...

% lin128rgb=im...
% figure(2)
% plot...


%% PROBLEM 7 (+2) ----------------------------------------------------

tic;
% imhist ...
toc;


% TODO. Compute the histogram.
tic;
% h=zeros(1,256);
% for ...
% plot(h)
toc;

%% PROBLEM 8 Binarize the image text.png (+1) ------------------------

% TODO. Read the image
% imtext = ...
figure(1)
imshow(imtext)
imhist(imtext)

% TODO. Define 3 different thresholds
% th1 = ...
% th2 = ...
% th3 = ...

% TODO. Apply the 3 thresholds 5 to the image
% threshimtext1 = ...
% threshimtext2 = ...
% threshimtext3 = ...

% TODO. Show the original image and the segmentations in a subplot
figure(1)
subplot(2,1,1);
imshow(imtext);
subplot(2,3,4);
imshow(threshimtext1);
subplot(2,3,5);
imshow(threshimtext2);
subplot(2,3,6);
imshow(threshimtext3);
title('Original image');


%% THE END -----------------------------------------------------------
% Well done, you finished this lab! Now, remember to deliver it 
% properly on Caronte.

% File name:
% lab0_NIU.zip 
% (put matlab file lab0.m and python file lab0.py in the same zip file)
% Example lab0_1234567.zip

















