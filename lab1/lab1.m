%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                              VC                                %%%
%%%                             Lab 1                              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 
clearvars,
close all,
clc,


%% PROBLEM 1 (+0.5) --------------------------------------------------
% TODO. LEER LAS IMAGENES DE LA CARPETA INPUT DE HIGHWAY 
files = dir('.\highway\input\*.jpg');
% im_color = imread(fullfile(files(1).folder, files(1).name));
% im_grey = rgb2gray(im_color);
% imshow(im_grey)
files_train = files(1051:1200); % TRAIN
files_test = files(1201:1350); % TEST

train = [];
for i = 1:length(files_train)
    path = strcat('./highway/input/', files_train(i).name);
    image = imread(path);
    image = rgb2gray(image); % LEO CADA IMAGEN EN GRIS
    train = cat(3, train, image); % GUARDO LOS VALORES DE CADA IMAGEN EN UN ARRAY
end

test = [];
for i = 1:length(files_test)
    path = strcat('./highway/input/', files_test(i).name);
    image = imread(path);
    image = rgb2gray(image); % LEO CADA IMAGEN EN GRIS
    test = cat(3, test, image);
end
%% PROBLEM 2 (+0.5) --------------------------------------------------
% TODO. CALCULAR LA MEDIA Y LA DESVIACIÓN ESTÁNDAR
mean_train = mean(train, 3);


% REESCALO LA IMAGEN EN UN RANGO DE 0 A 255
mean_train = rescale(mean_train, 0, 255);
% TRANSFORMO EL TIPO DE DOUBLE A UINT8
mean_train = uint8(mean_train); 
% MUESTRO LA IMAGEN
figure; 
imshow(mean_train, []);



%% DESVIACIÓN ESTÁNDAR

std_train = std(double(train), 0, 3);
% REESCALO LA IMAGEN EN UN RANGO DE 0 A 255
std_train = rescale(std_train, 0, 255);
% TRANSFORMO EL TIPO DE DOUBLE A UINT8
std_train = uint8(std_train); 
% MUESTRO LA IMAGEN
figure; 
imshow(std_train, []);


%% PROBELM 3 (+1.0) --------------------------------------------------
% TODO. FRAGMENTAR COCHES RESTANDO EL MODELO DEL FONDO

threshold = 40;
train_no_bg = [];
for n=1:length(train(1,1,:))
    image = double(train(:,:,n));
    image = abs(image - double(mean_train));
    image = image > threshold;
    imshow(image)
    train_no_bg = cat(3, train_no_bg, image);
end

%% PROBLEM 4 (+1.0) --------------------------------------------------

% TODO. FRAGMENTAR COCHES CON UN MODELO MÁS PREPARADO

threshold = 40;
alpha =0.5;
beta = 20;
kernel = strel('disk', 3);
kernel_close = strel('disk', 10);
train_no_bg_2 = [];
for n=1:length(train(1,1,:))
    image = double(train(:,:,n));
    image = abs(image - double(mean_train));
    image = imopen(image, kernel);
    image = imclose(image, kernel_close);
    image = image > threshold*alpha + beta;
    imshow(image)
    train_no_bg_2 = cat(3, train_no_bg_2, image);
end














