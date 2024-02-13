%% IMÁGEN RGB
clearvars,
close all,
clc,
img = imread('mazingerZ.png');

img_flat = double(reshape(img, [], 3));

%% IMAGEN EN HSV
clearvars,
close all,
clc,
% CARGO LA IMÁGEN RGB
img = imread('mazingerZ.png');

img = rgb2hsv(img);
% Convertimos la imagen a doble precisión
img = im2double(img);

img_flat = double(reshape(img, [], 3));

% Convertimos el canal de matiz (Hue) a representación circular
img_flat(:,1) = img_flat(:,1) / 360 * 2 * pi;

%% ALGORITMO KMEANS

% NUMERO DE CLUSTERS
K = 4;

% INICIO LOS CENTROIDES DE MANERA ALEATORIA
centroids = rand(K, 3) * 255;

% VARIABLES
max_iters = 10; % NUMERO DE ITERACIONES
idx = zeros(size(img_flat, 1), 1);
distances = zeros(size(img_flat, 1), K);

% ALGORITMO KMEANS
for iter = 1:max_iters
    
    % Asigna cada punto de datos al centroide más cercano
    for i = 1:size(img_flat, 1)
        for j = 1:K
            distances(i, j) = norm(img_flat(i,:) - centroids(j,:));
        end
        [~, idx(i)] = min(distances(i,:));
    end
    
    % Actualizo los centros
    for j = 1:K
        centroids(j,:) = mean(img_flat(idx == j,:), 1);
    end
    
end

%% OUTPUT RGB (si la imagen era RGB)

% Reshape la matriz de idx para que tenga el tamaño original de la imágen.
idx = reshape(idx, size(img,1), size(img,2));

% CREO UNA NUEVA IMÁGEN DONDE COLOREO CADA PIXEL CON EL COLOR DE SU CENTRO
img_new = zeros(size(img));
for i = 1:K
    mask = idx == i;
    img_new(:,:,1) = img_new(:,:,1) + mask .* centroids(i,1);
    img_new(:,:,2) = img_new(:,:,2) + mask .* centroids(i,2);
    img_new(:,:,3) = img_new(:,:,3) + mask .* centroids(i,3);
end

% Muestro la imágen final
figure;
imshow(uint8(img_new));
%imwrite(uint8(img_new), 'outputRGB_4.png', 'png')


%% OUTPUT HSV (si la imagen era HSV)
% Convertimos el canal de matiz a grados
img_flat(:,1) = img_flat(:,1) / (2 * pi) * 360;

% Reshape la matriz de idx para que tenga el tamaño original de la imágen.
idx = reshape(idx, size(img,1), size(img,2));

% CREO UNA NUEVA IMÁGEN DONDE COLOREO CADA PIXEL CON EL COLOR DE SU CENTRO
img_new = zeros(size(img));
for i = 1:K
    mask = idx == i;
    img_new(:,:,1) = img_new(:,:,1) + mask .* centroids(i,1);
    img_new(:,:,2) = img_new(:,:,2) + mask .* centroids(i,2);
    img_new(:,:,3) = img_new(:,:,3) + mask .* centroids(i,3);
end

% Convertimos el canal de matiz de la nueva imagen de nuevo a representación en grados
img_new(:,:,1) = img_new(:,:,1) / (2 * pi) * 360;

% Convertimos la nueva imagen de nuevo a HSV
img_new = hsv2rgb(img_new);

% Mostramos la imágen resultante
figure;
imshow(img_new);
%imwrite(img_new, 'outputHSV_4.png', 'png')
%% README NOTAS
% MUCHAS veces puede suceder que la imágen resultante salga en negro pero
% hay veces en las que si que lo hace bien todo y muestra la imágen
% resultante
