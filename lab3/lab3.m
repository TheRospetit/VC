%% MANUAL
clearvars,
close all,
clc,

% Cargo las imágenes.
imageDir = fullfile('imatges', 'test*');
imageScene = imageDatastore(imageDir);

% Muestra un montaje de las imágenes.
montage(imageScene.Files)
%%
I = readimage(imageScene,1);

grayImage1 = im2gray(I);
imshow(grayImage1)
puntos = ginput(4); % Marco los poi
%puntos = [608.75,	391.25; 751.25,	293.75; 1231.25,	317.75; 1231.25, 497.75];
%%
numeroImagenes = numel(imageScene.Files);
tforms(numeroImagenes) = projtform2d;

imageSize = zeros(numeroImagenes,2);
%%
for n = 2:numeroImagenes
    % Guardo los poi de I(n-1).
    puntosPrevios = puntos;
        
    % Leo la imagen que toca actualmente y la transformo en grises
    I = readimage(imageScene, n);
    grayImage2 = im2gray(I);    
    
    % Guardo el tamaño de la imagen
    imageSize(n,:) = size(grayImage2);
    
    % Detecto las features mediante SURF de la imagen actual
    imshow(grayImage2)
    puntos = ginput(4);    
    %puntos = [146.75,	386.75; 301.25,	292.25;764.75,	328.25;766.25,	499.25];

    % Busco las correspondéncias entre los puntos I(n) e I(n-1).
    A = [];
    for i = 1:size(puntos,1)
        xi = puntos(i,1);
        yi = puntos(i,2);
        ui = puntosPrevios(i,1);
        vi = puntosPrevios(i,2);
        A = [A; xi yi 1 0 0 0 -ui*xi -ui*yi -ui; 0 0 0 xi yi 1 -vi*xi -vi*yi -vi];
    end
    [U,S,V] = svd(A);
    h = V(:,end);
    H = reshape(h, [3, 3])';
    H = double(H / H(3,3));

    % Estimo la posible transformada que pueda necesitar I(n) e I(n-1).
    tforms(n) = projtform2d(H);
    
    % Calculo T(1) * T(2) * ... * T(n-1) * T(n).
    tforms(n).A = tforms(n-1).A * tforms(n).A; 
end
%% CALCULO LOS LÍMITES DE CADA TRANSFORMACIÓN
for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);    
end

%% BUSCO QUE IMÁGEN SE ENCUENTRA EN EL CENTRO
avgXLim = mean(xlim, 2);
[~,idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);

%% APLICO LA TRANSFORMADA INVERSA DE LA IMAGEN A TODAS LAS OTRAS
Tinv = invert(tforms(centerImageIdx));
for i = 1:numel(tforms)    
    tforms(i).A = Tinv.A * tforms(i).A;
end

%% INICIALIZO LA PANORÁMICA DE LAS IMÁGENES QUE SE HAN COGIDO

maxImageSize = max(imageSize);

% Cojo los límites previamente calculados 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Calculo el ancho y alto de la imagen resultante panorámica
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Inicializo la panorámica en negro
panorama = zeros([height width 3], 'like', I);
panorama2 = zeros([height width 3], 'like', I);

%% CREO LA PANORÁMICA usando un blender
blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');  

% Creo un objeto 2D para definir el tamaño de la panorámica
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Creo la panorámica
for i = 1:numeroImagenes
    
    I = readimage(imageScene, i);   
   
    % Transformo la imagen inicial en la panorámica
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
                  
    % Creo una máscara binaria    
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
    
    % Superpongo la imagen nueva con la panoramica
    panorama2 = my_blend_with_mask(panorama, warpedImage, mask);
    panorama = step(blender, panorama, warpedImage, mask);
end

figure;
imshow(panorama)
title('Panoramica CON Aalphablend');
figure;
imshow(panorama2)
title('Panoramica SIN Aalphablend');
imwrite(panorama, 'resultado_test.png', 'png') % GUARDO LA IMAGEN FINAL

%% NOTAS
% Los puntos siempre los he escogido en el orden de arriba izquierda,
% arriba derecha, abajo derecha y abajo izquierda además quedaban formando
% un cuadrilátero irregular



%% VERISÓN FINAL DEL CÓDIGO USANDO SURF Y BLEND (AUTOMÁTICO)
clearvars,
close all,
clc,
% Cargo las imágenes.
imageDir = fullfile('imatges', 'test*');
imageScene = imageDatastore(imageDir);

% Display images to be stitched.
montage(imageScene.Files)

%% LEO LA PRIMERA IMÁGEN DEL SET
I = readimage(imageScene,1);

%Inicializo las features de la imagen 1
grayImage1 = im2gray(I);
puntos = detectSURFFeatures(grayImage1); % Uso el detector SURF
[features, puntos] = extractFeatures(grayImage1,puntos);

%% Inicializo las transformaciones
numeroImagenes = numel(imageScene.Files);
tforms(numeroImagenes) = projtform2d;

imageSize = zeros(numeroImagenes,2);

%% Itero por las otras parejas de imágenes que quedan
for n = 2:numeroImagenes
    % Store points and features for I(n-1).
    puntosPrevios = puntos;
    featuresPrevious = features;
        
    % Leo la imagen que toca actualmente y la transformo en grises
    I = readimage(imageScene, n);
    grayImage2 = im2gray(I);    
    
    % Guardo el tamaño de la imagen
    imageSize(n,:) = size(grayImage2);
    
    % Detecto las features mediante SURF de la imagen actual
    puntos = detectSURFFeatures(grayImage2);    
    [features, puntos] = extractFeatures(grayImage2, puntos);
  
    % Busco las correspondéncias entre los puntos I(n) e I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);
    matchedPoints = puntos(indexPairs(:,1), :);
    matchedPointsPrev = puntosPrevios(indexPairs(:,2), :); 

    % Muestro los puntos que se han correlacionados
    figure;
    showMatchedFeatures(grayImage1, grayImage2, matchedPoints, matchedPointsPrev);
    
    % Estimo la posible transformada que pueda necesitar I(n) e I(n-1).
    tforms(n) = estgeotform2d(matchedPoints, matchedPointsPrev,'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    
    % Calculo T(1) * T(2) * ... * T(n-1) * T(n).
    tforms(n).A = tforms(n-1).A * tforms(n).A; 
end
%% CALCULO LOS LÍMITES DE CADA TRANSFORMACIÓN
for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);    
end

%% BUSCO QUE IMÁGEN SE ENCUENTRA EN EL CENTRO
avgXLim = mean(xlim, 2);
[~,idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);

%% APLICO LA TRANSFORMADA INVERSA DE LA IMAGEN A TODAS LAS OTRAS
Tinv = invert(tforms(centerImageIdx));
for i = 1:numel(tforms)    
    tforms(i).A = Tinv.A * tforms(i).A;
end

%% INICIALIZO LA PANORÁMICA DE LAS IMÁGENES QUE SE HAN COGIDO
for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Cojo los límites previamente calculados 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Calculo el ancho y alto de la imagen resultante panorámica
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Inicializo la panorámica en negro
panorama = zeros([height width 3], 'like', I);

%% CREO LA PANORÁMICA usando un blender
blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');  

% Creo un objeto 2D para definir el tamaño de la panorámica
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Creo la panorámica
for i = 1:numeroImagenes
    
    I = readimage(imageScene, i);   
   
    % Transformo la imagen inicial en la panorámica
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
                  
    % Creo una máscara binaria    
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
    
    % Superpongo la imagen nueva con la panoramica
    panorama = step(blender, panorama, warpedImage, mask);
end

figure;
imshow(panorama);
%% FUNCIONES
function blended_image = my_blend_with_mask(image1, image2, mask)

    % Create a color mask from the binary mask
    color_mask = cat(3, mask, mask, mask);

    % Blend the images using the color mask
    blended_image = uint8(double(image1).*(1-color_mask) + double(image2).*color_mask);
end