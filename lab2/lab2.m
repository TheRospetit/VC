%% TODO LEER LAS IMÁGENES DE LA CARPETA
clearvars,
close all,
clc,

files = dir('./imatges/petites/*.jpg');

% TODO. CARGAMOS UNA PRIMERA IMAGEN PARA HACER TODAS LAS PRUEBAS
for i = 1:length(files)
    path = strcat('./imatges/petites/', files(i).name);
    image = imread(path);
    % image = imread('peppers.png');

    % TODO. SEPARO LA IMAGEN EN TRES COMPONENTES
    [height, width, ~] = size(image);
    tercera_parte = floor(height / 3);
    R = image(1:tercera_parte, :, :);
    G = image(tercera_parte+1:tercera_parte*2, :, :);
    B = image(tercera_parte*2+1:end, :, :);

    % TODO. REESTRUCTURO LAS IMÁGENES PARA QUE TODAS TENGAN EL MISMO TAMAÑO PARA CUANDO LAS TENGA QUE FUSIONAR
    [height1, width1, ~] = size(R);
    [height2, width2, ~] = size(G);
    [height3, width3, ~] = size(B);

    height_img = min([height1, height2, height3]);
    R = R(end-height_img+1:end, :, :);
    G = G(end-height_img+1:end, :, :);
    B = B(end-height_img+1:end, :, :);

    % TODO. RECORTO UN PORCENTAJE LOS BORDES DE LA IMÁGEN PARA EVITAR PROBLEMAS
    rec_height = floor(height_img * 0.1);
    rec_width = floor(width1 * 0.1);
    R = R(rec_height+1:end-rec_height, rec_width+1:end-rec_width, :);
    G = G(rec_height+1:end-rec_height, rec_width+1:end-rec_width, :);
    B = B(rec_height+1:end-rec_height, rec_width+1:end-rec_width, :);
    [height_fin, width_fin, ~] = size(R);

    figure('Name','Inicial')
    rgb_img = cat(3, R, G, B);
    imshow(rgb_img);

    corr_rg = correlacion_cruzada(R, G);
    corr_rb = correlacion_cruzada(R, B);

    shift_g = [corr_rg(1) - floor(size(R,1)/2), corr_rg(2) - floor(size(R,2)/2)];
    shift_b = [corr_rb(1) - floor(size(R,1)/2), corr_rb(2) - floor(size(R,2)/2)];

    
    g_shifted = imtranslate(G, [-shift_g(1), shift_g(2)]);
    %M = single([1, 0, -shift_b(0); 0, 1, shift_b(1); 0, 0, 1]);
    b_shifted = imtranslate(B, [-shift_b(1), shift_b(2)]);

    figure('Name','Correlació');
    rgb_new = cat(3, R, g_shifted, b_shifted);
    imshow(rgb_new);
end