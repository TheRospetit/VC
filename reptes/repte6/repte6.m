%% CARGAMOS LAS IMAGENES A MANO
clear all, close all
image1 = imread('repte6_1.jpg');
image2 = imread('repte6_2.jpg');

I1 = rgb2gray(image1);
I2 = rgb2gray(image2);

%% MUESTRO LAS IMAGENES QUE HAN SIDO PASADAS A GRIS
figure;
imshow(I1)
figure;
imshow(I2)

%% PRIMERA VERSIÓN DEL CÓDIGO
corners1 = detectHarrisFeatures(I1);
corners2 = detectHarrisFeatures(I2);
[features1, valid_corners1] = extractFeatures(I1, corners1);
[features2, valid_corners2] = extractFeatures(I2, corners2);

figure;
imshow(I1);
hold on;
plot(corners1.selectStrongest(50));

figure;
imshow(I2);
hold on;
plot(corners2.selectStrongest(50));

%% SEGUNDA VERSIÓN DEL CÓDIGO

clear all, close all
% Leer la imagen
path='./';
li=dir([path '*.jpg']);

ima={};
descs={};

for i=1:length(li)
    image=imread([path li(i).name]);
    im=rgb2gray(image);
    ima{i} = im;
    
    % Calcular los gradientes en x e y de la imagen
    [dIx, dIy] = imgradientxy(im);
    
    % Calcular el producto de los gradientes
    A = dIx.^2;
    B = dIy.^2;
    C = dIx.*dIy;
    
    % Aplicar un filtro gaussiano a los productos de gradientes
    sigma = 2;
    Ag = imgaussfilt(A, sigma);
    Bg = imgaussfilt(B, sigma);
    Cg = imgaussfilt(C, sigma);
    
    k = 0.05;
    %Calcular la matriz A de Harris en cada pixel
    H = (Ag .* Cg) - (Cg .* Bg);
    traceA = Ag + Bg;
    R = H-k*(traceA .^2);
    desc = imregionalmax((R >  9.966474854294462e+06).* R); 


    [row,col]=find(desc);
    val=R(sub2ind(size(R),row,col));
    
    N=100; % N=10000;
    N=min(N,length(val));
    [valsort,ind]=sort(val,1,'descend');
    figure;
    imshow(im);
    hold on;
    plot(col(ind(1:N)),row(ind(1:N)),'ro','MarkerSize',5);
    drawnow
    pt{i}=[row(ind(1:N)), col(ind(1:N))];

end
%% DESCRIPTORS
subim_size = 5;
for j = 1:length(pt)
    for i = 1:length(pt{j})
        x = pt{j}(i, 1);
        y = pt{j}(i, 2);
        % obtener la submatriz alrededor del punto de interés
        if x-subim_size > 0
            lft = x-subim_size;
        else
            lft = 1;
        end
        if y-subim_size > 0
            up = y-subim_size;
        else
            up = 1;
        end

    % Subimatge al voltant del punt
    subim = imcrop(image, [lft/2 up/2 subim_size-1 subim_size-1]);
    
    
    % almacenar el descriptor
    descriptors(i,:) = reshape(subim, 1, []);
    descs{j} = descriptors;
    end
end
%% MATCHING
match = zeros(1,N);
for j=1:N
    d1=descs{1}(j,:);
    [~,match(j)]=min(sum(abs(descs{2}-d1)));
end

figure,
imshow([ima{1},ima{2}]);
hold on
despl = size(ima{1},2);
plot(pt{1}(:,2),pt{1}(:,1),'ro','MarkerSize',5);
plot(pt{2}(:,2)+despl,pt{2}(:,1),'yo','MarkerSize',5);
for j=1:500
    column = [pt{1}(j,2),pt{2}(match(j),2)+despl];
    fila = [pt{1}(j,1),pt{2}(match(j),1)];
    line(column,fila);
end
hold off
%% SEGONA MANERA DE CORRELACIONAR (NO FEM CAS AL MATCH)

figure,
imshow([ima{1},ima{2}]);
hold on
despl = size(ima{1},2);
plot(pt{1}(:,2),pt{1}(:,1),'ro','MarkerSize',5);
plot(pt{2}(:,2)+despl,pt{2}(:,1),'yo','MarkerSize',5);
for j=1:size(pt{1},1)
    column = [pt{1}(j,2),pt{2}(j,2)+despl];
    fila = [pt{1}(j,1),pt{2}(j,1)];
    line(column,fila);
end
hold off

%% NOTACIONS: AQUESTA SEGONA MANERA DE CORRELACIONAR ELS PUNTS NOMÉS
%% BUSCA CORRELACIONAR TOTS ELS PUNTS, HO FA PER ORDRE AIXI QUE NO TÉ
%% CAP MENA DE COHERÉNCIA PERÒ NO TINDREM UN CAS EN EL QUE ES QUEDI "LLIURE"