clearvars,
close all,
clc,

ima=imread('./imatges/ima1_smol.jpg');
% Definir el descriptor per a cada píxel
color = double(ima);
texture = stdfilt(ima);
%% MARCAMOS LOS PUNTOS QUE SERÁN ARBOLES
imshow(ima)
arbres = ginput(100);

% Calcular el descriptor per a cada punt seleccionat
arbres_color = zeros(100, 1);
arbres_texture = zeros(100, 1);
for i = 1:100
    x = round(arbres(i, 1));
    y = round(arbres(i, 2));
    arbres_color(i) = color(y, x);
    arbres_texture(i) = texture(y, x);
end


%% MARCAMOS LOS PUNTOS QUE NO SERÁN ARBOLES
imshow(ima)
no_arbre = ginput(100);

% Calcular el descriptor per a cada punt seleccionat
no_arbre_color = zeros(100, 1);
no_arbre_texture = zeros(100, 1);
for i = 1:100
    x = round(no_arbre(i, 1));
    y = round(no_arbre(i, 2));
    no_arbre_color(i) = color(y, x);
    no_arbre_texture(i) = texture(y, x);
end

%% MUESTRO LOS PUNTOS ESCOGIDOS
imshow(ima);
impixelinfo;
hold on;
plot(arbres(:,1), arbres(:,2), 'go');
plot(no_arbre(:,1), no_arbre(:,2), 'ro');
hold off;

%% SEPARAMOS LOS PUNTOS EN TRAIN Y TEST
num_puntos = size(arbres, 1);

porcentaje_entrenamiento = 0.5;
porcentaje_prueba = 1 - porcentaje_entrenamiento;

indices = crossvalind('HoldOut', num_puntos, porcentaje_prueba);

arbres_entreno = arbres(indices, :);
arbres_color_entreno = arbres_color(indices, :);
arbres_texture_entreno = arbres_texture(indices, :);

arbres_test = arbres(~indices, :);
arbres_color_test = arbres_color(~indices, :);
arbres_texture_test = arbres_texture(~indices, :);

no_arbres_entreno = no_arbre(indices, :);
no_arbres_color_entreno = no_arbre_color(indices, :);
no_arbres_texture_entreno = no_arbre_texture(indices, :);

no_arbres_test = no_arbre(~indices, :);
no_arbres_color_test = no_arbre_color(~indices, :);
no_arbres_texture_test = no_arbre_texture(~indices, :);

%% Creacion parámetros
% Crear matriu de características X
color_train_entreno = [arbres_color_entreno; no_arbres_color_entreno];
texture_train_entreno = [arbres_texture_entreno; no_arbres_texture_entreno];
X_train = [color_train_entreno, texture_train_entreno];


color_train_test = [arbres_color_test; no_arbres_color_test];
texture_train_test = [arbres_texture_test; no_arbres_texture_test];
X_test = [color_train_test, texture_train_test];

% Crear vector d'etiquetes Y
y_train = [ones(size(arbres_entreno,1),1); -1*ones(size(no_arbres_entreno,1),1)];

y_test = [ones(size(arbres_test,1),1); -1*ones(size(no_arbres_test,1),1)];

%% SVM
% Entrenar el classificador SVM
svm_model = fitcsvm(X_train, y_train);

% Classificar les dades de prova amb el model SVM
predicted_labels_svm = predict(svm_model, X_test);

% Calcular l'accuracy del model SVM
accuracy_svm = sum(predicted_labels_svm == y_test) / length(y_test);
fprintf('Accuracy del model SVM: %.2f\n', accuracy_svm);

%% Naive bayes
% Entrenamiento del clasificador LDA
nb_model = fitcnb(X_train, y_train);

y_pred = predict(nb_model, X_test);

accuracy_nb = sum(y_pred == y_test) / length(y_test);
fprintf('Accuracy del model Naïve Bayes: %.2f\n', accuracy_nb);

%% LDA (LINEAR DISCRIMINANT ANALYSIS)
% Entrenamiento del clasificador LDA
lda_model = fitcdiscr(X_train, y_train);

% Predicción con el conjunto de prueba
y_pred_lda = predict(lda_model, X_test);

% Cálculo de la precisión del clasificador LDA
accuracy_lda = sum(y_pred_lda == y_test) / length(y_test);
fprintf('Accuracy del model LDA: %.2f\n', accuracy_lda);

%% Mostrar imagen binarizada con el modelo con mejor accuracy
% Crear una matriz con los descriptores de cada píxel de la imagen
ima_desc = zeros(size(ima,1)*size(ima,2),2);
for i=1:size(ima,1)
    for j=1:size(ima,2)
        ima_desc((i-1)*size(ima,2)+j,:) = [double(ima(i,j)) stdfilt(ima(i,j))];
    end
end

% Clasificar los píxeles de la imagen con el clasificador LDA
ima_pred = predict(nb_model, ima_desc);

% Crear una imagen binarizada para visualizar la diferencia entre clases
ima_bin = zeros(size(ima));
for i=1:size(ima,1)
    for j=1:size(ima,2)
        if ima_pred((i-1)*size(ima,2)+j) == 1
            ima_bin(i,j) = 1; % Clase de árboles
        else
            ima_bin(i,j) = 0; % Otras clases
        end
    end
end

% Visualizar la imagen binarizada
imshow(ima_bin);
%% IMAGEN RESULTADO
ima_bin(:,:,2) = ima_bin(:,:,1);
ima_bin(:,:,3) = ima_bin(:,:,1);
imshow(ima_bin);
%%
im_bin = ima_bin(:,:,1)/255;
% Mostrar solo los puntos clasificados como "árboles"
ima_arboles = imoverlay(ima, im_bin == 1, [0 0 0]);

% Mostrar solo los puntos clasificados como "no árboles"
ima_no_arboles = imoverlay(ima, im_bin == 0, [0 0 0]);

% Mostrar ambas imágenes en una misma figura
imshowpair(ima_arboles, ima_no_arboles, 'montage');