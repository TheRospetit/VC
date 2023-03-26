function max_pos = correlacion_cruzada(matriz1, matriz2)
    % Calcular la correlación cruzada
    corr = conv2(matriz1, fliplr(flipud(matriz2)), 'same'); % Tienen que ser al revés en G se usa R como máscara
    
    % Encontrar todos los valores máximos
    [max_vals_row, max_vals_col] = find(corr == max(corr(:)));
    max_vals = [max_vals_row, max_vals_col];
    
    % Encontrar la posición más cercana al centro de la matriz
    centro = floor(size(matriz2) / 2); % 80 95
    distancias = sqrt((max_vals(:,1) - centro(1)).^2 + (max_vals(:,2) - centro(2)).^2);
    [~, idx_max] = min(distancias);
    max_pos = max_vals(idx_max, :);
end