# -*- coding: utf-8 -*-
"""
Script de demonstração para salvar máscaras previstas
Este script mostra como salvar as imagens da IA como arquivos locais
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

print("=== DEMONSTRAÇÃO DE SALVAMENTO DE MÁSCARAS ===")

# Configurações
images_folder = 'dataset'
masks_folder = 'mascara'
output_folder = 'resultado_mascara'

# Criar pasta de resultados se não existir
os.makedirs(output_folder, exist_ok=True)
print(f"📁 Pasta de resultados: {output_folder}")

# Carregar apenas 2-3 imagens para treinamento rápido
print("1. Carregando imagens...")

def load_limited_data(limit=3):
    images = []
    masks = []
    
    image_files = sorted(os.listdir(images_folder))[:limit]  # Limitar para treino rápido
    
    for image_file in image_files:
        base_name, ext = os.path.splitext(image_file)
        mask_file = base_name + '_mask' + ext
        
        image_path = os.path.join(images_folder, image_file)
        mask_path = os.path.join(masks_folder, mask_file)
        
        if os.path.exists(image_path) and os.path.exists(mask_path):
            # Carregar e redimensionar para 128x128 (mais rápido)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            img = img.astype(np.float32) / 255.0
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (128, 128))
            mask = (mask > 0).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)
            
            images.append(img)
            masks.append(mask)
            print(f"   ✓ Carregado: {image_file}")
    
    return np.array(images), np.array(masks)

# Carregar dados limitados
images, masks = load_limited_data(3)
print(f"Total de imagens carregadas: {len(images)}")

# U-Net simplificada (menor e mais rápida)
def simple_unet(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    
    # Encoder (simplificado)
    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = Conv2D(128, 3, activation='relu', padding='same')(p2)
    
    # Decoder (simplificado)
    u4 = UpSampling2D((2, 2))(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(64, 3, activation='relu', padding='same')(u4)
    
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(32, 3, activation='relu', padding='same')(u5)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(c5)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

print("2. Criando modelo U-Net simplificado...")
model = simple_unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("3. Treinamento rápido (10 epochs para melhor aprendizado)...")
# Treinar rapidamente com poucos dados mas mais epochs
history = model.fit(images, masks, epochs=10, validation_split=0.3, verbose=1, batch_size=1)

print("4. Fazendo previsão e salvando resultados...")

# Usar a primeira imagem para teste
test_image = images[0]
test_image_input = np.expand_dims(test_image, axis=0)

# Fazer previsão
predicted_mask = model.predict(test_image_input)

# Debug: vamos verificar os valores da previsão
print(f"   Valores da máscara prevista - Min: {predicted_mask.min():.4f}, Max: {predicted_mask.max():.4f}, Média: {predicted_mask.mean():.4f}")

# Usar threshold mais baixo e tentar diferentes valores
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
best_threshold = 0.1  # Começar com threshold baixo

# Encontrar o melhor threshold (que produz mais pixels brancos)
best_pixel_count = 0
for thresh in thresholds:
    mask_test = (predicted_mask[0] > thresh).astype(np.uint8)
    white_pixels = np.sum(mask_test)
    print(f"   Threshold {thresh}: {white_pixels} pixels brancos")
    if white_pixels > best_pixel_count and white_pixels < (128*128*0.8):  # Não mais que 80% da imagem
        best_threshold = thresh
        best_pixel_count = white_pixels

print(f"   Usando threshold: {best_threshold}")
predicted_mask_binary = (predicted_mask[0] > best_threshold).astype(np.uint8)

# Salvar resultados
print("5. Salvando arquivos de imagem...")

# 1. Imagem original
original_bgr = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
original_path = os.path.join(output_folder, 'imagem_original_demo.png')
cv2.imwrite(original_path, (original_bgr * 255).astype(np.uint8))
print(f"   ✓ Salvo: {original_path}")

# 2. Máscara prevista
mask_to_save = (predicted_mask_binary.squeeze() * 255).astype(np.uint8)
mask_path = os.path.join(output_folder, 'mascara_prevista_demo.png')
cv2.imwrite(mask_path, mask_to_save)
print(f"   ✓ Salvo: {mask_path}")

# 3. Máscara real (para comparação)
real_mask = (masks[0].squeeze() * 255).astype(np.uint8)
real_mask_path = os.path.join(output_folder, 'mascara_real_demo.png')
cv2.imwrite(real_mask_path, real_mask)
print(f"   ✓ Salvo: {real_mask_path}")

# 4. Comparação lado a lado
# Redimensionar máscaras para 3 canais
mask_3ch = cv2.cvtColor(mask_to_save, cv2.COLOR_GRAY2BGR)
real_mask_3ch = cv2.cvtColor(real_mask, cv2.COLOR_GRAY2BGR)
original_display = (original_bgr * 255).astype(np.uint8)

# Concatenar: Original | Real | Prevista
comparison = np.hstack((original_display, real_mask_3ch, mask_3ch))
comparison_path = os.path.join(output_folder, 'comparacao_completa_demo.png')
cv2.imwrite(comparison_path, comparison)
print(f"   ✓ Salvo: {comparison_path}")

# 5. Plot com matplotlib (salvado como arquivo)
try:
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(test_image)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(masks[0].squeeze(), cmap='gray')
    plt.title('Máscara Real')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask_binary.squeeze(), cmap='gray')
    plt.title('Máscara Prevista')
    plt.axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'plot_matplotlib_demo.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Salvo: {plot_path}")
    
    # Tentar mostrar (pode não funcionar no VS Code)
    plt.show()
    
except Exception as e:
    print(f"   ⚠ Matplotlib display não disponível: {e}")

print("\n=== RESUMO DOS ARQUIVOS SALVOS ===")
saved_files = [
    'imagem_original_demo.png',
    'mascara_prevista_demo.png', 
    'mascara_real_demo.png',
    'comparacao_completa_demo.png',
    'plot_matplotlib_demo.png'
]

for file in saved_files:
    file_path = os.path.join(output_folder, file)
    if os.path.exists(file_path):
        print(f"✓ {file_path} - {os.path.getsize(file_path)} bytes")
    else:
        print(f"✗ {file_path} - não encontrado")

print(f"\nTodos os arquivos foram salvos na pasta: {output_folder}/")
print("Você pode abrir estes arquivos PNG no VS Code ou qualquer visualizador de imagens!")
