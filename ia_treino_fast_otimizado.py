# -*- coding: utf-8 -*-
"""
IA Treino FAST - Versão Otimizada
Baseado na versão que está funcionando melhor
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model

print("🚀 IA TREINO FAST - VERSÃO OTIMIZADA")
print("=" * 50)

# Configurações
images_folder = 'dataset'
masks_folder = 'mascara'
output_folder = 'resultado_fast_otimizado'

# Criar pasta de resultados se não existir
os.makedirs(output_folder, exist_ok=True)
print(f"📁 Pasta de resultados: {output_folder}")

def load_data_otimizado(limit=None):  # TODAS as imagens do dataset
    """Carrega dados com verificações melhoradas"""
    images = []
    masks = []
    
    all_image_files = sorted(os.listdir(images_folder))
    if limit is None:
        image_files = all_image_files
        print(f"🔍 Carregando TODAS as {len(all_image_files)} imagens do dataset...")
    else:
        image_files = all_image_files[:limit]
        print(f"🔍 Carregando até {limit} imagens...")
    
    for image_file in image_files:
        base_name, ext = os.path.splitext(image_file)
        mask_file = base_name + '_mask' + ext
        
        image_path = os.path.join(images_folder, image_file)
        mask_path = os.path.join(masks_folder, mask_file)
        
        if os.path.exists(image_path) and os.path.exists(mask_path):
            # Carregar imagem
            img = cv2.imread(image_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))  # Manter 128x128 que funciona
            img = img.astype(np.float32) / 255.0
            
            # Carregar máscara
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            mask = cv2.resize(mask, (128, 128))
            
            # Verificar se máscara tem conteúdo
            if mask.max() > 0:
                mask = (mask > 0).astype(np.float32)  # Usar threshold baixo
                mask = np.expand_dims(mask, axis=-1)
                
                images.append(img)
                masks.append(mask)
                
                mask_pixels = np.sum(mask)
                print(f"   ✓ {image_file}: {mask_pixels:.0f} pixels de estrada")
            else:
                print(f"   ⚠️ {image_file}: Máscara vazia, pulando...")
    
    return np.array(images), np.array(masks)

def unet_fast_otimizado(input_size=(128, 128, 3)):
    """U-Net simplificada mas com algumas melhorias"""
    inputs = Input(input_size)
    
    # Encoder 
    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)  # Adicionar batch norm
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Bottleneck com dropout
    c3 = Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)  # Dropout leve
    
    # Decoder 
    u4 = UpSampling2D((2, 2))(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(64, 3, activation='relu', padding='same')(u4)
    c4 = BatchNormalization()(c4)
    
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(32, 3, activation='relu', padding='same')(u5)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(c5)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Carregar dados
images, masks = load_data_otimizado()  # TODAS as imagens do dataset

if len(images) == 0:
    print("❌ Nenhuma imagem carregada! Verifique os dados.")
    exit()

print(f"📊 Total de imagens carregadas: {len(images)}")

# Criar modelo
print("🔧 Criando modelo U-Net Fast Otimizado...")
model = unet_fast_otimizado()

# Compilar com configurações otimizadas
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("🎯 Iniciando treinamento otimizado...")

# Treinar (usar validação com dataset completo)
if len(images) >= 8:
    # Dividir em treino/validação
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42  # 20% para validação
    )
    
    print(f"📊 Dataset completo dividido:")
    print(f"   • Treino: {len(train_images)} imagens")
    print(f"   • Validação: {len(val_images)} imagens")
    
    history = model.fit(
        train_images, train_masks,
        epochs=15,  # Mais epochs para dataset maior
        validation_data=(val_images, val_masks),
        verbose=1,
        batch_size=2  # Batch maior para dataset completo
    )
    test_image = val_images[0]
    test_mask_real = val_masks[0]
else:
    # Poucos dados, treinar sem validação
    history = model.fit(
        images, masks,
        epochs=12,
        verbose=1,
        batch_size=1
    )
    test_image = images[0]
    test_mask_real = masks[0]

print("✅ Treinamento concluído!")

# Fazer previsão
print("🔮 Fazendo previsão...")
test_image_input = np.expand_dims(test_image, axis=0)
predicted_mask = model.predict(test_image_input)

print(f"📊 Estatísticas da previsão:")
print(f"   • Min: {predicted_mask.min():.4f}")
print(f"   • Max: {predicted_mask.max():.4f}")
print(f"   • Média: {predicted_mask.mean():.4f}")

# Testar múltiplos thresholds
thresholds = [0.04,0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
print(f"\n🎚️ Testando thresholds:")

resultados_threshold = {}
for thresh in thresholds:
    mask_test = (predicted_mask[0] > thresh).astype(np.uint8)
    white_pixels = np.sum(mask_test)
    percentage = (white_pixels / (128*128)) * 100
    
    resultados_threshold[thresh] = {
        'pixels': white_pixels,
        'percentage': percentage,
        'mask': mask_test
    }
    
    print(f"   • Threshold {thresh}: {white_pixels} pixels ({percentage:.1f}%)")

# Escolher melhor threshold (baseado em ter detecção mas não demais)
best_threshold = 0.1
best_score = 0

for thresh, dados in resultados_threshold.items():
    percentage = dados['percentage']
    # Score baseado em ter entre 1% e 25% da imagem detectada
    if 1 <= percentage <= 25:
        score = 100 - abs(percentage - 8)  # Ideal em torno de 8%
        if score > best_score:
            best_score = score
            best_threshold = thresh

print(f"\n🏆 Melhor threshold escolhido: {best_threshold}")
print(f"   • Detecção: {resultados_threshold[best_threshold]['percentage']:.1f}%")

# Salvar resultados
print(f"\n💾 Salvando resultados em {output_folder}/...")

# 1. Imagem original
original_bgr = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
original_path = os.path.join(output_folder, 'imagem_original.png')
cv2.imwrite(original_path, (original_bgr * 255).astype(np.uint8))

# 2. Máscara prevista (melhor threshold)
best_mask = resultados_threshold[best_threshold]['mask']
mask_to_save = (best_mask.squeeze() * 255).astype(np.uint8)
mask_path = os.path.join(output_folder, f'mascara_prevista_threshold_{best_threshold}.png')
cv2.imwrite(mask_path, mask_to_save)

# 3. Máscara real
real_mask = (test_mask_real.squeeze() * 255).astype(np.uint8)
real_mask_path = os.path.join(output_folder, 'mascara_real.png')
cv2.imwrite(real_mask_path, real_mask)

# 4. Todas as máscaras por threshold
for thresh in thresholds:
    thresh_mask = (resultados_threshold[thresh]['mask'].squeeze() * 255).astype(np.uint8)
    thresh_path = os.path.join(output_folder, f'mascara_threshold_{thresh:.2f}.png')
    cv2.imwrite(thresh_path, thresh_mask)

# 5. Comparação visual
try:
    plt.figure(figsize=(20, 8))
    
    # Plot principal: Original, Real, Melhor Prevista
    plt.subplot(2, 4, 1)
    plt.imshow(test_image)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(test_mask_real.squeeze(), cmap='gray')
    plt.title('Máscara Real')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(best_mask.squeeze(), cmap='gray')
    plt.title(f'Melhor Prevista\n(Threshold {best_threshold})')
    plt.axis('off')
    
    # Comparação lado a lado
    mask_3ch = cv2.cvtColor(mask_to_save, cv2.COLOR_GRAY2BGR)
    real_mask_3ch = cv2.cvtColor(real_mask, cv2.COLOR_GRAY2BGR)
    original_display = (original_bgr * 255).astype(np.uint8)
    comparison = np.hstack((original_display, real_mask_3ch, mask_3ch))
    
    plt.subplot(2, 4, 4)
    plt.imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    plt.title('Original | Real | Prevista')
    plt.axis('off')
    
    # Mostrar diferentes thresholds
    for i, thresh in enumerate([0.1, 0.2, 0.3, 0.4]):
        plt.subplot(2, 4, 5 + i)
        plt.imshow(resultados_threshold[thresh]['mask'].squeeze(), cmap='gray')
        plt.title(f'Threshold {thresh}\n{resultados_threshold[thresh]["percentage"]:.1f}%')
        plt.axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'analise_completa.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Análise visual salva em: {plot_path}")
    
except Exception as e:
    print(f"⚠️ Erro ao criar visualização: {e}")

# Salvar comparação lado a lado
comparison_path = os.path.join(output_folder, 'comparacao_lado_a_lado.png')
cv2.imwrite(comparison_path, comparison)

print(f"\n🎉 RESULTADOS FINAIS:")
print(f"   📁 Pasta: {output_folder}/")
print(f"   🎯 Melhor threshold: {best_threshold}")
print(f"   📊 Detecção: {resultados_threshold[best_threshold]['percentage']:.1f}%")
print(f"   🖼️ Principais arquivos:")
print(f"      • {mask_path}")
print(f"      • {comparison_path}")
print(f"      • analise_completa.png")

print(f"\n💡 Para usar este threshold no script principal:")
print(f"   predicted_mask_binary = (predicted_mask > {best_threshold}).astype(np.uint8)")
