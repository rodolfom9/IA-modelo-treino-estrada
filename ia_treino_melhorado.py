# Script otimizado para resultados melhores
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def carregar_dados_melhorado():
    """Carrega dados com verificações adicionais"""
    
    images_folder = 'dataset'
    masks_folder = 'mascara'
    
    images = []
    masks = []
    
    print("🔍 Carregando e verificando dados...")
    
    image_files = sorted(os.listdir(images_folder))
    
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
            img = cv2.resize(img, (256, 256))  # Redimensionar para tamanho fixo
            img = img.astype(np.float32) / 255.0
            
            # Carregar máscara
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            mask = cv2.resize(mask, (256, 256))
            
            # VERIFICAÇÃO IMPORTANTE: Se a máscara tem pixels brancos
            if mask.max() < 50:
                print(f"⚠️  Máscara {mask_file} parece estar muito escura (max: {mask.max()})")
                # Normalizar diferente se necessário
                mask = (mask > 10).astype(np.float32)
            else:
                mask = (mask > 128).astype(np.float32)
            
            mask = np.expand_dims(mask, axis=-1)
            
            # Verificar se temos detecções na máscara
            if np.sum(mask) > 0:
                images.append(img)
                masks.append(mask)
                print(f"✅ {image_file}: {np.sum(mask)} pixels de estrada")
            else:
                print(f"⚠️  {image_file}: Nenhum pixel de estrada detectado na máscara")
    
    return np.array(images), np.array(masks)

def unet_melhorado(input_size=(256, 256, 3)):
    """U-Net com melhorias para evitar máscaras pretas"""
    
    inputs = Input(input_size)
    
    # Encoder com Batch Normalization
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.3)(c4)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    
    # Decoder
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)
    
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)
    
    # Saída com sigmoid
    outputs = Conv2D(1, 1, activation='sigmoid')(c7)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def treinar_modelo_melhorado():
    """Treinamento otimizado"""
    
    print("🚀 TREINAMENTO MELHORADO")
    print("=" * 50)
    
    # Carregar dados
    images, masks = carregar_dados_melhorado()
    
    if len(images) == 0:
        print("❌ Nenhum dado carregado!")
        return None
    
    print(f"📊 Dados carregados: {len(images)} imagens")
    
    # Dividir dados
    train_images, test_images, train_masks, test_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )
    
    # Criar modelo
    model = unet_melhorado()
    
    # Compilar com configurações otimizadas
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("🔧 Iniciando treinamento...")
    
    # Callbacks para melhor treinamento
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
    ]
    
    # Treinar
    history = model.fit(
        train_images, train_masks,
        epochs=15,  # Mais epochs
        batch_size=4,  # Batch menor para dataset pequeno
        validation_data=(test_images, test_masks),
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ Treinamento concluído!")
    
    # Fazer previsão
    test_image = test_images[0:1]
    prediction = model.predict(test_image)
    
    print(f"📊 Previsão - Min: {prediction.min():.4f}, Max: {prediction.max():.4f}, Média: {prediction.mean():.4f}")
    
    # Testar múltiplos thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    output_folder = "resultado_mascara_melhorado"
    os.makedirs(output_folder, exist_ok=True)
    
    for i, thresh in enumerate(thresholds):
        mask_binary = (prediction[0] > thresh).astype(np.uint8) * 255
        white_pixels = np.sum(mask_binary > 0)
        
        # Salvar máscara
        mask_path = os.path.join(output_folder, f"mascara_threshold_{thresh:.1f}.png")
        cv2.imwrite(mask_path, mask_binary.squeeze())
        
        print(f"🎯 Threshold {thresh}: {white_pixels} pixels detectados → {mask_path}")
    
    # Salvar imagem original para comparação
    original_bgr = cv2.cvtColor(test_image[0], cv2.COLOR_RGB2BGR)
    original_path = os.path.join(output_folder, "imagem_original.png")
    cv2.imwrite(original_path, (original_bgr * 255).astype(np.uint8))
    
    print(f"\n✅ Resultados salvos em: {output_folder}/")
    print("📋 Verifique os diferentes thresholds para escolher o melhor!")
    
    return model

if __name__ == "__main__":
    treinar_modelo_melhorado()
