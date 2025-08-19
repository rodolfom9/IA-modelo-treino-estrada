import json
import numpy as np
import cv2
from PIL import Image # Ou use cv2 para carregar imagens

# Caminho para o arquivo JSON exportado do VIA
via_json_path = 'dataset_rodolfo.json'
# Pasta onde estão as imagens originais
images_folder = 'dataset/'
# Pasta onde serão salvas as máscaras
masks_output_folder = 'mascara/'

# Crie a pasta de saída se não existir
import os
if not os.path.exists(masks_output_folder):
    os.makedirs(masks_output_folder)

with open(via_json_path, 'r') as f:
    via_data = json.load(f)

# O formato exato do JSON pode variar um pouco dependendo da versão do VIA
# e como você exportou. Este é um exemplo comum:
# via_data['_via_img_metadata'] contém informações sobre cada imagem e suas anotações

for image_id, image_info in via_data['_via_img_metadata'].items():
    image_filename = image_info['filename']
    regions = image_info['regions']

    # Carregue a imagem original para obter as dimensões
    try:
        #img = cv2.imread(os.path.join(images_folder, image_filename))
        #height, width, _ = img.shape
        # Ou usando PIL:
        img_pil = Image.open(os.path.join(images_folder, image_filename))
        width, height = img_pil.size

    except FileNotFoundError:
        print(f"Imagem original não encontrada: {image_filename}. Pulando.")
        continue


    # Crie uma máscara em branco (preenchida com zeros)
    # Use dtype=np.uint8 ou np.float32 dependendo do que você precisa para o treinamento
    mask = np.zeros((height, width), dtype=np.uint8)

    # Desenhe cada polígono de rodovia na máscara
    for region in regions:
        # Suponha que você anotou rodovias como polígonos
        if region['shape_attributes']['name'] == 'polygon':
            all_points_x = region['shape_attributes']['all_points_x']
            all_points_y = region['shape_attributes']['all_points_y']

            # Combine X e Y em uma lista de pontos para o OpenCV
            polygon_points = np.array([[x, y] for x, y in zip(all_points_x, all_points_y)], dtype=np.int32)

            # Preencha o polígono na máscara (valor 255 para a rodovia)
            cv2.fillPoly(mask, [polygon_points], 255) # Use 1 ou 255

    # Salve a máscara gerada
    mask_filename = os.path.splitext(image_filename)[0] + '_mask.png'
    cv2.imwrite(os.path.join(masks_output_folder, mask_filename), mask)
    print(f"Máscara salva para {image_filename}")

print("Processo de conversão concluído.")