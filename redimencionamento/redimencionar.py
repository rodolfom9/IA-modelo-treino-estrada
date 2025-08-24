import cv2
import os
import numpy as np
from PIL import Image

def redimensionar_imagens(pasta_origem, pasta_destino, novo_tamanho=(256, 256)):
    """
    Redimensiona todas as imagens de uma pasta para um novo tamanho
    
    Args:
        pasta_origem (str): Caminho da pasta com as imagens originais
        pasta_destino (str): Caminho da pasta onde salvar as imagens redimensionadas
        novo_tamanho (tuple): Novo tamanho (largura, altura) - padrão (256, 256)
    """
    
    # Criar pasta de destino se não existir
    os.makedirs(pasta_destino, exist_ok=True)
    
    # Extensões de imagem suportadas
    extensoes_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    
    # Listar todos os arquivos da pasta origem
    arquivos = os.listdir(pasta_origem)
    arquivos_imagem = [f for f in arquivos if f.lower().endswith(extensoes_validas)]
    
    print(f"📁 Pasta origem: {pasta_origem}")
    print(f"📁 Pasta destino: {pasta_destino}")
    print(f"🔧 Novo tamanho: {novo_tamanho[0]}x{novo_tamanho[1]}")
    print(f"📷 Encontradas {len(arquivos_imagem)} imagens para redimensionar")
    print("-" * 50)
    
    contador_sucesso = 0
    contador_erro = 0
    
    for arquivo in arquivos_imagem:
        try:
            # Caminhos completos
            caminho_origem = os.path.join(pasta_origem, arquivo)
            caminho_destino = os.path.join(pasta_destino, arquivo)
            
            # Carregar imagem usando OpenCV
            imagem = cv2.imread(caminho_origem)
            
            if imagem is None:
                print(f"❌ Erro ao carregar: {arquivo}")
                contador_erro += 1
                continue
            
            # Obter dimensões originais
            altura_original, largura_original = imagem.shape[:2]
            
            # Redimensionar imagem
            imagem_redimensionada = cv2.resize(imagem, novo_tamanho, interpolation=cv2.INTER_LANCZOS4)
            
            # Salvar imagem redimensionada
            sucesso = cv2.imwrite(caminho_destino, imagem_redimensionada)
            
            if sucesso:
                print(f"✅ {arquivo}: {largura_original}x{altura_original} → {novo_tamanho[0]}x{novo_tamanho[1]}")
                contador_sucesso += 1
            else:
                print(f"❌ Erro ao salvar: {arquivo}")
                contador_erro += 1
                
        except Exception as e:
            print(f"❌ Erro ao processar {arquivo}: {str(e)}")
            contador_erro += 1
    
    print("-" * 50)
    print(f"✅ Imagens redimensionadas com sucesso: {contador_sucesso}")
    print(f"❌ Erros: {contador_erro}")
    print(f"📁 Todas as imagens redimensionadas foram salvas em: {pasta_destino}")

def redimensionar_com_aspecto_preservado(pasta_origem, pasta_destino, novo_tamanho=(256, 256)):
    """
    Redimensiona imagens preservando a proporção (aspect ratio) e preenchendo com padding preto
    """
    
    os.makedirs(pasta_destino, exist_ok=True)
    extensoes_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    arquivos = os.listdir(pasta_origem)
    arquivos_imagem = [f for f in arquivos if f.lower().endswith(extensoes_validas)]
    
    print(f"📁 Redimensionando com preservação de aspecto para {novo_tamanho[0]}x{novo_tamanho[1]}")
    print(f"📷 Processando {len(arquivos_imagem)} imagens...")
    print("-" * 50)
    
    contador_sucesso = 0
    
    for arquivo in arquivos_imagem:
        try:
            caminho_origem = os.path.join(pasta_origem, arquivo)
            caminho_destino = os.path.join(pasta_destino, f"resize_{arquivo}")
            
            # Usar PIL para melhor controle de aspecto
            with Image.open(caminho_origem) as img:
                # Converter para RGB se necessário
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calcular novo tamanho preservando aspecto
                img.thumbnail(novo_tamanho, Image.Resampling.LANCZOS)
                
                # Criar imagem com fundo preto do tamanho desejado
                nova_img = Image.new('RGB', novo_tamanho, (0, 0, 0))
                
                # Centralizar a imagem redimensionada
                x = (novo_tamanho[0] - img.width) // 2
                y = (novo_tamanho[1] - img.height) // 2
                nova_img.paste(img, (x, y))
                
                # Salvar
                nova_img.save(caminho_destino)
                print(f"✅ {arquivo} → resize_{arquivo}")
                contador_sucesso += 1
                
        except Exception as e:
            print(f"❌ Erro ao processar {arquivo}: {str(e)}")
    
    print("-" * 50)
    print(f"✅ {contador_sucesso} imagens processadas com preservação de aspecto")

if __name__ == "__main__":
    # Configurações - sempre procurar na pasta onde está o script
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Pasta onde está o script
    pasta_origem = script_dir  # A própria pasta do script (redimencionamento)
    pasta_destino = os.path.join(script_dir, "imagens_256x256")   # Pasta onde salvar as imagens redimensionadas
    
    print("🖼️  REDIMENSIONADOR DE IMAGENS PARA 256x256")
    print("=" * 60)
    
    # Listar arquivos na pasta atual para verificar se há imagens
    arquivos_atuais = os.listdir(pasta_origem)
    extensoes_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    imagens_encontradas = [f for f in arquivos_atuais if f.lower().endswith(extensoes_validas)]
    
    print(f"📍 Pasta do script: {pasta_origem}")
    print(f"📁 Arquivos encontrados: {len(arquivos_atuais)}")
    print(f"🖼️  Imagens encontradas: {len(imagens_encontradas)}")
    
    if imagens_encontradas:
        print(f"\n� Imagens que serão redimensionadas:")
        for img in imagens_encontradas:
            print(f"   • {img}")
        
        print(f"\n🔧 Redimensionando imagens...")
        redimensionar_imagens(pasta_origem, pasta_destino)
    else:
        print(f"\n⚠️  Nenhuma imagem encontrada na pasta atual!")
        print(f"� Coloque arquivos de imagem (.png, .jpg, .jpeg, .bmp, .tiff) na pasta 'redimencionamento'")
        print(f"📍 Pasta atual: {os.path.abspath('.')}")
    
    print("\n" + "=" * 60)
    print("🎯 INSTRUÇÕES:")
    print("1. Coloque suas imagens na pasta 'redimencionamento'")
    print("2. Execute o script novamente")
    print("3. As imagens redimensionadas aparecerão na pasta 'imagens_256x256'")