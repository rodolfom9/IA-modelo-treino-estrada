import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def analisar_resultados():
    """
    Analisa os resultados na pasta resultado_mascara e diagnóstica problemas
    """
    
    pasta_resultados = "resultado_mascara"
    
    print("🔍 ANÁLISE DOS RESULTADOS")
    print("=" * 50)
    
    if not os.path.exists(pasta_resultados):
        print("❌ Pasta resultado_mascara não encontrada!")
        return
    
    # Listar arquivos
    arquivos = os.listdir(pasta_resultados)
    print(f"📁 Arquivos encontrados: {len(arquivos)}")
    for arquivo in arquivos:
        tamanho = os.path.getsize(os.path.join(pasta_resultados, arquivo))
        print(f"   • {arquivo} - {tamanho} bytes")
    
    # Analisar cada imagem
    print("\n🖼️  ANÁLISE DAS IMAGENS:")
    print("-" * 30)
    
    # 1. Analisar máscara prevista
    mascara_path = os.path.join(pasta_resultados, "resultado_mascara_prevista.png")
    if os.path.exists(mascara_path):
        mascara = cv2.imread(mascara_path, cv2.IMREAD_GRAYSCALE)
        if mascara is not None:
            print(f"📊 MÁSCARA PREVISTA:")
            print(f"   • Dimensões: {mascara.shape}")
            print(f"   • Valores únicos: {np.unique(mascara)}")
            print(f"   • Valor mínimo: {mascara.min()}")
            print(f"   • Valor máximo: {mascara.max()}")
            print(f"   • Pixels brancos: {np.sum(mascara > 128)}")
            print(f"   • Pixels pretos: {np.sum(mascara <= 128)}")
            
            if mascara.max() == 0:
                print("   ⚠️  PROBLEMA: Máscara está completamente preta!")
            elif np.sum(mascara > 128) < 100:
                print("   ⚠️  PROBLEMA: Muito poucos pixels detectados!")
            else:
                print("   ✅ Máscara parece ter detecções")
        else:
            print("   ❌ Erro ao carregar máscara")
    
    # 2. Analisar imagem original
    original_path = os.path.join(pasta_resultados, "imagem_original_teste.png")
    if os.path.exists(original_path):
        original = cv2.imread(original_path)
        if original is not None:
            print(f"\n📊 IMAGEM ORIGINAL:")
            print(f"   • Dimensões: {original.shape}")
            print(f"   • Tipo: {original.dtype}")
        else:
            print("\n   ❌ Erro ao carregar imagem original")
    
    # 3. Criar análise visual melhorada
    print(f"\n🔧 CRIANDO ANÁLISE VISUAL MELHORADA...")
    
    if os.path.exists(mascara_path) and os.path.exists(original_path):
        mascara = cv2.imread(mascara_path, cv2.IMREAD_GRAYSCALE)
        original = cv2.imread(original_path)
        
        if mascara is not None and original is not None:
            # Criar visualização com diferentes thresholds
            plt.figure(figsize=(20, 5))
            
            # Original
            plt.subplot(1, 5, 1)
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.title("Original")
            plt.axis('off')
            
            # Máscara como foi salva
            plt.subplot(1, 5, 2)
            plt.imshow(mascara, cmap='gray')
            plt.title(f"Máscara Atual\nMax: {mascara.max()}")
            plt.axis('off')
            
            # Diferentes thresholds para debugging
            thresholds = [64, 128, 192]
            for i, thresh in enumerate(thresholds):
                plt.subplot(1, 5, 3 + i)
                mask_thresh = (mascara > thresh).astype(np.uint8) * 255
                plt.imshow(mask_thresh, cmap='gray')
                plt.title(f"Threshold > {thresh}\nPixels: {np.sum(mask_thresh > 0)}")
                plt.axis('off')
            
            plt.tight_layout()
            analise_path = os.path.join(pasta_resultados, "analise_detalhada.png")
            plt.savefig(analise_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Análise salva em: {analise_path}")
            plt.close()
            
            # Criar histogram
            plt.figure(figsize=(10, 6))
            plt.hist(mascara.flatten(), bins=50, alpha=0.7, edgecolor='black')
            plt.title("Histograma dos Valores da Máscara")
            plt.xlabel("Valor do Pixel")
            plt.ylabel("Frequência")
            plt.grid(True, alpha=0.3)
            
            hist_path = os.path.join(pasta_resultados, "histograma_mascara.png")
            plt.savefig(hist_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Histograma salvo em: {hist_path}")
            plt.close()
    
    print(f"\n💡 DIAGNÓSTICOS E SOLUÇÕES:")
    print("-" * 40)
    
    # Verificar problemas comuns
    if os.path.exists(mascara_path):
        mascara = cv2.imread(mascara_path, cv2.IMREAD_GRAYSCALE)
        if mascara is not None:
            if mascara.max() == 0:
                print("🔴 PROBLEMA: Máscara completamente preta")
                print("   💡 Soluções:")
                print("   1. Reduzir threshold (usar 0.1 em vez de 0.5)")
                print("   2. Treinar por mais epochs")
                print("   3. Verificar se as máscaras de treino estão corretas")
                print("   4. Ajustar learning rate")
            
            elif np.sum(mascara > 128) < 100:
                print("🟡 PROBLEMA: Muito poucos pixels detectados")
                print("   💡 Soluções:")
                print("   1. Reduzir threshold")
                print("   2. Melhorar qualidade dos dados de treino")
                print("   3. Aumentar número de epochs")
            
            elif np.sum(mascara > 128) > (mascara.size * 0.8):
                print("🟡 PROBLEMA: Muitos pixels detectados (possível ruído)")
                print("   💡 Soluções:")
                print("   1. Aumentar threshold")
                print("   2. Adicionar regularização")
                print("   3. Verificar overfitting")
            
            else:
                print("✅ Máscara parece razoável, mas pode ser melhorada")
                print("   💡 Melhorias:")
                print("   1. Mais dados de treino")
                print("   2. Data augmentation")
                print("   3. Fine-tuning do threshold")

if __name__ == "__main__":
    analisar_resultados()
