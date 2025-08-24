# Script para corrigir máscaras pretas - Reduzir threshold
import cv2
import numpy as np
import os

def corrigir_mascara_preta():
    """
    Reprocessa a máscara com threshold mais baixo
    """
    
    print("🔧 CORREÇÃO DE MÁSCARA PRETA")
    print("=" * 40)
    
    # Tentar encontrar arquivo de modelo ou resultados anteriores
    # Por agora, vamos criar uma versão melhorada do script principal
    
    # Valores de threshold para testar
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3]
    
    print("💡 SOLUÇÕES RECOMENDADAS:")
    print("1. Usar thresholds mais baixos:")
    for t in thresholds:
        print(f"   - Threshold {t}: Mais sensível à detecção")
    
    print("\n2. Modificar o script principal:")
    print("   - Diminuir o threshold de 0.5 para 0.1")
    print("   - Aumentar epochs de 5 para 10")
    print("   - Verificar se máscaras de treino estão corretas")
    
    print("\n3. Melhorar dados de treino:")
    print("   - Verificar se máscaras têm pixels brancos")
    print("   - Garantir que estradas estão bem marcadas")
    print("   - Usar mais imagens variadas")

if __name__ == "__main__":
    corrigir_mascara_preta()
