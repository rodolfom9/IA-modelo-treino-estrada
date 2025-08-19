#!/bin/bash

# Script para instalar dependências do projeto de IA
echo "================================================"
echo "Instalando dependências para o projeto de IA..."
echo "================================================"

# Verificar se estamos no diretório correto
if [ ! -f "script.py" ] || [ ! -f "ia_treino.py" ]; then
    echo "Erro: Execute este script no diretório do projeto (onde estão os arquivos script.py e ia_treino.py)"
    exit 1
fi

# Ativar o ambiente virtual se existir
if [ -d ".venv" ]; then
    echo "Ativando ambiente virtual..."
    source .venv/bin/activate
    echo "Ambiente virtual ativado!"
else
    echo "Ambiente virtual não encontrado. Criando um novo..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Ambiente virtual criado e ativado!"
fi

# Atualizar pip
echo "Atualizando pip..."
pip install --upgrade pip

# Instalar dependências básicas
echo "Instalando NumPy..."
pip install numpy

echo "Instalando OpenCV..."
pip install opencv-python

echo "Instalando Pillow..."
pip install pillow

echo "Instalando scikit-learn..."
pip install scikit-learn

echo "Instalando matplotlib..."
pip install matplotlib

echo "Instalando TensorFlow..."
pip install tensorflow

echo "================================================"
echo "Instalação concluída!"
echo "================================================"

# Verificar se as instalações foram bem-sucedidas
echo "Verificando instalações..."
python -c "import numpy; print('✓ NumPy:', numpy.__version__)"
python -c "import cv2; print('✓ OpenCV:', cv2.__version__)"
python -c "import PIL; print('✓ Pillow:', PIL.__version__)"
python -c "import sklearn; print('✓ scikit-learn:', sklearn.__version__)"
python -c "import matplotlib; print('✓ matplotlib:', matplotlib.__version__)"
python -c "import tensorflow; print('✓ TensorFlow:', tensorflow.__version__)"

echo "================================================"
echo "Todas as dependências foram instaladas com sucesso!"
echo "Para executar os scripts, use:"
echo "  source .venv/bin/activate"
echo "  python script.py"
echo "  python ia_treino.py"
echo "================================================"
