# Sistema de Detecção de Quedas 
Aplicação de IA para detecção de quedas usando Pose Detection. O sistema utiliza o modelo 100% aberto YOLOv8-Pose para analisar os keypoints do corpo e identificar quedas em tempo real. A solução é apresentada em uma interface web interativa construída com Streamlit, demonstrando uma aplicação de alta relevância na prevenção de quedas.


##  Objetivo

Detectar quedas em vídeos utilizando análise de postura com redes neurais convolucionais (YOLOv8-Pose) combinada com algoritmos de classificação (Random Forest, MLP, SVM, KNN, etc).

## Dataset e Modelos

- **Total de vídeos processados**: 89
- **Vídeos de queda**: 50
- **Vídeos de atividades normais**: 39
- **Modelos treinados**: 11 classificadores diferentes
- **Melhor modelo**: Random Forest (d=10) com F1-Score: 0.94

## Como Acessar

### 1. Notebook Colab-
https://colab.research.google.com/drive/1PExzYAG-rOZkA944bWF10hQI2NA3nNo3#scrollTo=5BxAkgC-3xcC

### **Opção 2: Rodando Localmente (Python)**

#### **Pré-requisitos**
- Python 3.12+
- pip instalado

#### **Passo 1: Clonar o Repositório**

#### **Passo 2: Instalar Dependências**

#### **Passo 3: Rodar o App**

#### **Passo 4: Copie os 03 arquivos .pkl e coloque na pasta do app_quedas.py**

### **Passo 5: streamlit run app_quedas.py\n"**

**O app abrirá no navegador em:** `http://localhost:8501`

#### **Para Testar o app:**

1. **Incluir vídeo na aplicação**: 
   - Faça upload de um vídeo
   - Veja a predição (Queda ou Normal)
   - Veja a probabilidade e as features extraídas

2. **Resultados do Treinamento**:
   - Tabela comparativa de todos os modelos
   - Gráficos de F1-Score
   - Métricas de Accuracy, Precision, Recall

## Estrutura do Projeto
- `app.py`: Interface web com Streamlit
- `processar_quedas.py`: Script de treinamento
- `output/features_de_queda_final.csv`: Dataset
- `detector_modelo_melhor.pkl`: Modelo treinado
- `scaler.pkl`: Normalizador
  
## Vídeo 
https://youtu.be/O3pHX3Qhj2E
