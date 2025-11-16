# detector-quedas-ia
Aplicação de IA para detecção de quedas usando Pose Detection. O sistema utiliza o modelo 100% aberto YOLOv8-Pose para analisar os keypoints do corpo e identificar quedas em tempo real. A solução é apresentada em uma interface web interativa construída com Streamlit, demonstrando uma aplicação de alta relevância na prevenção de quedas.

# Sistema de Detecção de Quedas

## Como Acessar

### 1. Notebook Colab-
https://colab.research.google.com/drive/1PExzYAG-rOZkA944bWF10hQI2NA3nNo3#scrollTo=5BxAkgC-3xcC

### 2. App Web (Streamlit Cloud) - OPCIONAL
[Clique aqui para testar o app online](https://seu-usuario-detector-quedas.streamlit.app)

### 3. Rodar Localmente


## Estrutura do Projeto
- `app.py`: Interface web com Streamlit
- `processar_quedas.py`: Script de treinamento
- `output/features_de_queda_final.csv`: Dataset
- `detector_modelo_melhor.pkl`: Modelo treinado
- `scaler.pkl`: Normalizador
