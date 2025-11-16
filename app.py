import streamlit as st
import tempfile
from ultralytics import YOLO
import cv2  # Importamos o OpenCV
import os
import time

# --- 1. CARREGAR O MODELO DE IA ---
@st.cache_resource
def carregar_modelo():
    """Carrega o modelo YOLOv8-Pose e o armazena em cache."""
    print("Carregando modelo YOLOv8-Pose...")
    model = YOLO('yolov8n-pose.pt')
    print("Modelo carregado.")
    return model

model = carregar_modelo()

# --- 2. FUNÇÃO DE PROCESSAMENTO (A LÓGICA DO COLAB) ---
def processar_video_de_queda(caminho_video_entrada):
    """
    Analisa um arquivo de vídeo frame a frame para detectar quedas
    e retorna o resultado E o caminho para o vídeo processado.
    """
    
    # --- Configuração da Lógica ---
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    last_hip_y = 0
    fall_detected = False
    FALL_VELOCITY_THRESHOLD = 15.0 
    
    # --- Preparação dos Arquivos de Vídeo ---
    
    # Abrir o vídeo de entrada com OpenCV
    cap = cv2.VideoCapture(caminho_video_entrada)
    if not cap.isOpened():
        return "Erro: Não foi possível abrir o arquivo de vídeo.", None

    # Pegar propriedades do vídeo (largura, altura, FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30 # Define um padrão caso o FPS não seja encontrado

    # Criar um arquivo de vídeo temporário para a SAÍDA (a "prova")
    # Este arquivo terá os pontos da pose desenhados
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_out_file:
        caminho_video_saida = tmp_out_file.name

    # Definir o 'escritor' de vídeo (codec 'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(caminho_video_saida, fourcc, fps, (width, height))

    frame_number = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # Fim do vídeo

            # Enviar o FRAME para o modelo YOLO
            results = model(frame, stream=True, conf=0.5, verbose=False)

            frame_processado = frame # Começa com o frame original
            
            for r in results:
                # --- Lógica de Detecção (igual ao Colab) ---
                if r.keypoints.xy.numel() > 0: 
                    keypoints = r.keypoints.xy[0] 
                    left_hip_y = keypoints[LEFT_HIP][1]
                    right_hip_y = keypoints[RIGHT_HIP][1]
                    left_shoulder_y = keypoints[LEFT_SHOULDER][1]
                    right_shoulder_y = keypoints[RIGHT_SHOULDER][1]
                    
                    if not (left_hip_y == 0 or right_hip_y == 0 or left_shoulder_y == 0 or right_shoulder_y == 0):
                        current_hip_y = (left_hip_y + right_hip_y) / 2
                        current_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
                        
                        if last_hip_y != 0: 
                            vertical_velocity = current_hip_y - last_hip_y
                            
                            if vertical_velocity > FALL_VELOCITY_THRESHOLD:
                                if current_hip_y >= current_shoulder_y:
                                    print(f"Frame {frame_number}: *** QUEDA CONFIRMADA ***")
                                    fall_detected = True
                        
                        last_hip_y = current_hip_y
                
                # --- Preparar a "Prova" ---
                # Pega o frame com a pose desenhada pelo YOLO
                frame_processado = r.plot()
            
            # Escreve o frame (com ou sem pose) no vídeo de saída
            video_writer.write(frame_processado)
            frame_number += 1
            
            if fall_detected: # Para o processamento se a queda foi detectada
                break
                
    except Exception as e:
        print(f"Erro no loop de processamento: {e}")
        return f"Erro na análise: {e}", None
    finally:
        # Libera os arquivos de vídeo
        cap.release()
        video_writer.release()

    # --- Retorna o resultado e o caminho do vídeo de "prova" ---
    if fall_detected:
        return "ALERTA: Queda Detectada!", caminho_video_saida
    else:
        return "Normal: Nenhuma queda detectada.", caminho_video_saida

# --- 3. CONFIGURAÇÃO DA PÁGINA (Interface do Usuário) ---
st.set_page_config(page_title="Detector de Quedas", layout="wide")
st.title("Sistema de Detecção de Quedas (Projeto IA)")
st.write("Faça o upload de um vídeo (.mp4) para análise.")

uploaded_file = st.file_uploader("Escolha um arquivo de vídeo", type=["mp4"])

if uploaded_file is not None:
    # Mostra o player do vídeo ORIGINAL
    st.subheader("Vídeo Original:")
    st.video(uploaded_file)
    
    # Salva o vídeo temporariamente para a IA poder lê-lo
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        caminho_temporario = tmp_file.name

    # Botão para iniciar a análise
    if st.button("Analisar Vídeo"):
        with st.spinner("Analisando... A IA (CNN) está processando os frames..."):
            
            # Chama nossa função REAL de IA
            resultado_texto, caminho_video_processado = processar_video_de_queda(caminho_temporario)
            
            st.subheader("Resultado da Análise:")
            if "ALERTA" in resultado_texto:
                st.error(resultado_texto) 
            else:
                st.success(resultado_texto)
            
            # --- A "PROVA" ---
            # Mostra o vídeo processado (com a pose desenhada)
            if caminho_video_processado:
                st.subheader("Prova da Análise (Vídeo Processado pela IA):")
                st.video(caminho_video_processado)