import streamlit as st
import numpy as np
from ultralytics import YOLO
import joblib
import tempfile
import os
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Detecção de Quedas",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏥 Sistema de Detecção de Quedas")
st.markdown("Monitoramento Inteligente com IA - YOLOv8 Pose + ML")

@st.cache_resource
def load_models():
    try:
        pose_model = YOLO('yolov8n-pose.pt')
        classifier = joblib.load('detector_modelo_melhor.pkl')
        scaler = joblib.load('scaler.pkl')
        rf_model = joblib.load('detector_modelo_rf.pkl')
        return pose_model, classifier, scaler, rf_model
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
        return None, None, None, None

def extract_features_from_video(video_path, pose_model):
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6

    hip_y_series = []
    shoulder_y_series = []
    distances = []
    velocidades = []

    try:
        results = pose_model(source=video_path, stream=True, conf=0.5, verbose=False)

        for r in results:
            if r.keypoints is not None and r.keypoints.xy is not None and r.keypoints.xy.numel() > 0:
                keypoints = r.keypoints.xy[0]
                left_hip_y = float(keypoints[LEFT_HIP][1])
                right_hip_y = float(keypoints[RIGHT_HIP][1])
                left_shoulder_y = float(keypoints[LEFT_SHOULDER][1])
                right_shoulder_y = float(keypoints[RIGHT_SHOULDER][1])

                if (not np.isnan(left_hip_y) and not np.isnan(right_hip_y) and
                    not np.isnan(left_shoulder_y) and not np.isnan(right_shoulder_y) and
                    left_hip_y > 0 and right_hip_y > 0 and left_shoulder_y > 0 and right_shoulder_y > 0):

                    hip_y = (left_hip_y + right_hip_y) / 2.0
                    shoulder_y = (left_shoulder_y + right_shoulder_y) / 2.0

                    hip_y_series.append(hip_y)
                    shoulder_y_series.append(shoulder_y)

                    dist = abs(shoulder_y - hip_y)
                    distances.append(dist)

                    if len(hip_y_series) > 1:
                        vel = hip_y_series[-1] - hip_y_series[-2]
                        velocidades.append(vel)

        if len(hip_y_series) < 3:
            return None

        max_vel = max(velocidades) if velocidades else 0.0
        media_vel = np.mean(velocidades) if velocidades else 0.0
        diferenca_alt = hip_y_series[0] - hip_y_series[-1]
        std_alt = float(np.std(hip_y_series))
        terminou_horizontal = 1 if hip_y_series[-1] >= shoulder_y_series[-1] else 0
        distancia_final = float(distances[-1]) if distances else 0.0
        distancia_min = float(min(distances)) if distances else 0.0
        taxa_var = (max(hip_y_series) - min(hip_y_series)) / len(hip_y_series)

        features = {
            'max_velocidade_vertical': float(max_vel),
            'media_velocidade_vertical': float(media_vel),
            'diferenca_altura': float(diferenca_alt),
            'std_altura': float(std_alt),
            'terminou_horizontal': float(terminou_horizontal),
            'distancia_final_ombro_quadril': float(distancia_final),
            'distancia_minima': float(distancia_min),
            'taxa_variacao': float(taxa_var)
        }

        return features

    except Exception as e:
        st.error(f"Erro ao processar vídeo: {e}")
        return None

pose_model, classifier, scaler, rf_model = load_models()

if pose_model is None:
    st.error("Falha ao carregar modelos. Verifique arquivos .pkl e .pt.")
    st.stop()

st.sidebar.header("Menu")
menu = st.sidebar.radio(
    "Escolha uma opção:",
    ["Análise de Vídeo", "Resultados do Treinamento", "Descrição do Projeto"]
)

if menu == "Análise de Vídeo":
    st.header("Análise de Vídeo")

    uploaded_file = st.file_uploader("Carregue um vídeo para análise", type=['mp4', 'avi', 'mov', 'mkv'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()

        col_video, col_results = st.columns([1, 1])
        with col_video:
            st.subheader("Vídeo Enviado")
            st.video(tfile.name)

        with col_results:
            st.subheader("Resultado da IA")
            with st.spinner("Processando vídeo..."):
                features = extract_features_from_video(tfile.name, pose_model)
            if features:
                # Normalização
                feature_list = [
                    features['max_velocidade_vertical'],
                    features['media_velocidade_vertical'],
                    features['diferenca_altura'],
                    features['std_altura'],
                    features['terminou_horizontal'],
                    features['distancia_final_ombro_quadril'],
                    features['distancia_minima'],
                    features['taxa_variacao']
                ]
                features_array = np.array(feature_list).reshape(1, -1)
                features_scaled = scaler.transform(features_array)
                prediction = classifier.predict(features_scaled)[0]
                proba = classifier.predict_proba(features_scaled)[0]
                if prediction == 1:
                    st.error("🚨 QUE DA DETECTADA!")
                else:
                    st.success("✅ Atividade Normal")

                st.metric("Probabilidade de Queda", f"{proba[1]*100:.2f}%")
                st.metric("Confiança", f"{max(proba)*100:.2f}%")

                features_df = pd.DataFrame({
                    'Feature': features.keys(),
                    'Valor': [f"{v:.2f}" for v in features.values()]
                })
                st.dataframe(features_df, use_container_width=True)
            else:
                st.warning("Não foi possível processar o vídeo.")

        os.unlink(tfile.name)

elif menu == "Resultados do Treinamento":
    st.header("Resultados do Treinamento")
    try:
        df_resultados = pd.read_csv("resultados_cv_modelos.csv")
        st.write("Resultados Validação Cruzada 5-Fold:")
        st.dataframe(df_resultados, use_container_width=True)

        df_plot = df_resultados.sort_values('F1-Score_Mean', ascending=True)
        fig = go.Figure(data=[go.Bar(
            y=df_plot['Modelo'],
            x=df_plot['F1-Score_Mean'],
            orientation='h',
            marker=dict(color='coral')
        )])
        fig.update_layout(height=400, xaxis_title="F1-Score")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f'Erro ao ler resultados_cv_modelos.csv: {e}')

else:
    st.header("Descrição do Projeto")
    st.write("""
    Este projeto é um sistema em desenvolvimento para detecção de quedas usando YOLOv8 (Pose) e Machine Learning.

    - **Backend**: ultralytics, scikit-learn
    - **Interface**: Streamlit
    - **Autoras**: [Cleide Lustosa e Erika Borges]
    """)

st.markdown("---")
st.caption("Sistema de Detecção de Quedas - IA Aberta com Streamlit")
