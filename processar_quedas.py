import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import joblib
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SISTEMA DE DETECCAO DE QUEDAS - PROCESSAMENTO LOCAL")
print("="*70 + "\n")

BASE_PATH = r"C:\Users\Família\Downloads\FallDataset"
OUTPUT_DIR = "./output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Caminho base: {BASE_PATH}")
print(f"Diretorio de saida: {OUTPUT_DIR}\n")

if not os.path.exists(BASE_PATH):
    print(f"ERRO: Caminho nao encontrado: {BASE_PATH}")
    exit()

print("="*70)
print("FASE 1: CARREGAR MODELO YOLOV8-POSE")
print("="*70 + "\n")

print("Carregando modelo YOLOv8-Pose...")
print("(Primeira execucao: vai baixar ~6.5MB)\n")

try:
    model = YOLO('yolov8n-pose.pt')
    print("OK: Modelo carregado com sucesso!\n")
except Exception as e:
    print(f"ERRO ao carregar modelo: {e}")
    exit()

def extrair_features_do_video(video_path, model):
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6

    hip_y_series = []
    shoulder_y_series = []
    distances = []
    velocidades = []

    try:
        results = model(source=video_path, stream=True, conf=0.5, verbose=False)

        for r in results:
            if r.keypoints is not None and r.keypoints.xy is not None and r.keypoints.xy.numel() > 0:
                keypoints = r.keypoints.xy[0]

                try:
                    left_hip_y = float(keypoints[LEFT_HIP][1])
                    right_hip_y = float(keypoints[RIGHT_HIP][1])
                    left_shoulder_y = float(keypoints[LEFT_SHOULDER][1])
                    right_shoulder_y = float(keypoints[RIGHT_SHOULDER][1])

                    if (not np.isnan(left_hip_y) and not np.isnan(right_hip_y) and
                        not np.isnan(left_shoulder_y) and not np.isnan(right_shoulder_y) and
                        left_hip_y > 0 and right_hip_y > 0 and
                        left_shoulder_y > 0 and right_shoulder_y > 0):

                        hip_y = (left_hip_y + right_hip_y) / 2.0
                        shoulder_y = (left_shoulder_y + right_shoulder_y) / 2.0

                        hip_y_series.append(hip_y)
                        shoulder_y_series.append(shoulder_y)

                        dist = abs(shoulder_y - hip_y)
                        distances.append(dist)

                        if len(hip_y_series) > 1:
                            vel = hip_y_series[-1] - hip_y_series[-2]
                            velocidades.append(vel)

                except (ValueError, TypeError, RuntimeError):
                    continue

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

        return [float(max_vel), float(media_vel), float(diferenca_alt),
                float(std_alt), float(terminou_horizontal),
                float(distancia_final), float(distancia_min), float(taxa_var)]

    except Exception as e:
        return None

print("="*70)
print("FASE 3: ENCONTRAR E PROCESSAR VIDEOS")
print("="*70 + "\n")

def encontrar_videos(pasta_raiz, extensoes=['*.mp4', '*.avi', '*.mov', '*.mkv']):
    videos = []
    for extensao in extensoes:
        padrao = os.path.join(pasta_raiz, '**', extensao)
        videos.extend(glob.glob(padrao, recursive=True))
    return videos

path_fall = os.path.join(BASE_PATH, "Fall")
path_adl = os.path.join(BASE_PATH, "ADL")

print(f"Procurando videos em:")
print(f"  - {path_fall}")
print(f"  - {path_adl}\n")

videos_quedas = encontrar_videos(path_fall)
videos_normal = encontrar_videos(path_adl)

print(f"OK: Videos de queda encontrados: {len(videos_quedas)}")
print(f"OK: Videos normais encontrados: {len(videos_normal)}")
print(f"OK: Total: {len(videos_quedas) + len(videos_normal)} videos\n")

all_features = []
quedas_sucesso = 0
quedas_falha = 0

print("--- Processando videos de QUEDA ---")
for video_path in tqdm(videos_quedas, desc="Quedas"):
    features = extrair_features_do_video(video_path, model)
    if features:
        features.append(1)
        all_features.append(features)
        quedas_sucesso += 1
    else:
        quedas_falha += 1

print(f"OK: Sucessos: {quedas_sucesso}/{len(videos_quedas)}")
print(f"ERRO: Falhas: {quedas_falha}/{len(videos_quedas)}\n")

print("--- Processando videos NORMAIS ---")
normais_sucesso = 0
normais_falha = 0

for video_path in tqdm(videos_normal, desc="Normais"):
    features = extrair_features_do_video(video_path, model)
    if features:
        features.append(0)
        all_features.append(features)
        normais_sucesso += 1
    else:
        normais_falha += 1

print(f"OK: Sucessos: {normais_sucesso}/{len(videos_normal)}")
print(f"ERRO: Falhas: {normais_falha}/{len(videos_normal)}\n")

print("="*70)
print("FASE 4: CRIAR DATASET")
print("="*70 + "\n")

column_names = ['max_velocidade_vertical', 'media_velocidade_vertical',
                'diferenca_altura', 'std_altura', 'terminou_horizontal',
                'distancia_final_ombro_quadril', 'distancia_minima',
                'taxa_variacao', 'label']

df = pd.DataFrame(all_features, columns=column_names)
df = df.dropna()

print(f"OK: Total de amostras: {len(df)}")
print(f"OK: Total de features: {len(column_names) - 1}\n")

print("Distribuicao de classes:")
print(df['label'].value_counts())
print(f"\nPorcentagem:")
print(df['label'].value_counts(normalize=True) * 100)

csv_path = os.path.join(OUTPUT_DIR, 'features_de_queda_final.csv')
df.to_csv(csv_path, index=False)
print(f"\nOK: Dataset salvo em: {csv_path}\n")

print("="*70)
print("FASE 5: PREPARACAO DOS DADOS")
print("="*70 + "\n")

X = df.drop('label', axis=1).astype(np.float64)
y = df['label'].astype(np.int64)

print(f"Features: {X.shape}")
print(f"Labels: {y.shape}")
print(f"  - Classe 0 (Normal): {(y==0).sum()}")
print(f"  - Classe 1 (Queda): {(y==1).sum()}\n")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("OK: Dados normalizados\n")

print("="*70)
print("FASE 6: TREINAMENTO COM VALIDACAO CRUZADA (5-Fold)")
print("="*70 + "\n")

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'f1': make_scorer(f1_score, zero_division=0),
    'roc_auc': make_scorer(roc_auc_score, zero_division=0)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

modelos = {
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'KNN (k=7)': KNeighborsClassifier(n_neighbors=7),
    'Decision Tree (d=5)': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Decision Tree (d=10)': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest (d=5)': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1),
    'Random Forest (d=10)': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Random Forest (d=15)': RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1),
    'MLP (10,10)': MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.2),
    'MLP (64,32)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1500, random_state=42, early_stopping=True, validation_fraction=0.2),
    'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
}

resultados_cv = []

for nome, modelo in modelos.items():
    print(f"Treinando: {nome}...")

    try:
        cv_results = cross_validate(modelo, X_scaled, y, cv=cv, scoring=scoring, return_train_score=False)

        acc_mean = cv_results['test_accuracy'].mean()
        acc_std = cv_results['test_accuracy'].std()
        prec_mean = cv_results['test_precision'].mean()
        rec_mean = cv_results['test_recall'].mean()
        f1_mean = cv_results['test_f1'].mean()
        roc_mean = cv_results['test_roc_auc'].mean()

        print(f"  OK: Acc: {acc_mean:.4f}+/-{acc_std:.4f} | F1: {f1_mean:.4f}\n")

        resultados_cv.append({
            'Modelo': nome,
            'Accuracy_Mean': acc_mean,
            'Accuracy_Std': acc_std,
            'Precision_Mean': prec_mean,
            'Recall_Mean': rec_mean,
            'F1-Score_Mean': f1_mean,
            'ROC-AUC_Mean': roc_mean
        })
    except Exception as e:
        print(f"  ERRO: {e}\n")

print("="*70)
print("FASE 7: RESULTADOS E COMPARACAO")
print("="*70 + "\n")

df_resultados = pd.DataFrame(resultados_cv)
df_resultados = df_resultados.sort_values('F1-Score_Mean', ascending=False)

print("Tabela Comparativa de Modelos:")
print(df_resultados.to_string(index=False))
print()

csv_resultados = os.path.join(OUTPUT_DIR, 'resultados_cv_modelos.csv')
df_resultados.to_csv(csv_resultados, index=False)
print(f"OK: Resultados salvos em: {csv_resultados}\n")

print("="*70)
print("FASE 8: TREINAR MELHOR MODELO")
print("="*70 + "\n")

melhor_modelo_nome = df_resultados.iloc[0]['Modelo']
best_f1 = df_resultados.iloc[0]['F1-Score_Mean']

print(f"MELHOR MODELO: {melhor_modelo_nome}")
print(f"   F1-Score: {best_f1:.4f}\n")

best_model = modelos[melhor_modelo_nome]
best_model.fit(X_scaled, y)

print("OK: Modelo final treinado com todos os dados\n")

print("="*70)
print("FASE 9: SALVAR MODELOS")
print("="*70 + "\n")

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_scaled, y)

model_path = os.path.join(OUTPUT_DIR, 'detector_modelo_melhor.pkl')
scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
rf_path = os.path.join(OUTPUT_DIR, 'detector_modelo_rf.pkl')

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(rf_model, rf_path)

print(f"OK: Modelo melhor: {model_path}")
print(f"OK: Scaler: {scaler_path}")
print(f"OK: Random Forest: {rf_path}\n")

print("="*70)
print("RESUMO FINAL")
print("="*70 + "\n")

best_row = df_resultados.iloc[0]
print(f"MELHOR MODELO: {best_row['Modelo']}")
print(f"   Accuracy: {best_row['Accuracy_Mean']:.4f} (+/- {best_row['Accuracy_Std']:.4f})")
print(f"   Precision: {best_row['Precision_Mean']:.4f}")
print(f"   Recall: {best_row['Recall_Mean']:.4f}")
print(f"   F1-Score: {best_row['F1-Score_Mean']:.4f}")
print(f"   ROC-AUC: {best_row['ROC-AUC_Mean']:.4f}\n")

print("Dataset:")
print(f"   Total de amostras: {len(df)}")
print(f"   Quedas: {(y==1).sum()}")
print(f"   Normais: {(y==0).sum()}\n")

print(f"Arquivos salvos em: {OUTPUT_DIR}")
print("\nPROCESSAMENTO CONCLUIDO COM SUCESSO!\n")

print("="*70)
print("PROXIMOS PASSOS:")
print("="*70)
print("\n1. Copie os 3 arquivos .pkl da pasta './output/'")
print("2. Coloque na pasta do app_quedas.py")
print("3. Execute: streamlit run app_quedas.py\n")

