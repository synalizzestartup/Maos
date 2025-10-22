import cv2
import mediapipe as mp
import pickle
import numpy as np

# --- Configurações e Carregamento do Modelo ---
MODEL_PATH = "modelo_gestos.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
NUM_LANDMARKS = 21

print("Carregando modelo...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
except FileNotFoundError:
    print("Erro: Arquivos de modelo não encontrados. Execute 'treinamento.py' primeiro.")
    exit()

print("Modelo carregado com sucesso!")

# --- Inicialização do MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def extrair_features(hand_landmarks):
    """
    Extrai as features da mão para a predição, da mesma forma que no treinamento.
    """
    features = []
    pulso_x = hand_landmarks.landmark[0].x
    pulso_y = hand_landmarks.landmark[0].y

    for i in range(NUM_LANDMARKS):
        landmark_x = hand_landmarks.landmark[i].x
        landmark_y = hand_landmarks.landmark[i].y
        features.append(landmark_x - pulso_x)
        features.append(landmark_y - pulso_y)

    return features

# --- Loop Principal ---
cap = cv2.VideoCapture(0)

# --- Configuração da Janela ---
WINDOW_NAME = "Reconhecimento de Gestos"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inverte a imagem horizontalmente para um efeito de espelho
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
                # Itera sobre cada mão detectada
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Desenha os landmarks na mão
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Identifica se a mão é esquerda ou direita
            # A imagem é espelhada, então a mão 'Right' do MediaPipe é a sua direita, que aparece à esquerda na tela.
            handedness = results.multi_handedness[i].classification[0].label


            # Extrai as features e faz a predição
            features = extrair_features(hand_landmarks)
            prediction_numeric = model.predict([features])[0]
            predicted_label = le.inverse_transform([prediction_numeric])[0]

            # Define a posição do texto com base em qual mão é
            if handedness == 'Right':
                text = f"Direita: {predicted_label}"
                org = (10, 50)
                color = (0, 0, 0) # Verde
            else: # Left
                text = f"Esquerda: {predicted_label}"
                org = (frame.shape[1] - 250, 50) # Posição no canto superior direito
                color = (0, 0, 0) # Azul

            # Escreve o resultado na tela
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)


    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(10) & 0xFF == 27: # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()