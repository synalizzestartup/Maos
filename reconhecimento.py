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

def draw_rounded_rectangle(img, pt1, pt2, color, radius):
    """
    Desenha um retângulo com cantos arredondados.
    pt1: Canto superior esquerdo
    pt2: Canto inferior direito
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # Desenha os 4 cantos (círculos preenchidos)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)

    # Desenha os retângulos de preenchimento
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)

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

            # --- Configurações do Texto ---
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            font_scale = 1.5
            font_thickness = 2
            text_color = (0, 0, 0)       # Preto
            bg_color = (255, 255, 255)   # Branco
            corner_radius = 15           # Raio dos cantos
            padding = 10

            # Define a posição do texto com base em qual mão é
            if handedness == 'Left':
                text = f"Esquerda: {predicted_label}"
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                # Posição do retângulo
                rect_start = (10, 30)
                rect_end = (rect_start[0] + text_w + padding, rect_start[1] + text_h + padding)
                # Posição do texto (canto inferior esquerdo)
                text_org = (rect_start[0] + padding // 2, rect_start[1] + text_h + padding // 2)
            else: # Direita
                text = f"Direita: {predicted_label}"
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                rect_start = (frame.shape[1] - text_w - 20 - padding, 30)
                rect_end = (frame.shape[1] - 10, rect_start[1] + text_h + padding)
                text_org = (rect_start[0] + padding // 2, rect_start[1] + text_h + padding // 2)

            # Desenha o fundo e depois o texto
            draw_rounded_rectangle(frame, rect_start, rect_end, bg_color, corner_radius)
            cv2.putText(frame, text, text_org, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(10) & 0xFF == 27: # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()