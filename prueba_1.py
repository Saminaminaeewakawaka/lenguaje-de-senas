import cv2
import mediapipe as mp

def distancia_euclidiana(p1, p2):
    # Calcula la distancia euclidiana entre dos puntos
    return cv2.norm(p1, p2)

def draw_bounding_box(image, hand_landmarks):
    image_height, image_width, _ = image.shape
    x_min, y_min = image_width, image_height
    x_max, y_max = 0, 0

    # Itera a través de los puntos de referencia para encontrar las coordenadas del bounding box
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)

        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y

    # Dibuja el bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convierte la imagen de BGR a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Proceso de la imagen
        image.flags.writeable = False
        results = hands.process(image)

        # Convierte de nuevo la imagen a BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibuja las marcas de la mano
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Coordenadas de las puntas de los dedos
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

                # Calcular las distancias y realizar acciones
                if distancia_euclidiana((thumb_tip.x, thumb_tip.y), (index_finger_tip.x, index_finger_tip.y)) < 0.05:
                    cv2.putText(image, 'A', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 2)

                if abs(index_finger_tip.x - thumb_tip.x) < 0.05 and abs(index_finger_tip.y - middle_finger_tip.y) < 0.05:
                    cv2.putText(image, 'B', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 2)

                if abs(index_finger_tip.x - middle_finger_tip.x) < 0.05 and abs(index_finger_tip.y - ring_finger_pip.y) < 0.05:
                    cv2.putText(image, 'C', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 2)

                # Dibuja el bounding box alrededor de la mano
                draw_bounding_box(image, hand_landmarks)

        # Muestra la imagen resultante
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()