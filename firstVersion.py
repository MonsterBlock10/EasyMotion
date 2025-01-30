###################################################################################
# Librerías
import time
import cv2
import mediapipe as mp
import ctypes
import numpy as np


###################################################################################
#Funciones de WIN32 API

def move_mouse(x, y):
    ctypes.windll.user32.SetCursorPos(x, y)


def get_screen_size():
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    return screen_width, screen_height

def get_face_coordinates(face_landmarks, iw, ih, points):
    return [(int(face_landmarks.landmark[p].x * iw), int(face_landmarks.landmark[p].y * ih)) for p in points]

def check_main_click(check_distancia):
    global click, click_start_time, last_click_time

    if check_distancia > 0.74:
        current_time = time.time()

        if current_time - last_click_time > 1.5 and not click:
            print("CLICK!")
            left_click_down()
            last_click_time = current_time
            click_start_time = current_time
            click = True

        elif click and current_time - click_start_time > 2:
            print("HOLD CLICK!")
    else:
        if click:
            print("RELEASE CLICK!")
            left_click_up()
        click = False
        click_start_time = None

def check_eyebrow_right_click_via_nose(face_landmarks, iw, ih, distancia_orejas, left_eyebrow, right_eyebrow):
    global last_click_time

    nose = get_face_coordinates(face_landmarks, iw, ih, [1])[0]  # Punto de la nariz

    midpoint_eyebrows = (
    (left_eyebrow[0] + right_eyebrow[0]) // 2,
    (left_eyebrow[1] + right_eyebrow[1]) // 2
    )

    distance_to_nose = np.linalg.norm(np.array(midpoint_eyebrows) - np.array(nose)) / distancia_orejas


    current_time = time.time()
    if distance_to_nose > 0.41: 
        if current_time - last_click_time > 1.5:
            print("Right Click!")  
            right_click_down()
            right_click_up()
            last_click_time = current_time

    cv2.circle(frame, midpoint_eyebrows, 5, (0, 255, 255), -1) 
    cv2.line(frame, midpoint_eyebrows, nose, (0, 255, 0), 2)  
    cv2.putText(
        frame,
        f"Dist: {str(distance_to_nose)}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


def left_click_down():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)  

def left_click_up():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)  

def right_click_down():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)  

def right_click_up():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)  

###################################################################################
# Constantes y Variables

screen_width, screen_height = get_screen_size()
last_click_time = 0
click = False
click_start_time = None
initialized = False

CENTER_JAW = 152
NOSE_CENTER = 1

LEFT_EAR = 127
RIGHT_EAR = 356

RIGHT_EYEBROW = 65
LEFT_EYEBROW = 295


MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

DEBUG_MODE = True

###################################################################################
# Configuración de Mediapipe para detección facial

mp_face_mesh = mp.solutions.face_mesh  
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,  
    min_detection_confidence=0.5 
)


mp_drawing = mp.solutions.drawing_utils  

cap = cv2.VideoCapture(0)


#################################################################################
# Main process

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:

                ih, iw, _ = frame.shape

                points = get_face_coordinates(face_landmarks, iw, ih, [NOSE_CENTER, CENTER_JAW, LEFT_EAR, RIGHT_EAR, LEFT_EYEBROW, RIGHT_EYEBROW])
                (x_nose, y_nose), (x_jaw, y_jaw), (x_l_ear, y_l_ear), (x_r_ear, y_r_ear) , (x_l_eyebrow, y_l_eyebrow), (x_r_eyebrow, y_r_eyebrow) = points


                move_mouse(screen_width - x_nose, y_nose)

                distancia = float(np.linalg.norm(np.array([x_jaw, y_jaw]) - np.array([x_nose, y_nose])))
                distancia_orejas = float(np.linalg.norm(np.array([x_r_ear, y_r_ear]) - np.array([x_l_ear, y_l_ear])))


                if DEBUG_MODE:
                    cv2.circle(frame, (x_nose, y_nose), 5, (0, 0, 255), -1)

                    cv2.circle(frame, (x_nose, y_nose), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (x_jaw, y_jaw), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (x_l_ear, y_l_ear), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (x_r_ear, y_r_ear), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (x_l_eyebrow, y_l_eyebrow), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (x_r_eyebrow, y_r_eyebrow), 5, (0, 255, 0), -1)

                    cv2.line(frame, (x_l_eyebrow, y_l_eyebrow), (x_r_eyebrow, y_r_eyebrow), (0,0,255), 1)
                    cv2.line(frame, (x_nose, y_nose), (x_jaw, y_jaw), (0,0,255), 1)
                    cv2.line(frame, (x_r_ear, y_r_ear), (x_l_ear, y_l_ear), (0,0,255), 1)


                check_distancia = distancia/distancia_orejas

                cv2.putText(frame, str(check_distancia), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                check_main_click(check_distancia)

                check_eyebrow_right_click_via_nose(face_landmarks, iw, ih, distancia_orejas, (x_l_eyebrow, y_l_eyebrow), (x_r_eyebrow, y_r_eyebrow))

                cv2.putText(
                    frame,
                    f"Nose: ({x_nose}, {x_nose})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("EasyMotion", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
