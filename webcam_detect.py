import os
import cv2
import json
import math
import time
import threading
import pyttsx3
import mediapipe as mp
from ultralytics import YOLO

# 설정
TASK_FILENAME = "task_plan.json"
ARRIVE_THRESHOLD = 50      # 도착으로 인식할 거리 (정밀하게)
NEAR_THRESHOLD = 150       # 거의 도착한 경우
DISTANCE_DELTA = 30        # 거리 변화 감지 민감도
FEEDBACK_INTERVAL = 0.7
YOLO_INTERVAL = 1.5

# 초기화
model = YOLO("yolov8n.pt")
tts = pyttsx3.init()
tts.setProperty("rate", 200)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# 상태 저장용 변수
target_english = None
last_known_target_pos = None
prev_distance = None
last_feedback_time = 0
hand_pos = None
frame_for_display = None

frame_lock = threading.Lock()
feedback_lock = threading.Lock()

def give_feedback(text):
    print(f" {text}")
    try:
        tts.say(text)
        tts.runAndWait()
    except:
        pass
    finally:
        tts.stop()

def detect_hand(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        h, w, _ = image.shape
        landmarks = results.multi_hand_landmarks[0].landmark
        palm_indices = [0, 1, 5, 9, 13, 17]
        avg_x = sum([landmarks[i].x for i in palm_indices]) / len(palm_indices)
        avg_y = sum([landmarks[i].y for i in palm_indices]) / len(palm_indices)
        return (int(avg_x * w), int(avg_y * h))
    return None

def find_target_position(image, label):
    resized = cv2.resize(image, (320, 320))
    results = model(resized, verbose=False)
    scale_x = image.shape[1] / 320
    scale_y = image.shape[0] / 320

    for result in results:
        for box in result.boxes:
            cls_name = model.names[int(box.cls)].lower()
            if label.lower() in cls_name:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2 * scale_x)
                cy = int((y1 + y2) / 2 * scale_y)
                return (cx, cy)
    return None

def load_target_name():
    try:
        with open(TASK_FILENAME, "r", encoding="utf-8") as f:
            return json.load(f).get("target_english", None)
    except:
        return None

def feedback_loop():
    global last_known_target_pos, prev_distance, last_feedback_time

    while True:
        time.sleep(FEEDBACK_INTERVAL)

        with frame_lock:
            current_hand = hand_pos
            current_target = last_known_target_pos

        if not current_hand:
            give_feedback("손이 감지되지 않았습니다.")
            continue

        if not current_target:
            give_feedback("타겟이 감지되지 않았습니다.")
            continue

        hx, hy = current_hand
        ox, oy = current_target
        dx, dy = ox - hx, oy - hy
        distance = math.sqrt(dx**2 + dy**2)

        if abs(dx) > abs(dy):
            direction = "오른쪽으로 이동하세요." if dx > 0 else "왼쪽으로 이동하세요."
        else:
            direction = "아래로 이동하세요." if dy > 0 else "위로 이동하세요."

        if distance < ARRIVE_THRESHOLD:
            feedback = "도착했습니다! 손을 뻗어 잡으세요."
        elif distance < NEAR_THRESHOLD:
            feedback = f"거의 도착했어요! {direction}"
        elif prev_distance is not None:
            delta = distance - prev_distance
            if delta < -DISTANCE_DELTA:
                feedback = f"손이 잘 접근하고 있어요. {direction}"
            elif delta > DISTANCE_DELTA:
                feedback = f"손이 멀어지고 있어요. {direction}"
            else:
                feedback = f"거리를 유지하고 있어요. {direction}"
        else:
            feedback = f"손 위치 확인 중입니다. {direction}"

        prev_distance = distance
        give_feedback(feedback)

def yolo_loop():
    global last_known_target_pos

    while True:
        time.sleep(YOLO_INTERVAL)

        with frame_lock:
            if frame_for_display is not None:
                pos = find_target_position(frame_for_display, target_english)
                if pos:
                    last_known_target_pos = pos

if __name__ == "__main__":
    target_english= load_target_name()

    if not target_english:
        print("❌ 타겟 이름 불러오기 실패")
        exit()

    print(f" 타겟 이름: {target_english}")

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    threading.Thread(target=feedback_loop, daemon=True).start()
    threading.Thread(target=yolo_loop, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hand = detect_hand(frame)

        with frame_lock:
            frame_for_display = frame.copy()
            hand_pos = hand

        if hand:
            cv2.circle(frame_for_display, hand, 10, (0, 255, 0), -1)
        if last_known_target_pos:
            cv2.circle(frame_for_display, last_known_target_pos, 10, (0, 0, 255), -1)

        cv2.imshow("웹캠 미리보기", frame_for_display)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
