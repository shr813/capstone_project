import os
import sys
import cv2
import json
import math
import time
import threading
import subprocess
import pyttsx3
import mediapipe as mp
from ultralytics import YOLO

# 설정
TASK_FILENAME = "task_plan.json"
ARRIVE_THRESHOLD = 50
NEAR_THRESHOLD = 150
DISTANCE_DELTA = 30
FEEDBACK_INTERVAL = 0.7
YOLO_INTERVAL = 1.5

# 초기화
model = YOLO("yolov8n.pt")
tts = pyttsx3.init()
tts.setProperty("rate", 200)
tts_lock = threading.Lock()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# 상태 변수
target_english = None
last_known_target_pos = None
prev_distance = None
hand_pos = None
frame_for_display = None
found_target_recently = False
initial_guided = False
miss_count = 0
frame_lock = threading.Lock()

# 피드백 중복 방지
last_feedback_text = None
last_feedback_time = 0
MIN_FEEDBACK_INTERVAL = 3.5


def give_feedback(text):
    global last_feedback_text, last_feedback_time

    now = time.time()
    if text == last_feedback_text and now - last_feedback_time < MIN_FEEDBACK_INTERVAL:
        return

    print(f"🗣 {text}")
    last_feedback_text = text
    last_feedback_time = now

    def speak():
        with tts_lock:
            try:
                tts.say(text)
                tts.runAndWait()
            except:
                pass

    threading.Thread(target=speak, daemon=True).start()


def detect_hand(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        h, w, _ = image.shape
        palm = [results.multi_hand_landmarks[0].landmark[i] for i in [0, 1, 5, 9, 13, 17]]
        x = int(sum(l.x for l in palm) / len(palm) * w)
        y = int(sum(l.y for l in palm) / len(palm) * h)
        return (x, y)
    return None


def find_target_position(image, label, min_conf=0.5):
    resized = cv2.resize(image, (320, 320))
    results = model(resized, verbose=False)
    scale_x = image.shape[1] / 320
    scale_y = image.shape[0] / 320

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls = model.names[int(box.cls)].lower()
            if label.lower() in cls and conf > min_conf:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2 * scale_x)
                cy = int((y1 + y2) / 2 * scale_y)
                return (cx, cy)
    return None



def get_initial_direction_comment(pos, frame_size):
    x, y = pos
    w, h = frame_size
    x_rel, y_rel = x / w, y / h

    dir_x = "왼쪽" if x_rel < 0.3 else "오른쪽" if x_rel > 0.7 else "가운데"
    dir_y = "위" if y_rel < 0.3 else "아래" if y_rel > 0.7 else "가운데"

    if dir_x == "가운데" and dir_y == "가운데":
        return "타겟이 정중앙에 있습니다."
    else:
        return f"타겟이 {dir_x} {dir_y}에 있어요."


def load_target_name():
    try:
        with open(TASK_FILENAME, "r", encoding="utf-8") as f:
            return json.load(f).get("target_english", None)
    except:
        return None


def feedback_loop():
    global last_known_target_pos, prev_distance
    target_missing_warned = False
    last_hand_feedback_time = 0
    last_distance_feedback_time = 0  # NEW
    HAND_FEEDBACK_COOLDOWN = 10
    DISTANCE_FEEDBACK_COOLDOWN = 4.5  # NEW

    while True:
        time.sleep(FEEDBACK_INTERVAL)

        with frame_lock:
            hand = hand_pos
            target = last_known_target_pos

        if not target:
            if not target_missing_warned:
                give_feedback("타겟이 감지되지 않았습니다. 주변을 둘러봐 주세요.")
                target_missing_warned = True
            continue
        else:
            target_missing_warned = False

        if not hand:
            now = time.time()
            if now - last_hand_feedback_time > HAND_FEEDBACK_COOLDOWN:
                give_feedback("손이 감지되지 않았습니다.")
                last_hand_feedback_time = now
            continue

        hx, hy = hand
        tx, ty = target
        dx, dy = tx - hx, ty - hy
        distance = math.sqrt(dx**2 + dy**2)

        direction = "오른쪽" if dx > 0 else "왼쪽" if abs(dx) > abs(dy) else "아래" if dy > 0 else "위"
        msg = None
        now = time.time()

        if distance < ARRIVE_THRESHOLD:
            msg = "도착했습니다! 손을 뻗어 잡으세요."
        elif distance < NEAR_THRESHOLD:
            msg = f"거의 도착했어요! {direction}으로 이동하세요."
        elif prev_distance is not None:
            delta = distance - prev_distance
            if delta < -DISTANCE_DELTA:
                msg = f"손이 잘 접근하고 있어요. {direction}으로 이동하세요."
            elif delta > DISTANCE_DELTA:
                msg = f"손이 멀어지고 있어요. {direction}으로 이동하세요."
            else:
                msg = f"거리를 유지하고 있어요. {direction}으로 이동하세요."
        else:
            msg = f"손 위치 확인 중입니다. {direction}으로 이동하세요."

        prev_distance = distance

        if msg and now - last_distance_feedback_time > DISTANCE_FEEDBACK_COOLDOWN:
            give_feedback(msg)
            last_distance_feedback_time = now



def yolo_loop():
    global last_known_target_pos, found_target_recently, initial_guided, miss_count, target_english

    while True:
        time.sleep(YOLO_INTERVAL)

        with frame_lock:
            if frame_for_display is not None:
                pos = find_target_position(frame_for_display.copy(), target_english)
                if pos:
                    last_known_target_pos = pos
                    found_target_recently = True
                    miss_count = 0
                    if not initial_guided:
                        comment = get_initial_direction_comment(
                            pos,
                            (frame_for_display.shape[1], frame_for_display.shape[0])
                        )
                        give_feedback(comment)
                        initial_guided = True
                else:
                    found_target_recently = False
                    miss_count += 1

                    if miss_count == 3:
                        give_feedback("타겟이 보이지 않아요. 카메라를 좌우로 천천히 움직여 주세요.")
                    elif miss_count == 6:
                        give_feedback("여전히 타겟이 없습니다. 위아래로도 비춰봐 주세요.")
                    elif miss_count == 9:
                        give_feedback("조명이 어두울 수 있어요. 조명을 켜거나 물러나 보세요.")
                    elif miss_count == 12:
                        give_feedback("타겟을 찾지 못했어요. 각도를 바꿔보세요.")
                    elif miss_count >= 15:
                        give_feedback("타겟 인식에 어려움이 있습니다. 음성 명령을 다시 말해주세요.")
                        subprocess.run([sys.executable, "voice_command.py"])
                        subprocess.run([sys.executable, "task_planner.py"])
                        os.execv(sys.executable, [sys.executable] + sys.argv)


if __name__ == "__main__":
    target_english = load_target_name()
    if not target_english:
        print("❌ 타겟 이름 불러오기 실패")
        exit()

    print(f"📌 타겟 이름: {target_english}")

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    threading.Thread(target=feedback_loop, daemon=True).start()
    threading.Thread(target=yolo_loop, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with frame_lock:
            frame_for_display = frame.copy()

        with frame_lock:
            if found_target_recently:
                hand_pos = detect_hand(frame)
            else:
                hand_pos = None

        if hand_pos:
            cv2.circle(frame_for_display, hand_pos, 10, (0, 255, 0), -1)
        if last_known_target_pos:
            cv2.circle(frame_for_display, last_known_target_pos, 10, (0, 0, 255), -1)

        cv2.imshow("웹캠 미리보기", frame_for_display)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
