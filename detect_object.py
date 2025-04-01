import os
import cv2
import re
import json
import math
import pyttsx3
import mediapipe as mp
from ultralytics import YOLO

#  설정
FRAME_DIR = "frames"
TASK_FILENAME = "task_plan.json"
ARRIVE_THRESHOLD = 250      # ← 도착으로 인식할 거리
NEAR_THRESHOLD = 400        # ← 거의 도착한 경우
DISTANCE_DELTA = 30         # ← 거리 변화 감지 민감도

# 초기화
model = YOLO("yolov8n.pt")
tts = pyttsx3.init()
tts.setProperty("rate", 150)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def give_feedback(text):
    print(f"🗣 {text}")
    try:
        tts.say(text)
        tts.runAndWait()
    except KeyboardInterrupt:
        pass
    finally:
        tts.stop()

def detect_hand(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        h, w, _ = image.shape
        wrist = results.multi_hand_landmarks[0].landmark[0]
        return (int(wrist.x * w), int(wrist.y * h))
    return None

def find_target_position(image, target_english):
    results = model(image)
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)].lower()
            if target_english.lower() in label:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                return ((x1 + x2) // 2, (y1 + y2) // 2)
    return None

def sort_frames_by_number(files):
    def extract_number(f):
        match = re.search(r"frame_(\d+)", f)
        return int(match.group(1)) if match else float('inf')
    return sorted(files, key=extract_number)

def load_target_name():
    try:
        with open(TASK_FILENAME, "r", encoding="utf-8") as f:
            return json.load(f).get("target_english", None)
    except FileNotFoundError:
        print("❌ task_plan.json 파일이 없습니다.")
        return None

if __name__ == "__main__":
    target_english = load_target_name()
    if not target_english:
        print("❌ 타겟 물체 이름을 불러오지 못했습니다.")
        exit()
    print(f"📌 타겟 이름: {target_english}")

    frame_files = [f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg")]
    frame_files = sort_frames_by_number(frame_files)

    prev_distance = None
    last_known_target_pos = None

    for frame_file in frame_files:
        print(f"\n🖼 처리 중: {frame_file}")
        image_path = os.path.join(FRAME_DIR, frame_file)
        image = cv2.imread(image_path)

        hand_pos = detect_hand(image)
        if not hand_pos:
            give_feedback("손이 감지되지 않았습니다.")
            continue

        target_pos = find_target_position(image, target_english)
        if target_pos:
            last_known_target_pos = target_pos
        elif last_known_target_pos:
            print("⚠️ 타겟 감지 실패 - 이전 위치 사용")
            target_pos = last_known_target_pos
        else:
            give_feedback("타겟이 감지되지 않았습니다.")
            continue

        # 거리 및 방향 계산
        hx, hy = hand_pos
        ox, oy = target_pos
        dx = ox - hx
        dy = oy - hy
        distance = math.sqrt(dx ** 2 + dy ** 2)

        print(f"📏 손과 타겟 거리: {int(distance)}px, dx={dx}, dy={dy}")

        # 방향 판단: dx 우선
        if abs(dx) > abs(dy):
            direction_feedback = "오른쪽으로 이동하세요." if dx > 0 else "왼쪽으로 이동하세요."
        else:
            direction_feedback = "아래로 이동하세요." if dy > 0 else "위로 이동하세요."

        # 피드백 결정 (도착 / 접근 / 멀어짐)
        if distance < ARRIVE_THRESHOLD:
            feedback = "도착했습니다! 손을 뻗어 잡으세요."
        elif distance < NEAR_THRESHOLD:
            feedback = f"거의 도착했어요! {direction_feedback}"
        elif prev_distance is not None:
            delta = distance - prev_distance
            if delta < -DISTANCE_DELTA:
                feedback = f"손이 잘 접근하고 있어요. {direction_feedback}"
            elif delta > DISTANCE_DELTA:
                feedback = f"손이 멀어지고 있어요. {direction_feedback}"
            else:
                feedback = f"거리를 유지하고 있어요. {direction_feedback}"
        else:
            feedback = f"손 위치 확인 중입니다. {direction_feedback}"

        give_feedback(feedback)
        prev_distance = distance
