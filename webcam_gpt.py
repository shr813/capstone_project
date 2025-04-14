import os
import sys
import cv2
import json
import math
import time
import threading
import pyttsx3
import mediapipe as mp
from ultralytics import YOLO
from datetime import datetime
import openai
import base64

# 설정
TASK_FILENAME = "task_plan_destination.json"
ARRIVE_THRESHOLD = 50
NEAR_THRESHOLD = 150
DISTANCE_DELTA = 30
FEEDBACK_INTERVAL = 1.5
YOLO_INTERVAL = 1.5
TTS_RATE = 200
MIN_FEEDBACK_INTERVAL = 4
GRAB_HOLD_DURATION = 3.0  # 3초 이상 가까이 있으면 잡은 것으로 판단
HAND_FEEDBACK_INTERVAL = 9.0
api_key = os.getenv("OPENAI_API_KEY")

# 초기화
model = YOLO("yolov8n.pt")
tts = pyttsx3.init()
tts.setProperty("rate", TTS_RATE)
tts_lock = threading.Lock()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 전역 상태 변수
target, destination = None, None
target_pos, destination_pos = None, None
last_seen_target_pos, last_seen_destination_pos = None, None
hand_pos, frame_for_display = None, None
frame_lock = threading.Lock()
last_hand_feedback_time = 0
near_intro_done = False

# 피드백 상태
last_feedback_time, last_feedback_text = 0, None
current_tts_thread = None
step = "find_target"
prev_distance = None
target_intro_done, destination_intro_done = False, False
target_grabbed = False
last_close_to_target_time = None
initial_target_direction_given = False

# ------------------ 기능 함수 ------------------

def speak_feedback(text):
    global last_feedback_time, last_feedback_text, current_tts_thread
    now = time.time()
    if text == last_feedback_text and now - last_feedback_time < MIN_FEEDBACK_INTERVAL:
        return
    last_feedback_text = text
    last_feedback_time = now
    print("🗣", text)

    def tts_job():
        with tts_lock:
            try:
                tts.stop()
                tts.say(text)
                tts.runAndWait()
            except:
                pass

    if current_tts_thread and current_tts_thread.is_alive():
        pass
    current_tts_thread = threading.Thread(target=tts_job, daemon=True)
    current_tts_thread.start()

def speak_hand_feedback(text):
    global last_hand_feedback_time, current_tts_thread
    now = time.time()
    if now - last_hand_feedback_time < HAND_FEEDBACK_INTERVAL:
        return
    last_hand_feedback_time = now
    print("🗣", text)

    def tts_job():
        with tts_lock:
            try:
                tts.stop()
                tts.say(text)
                tts.runAndWait()
            except:
                pass

    if current_tts_thread and current_tts_thread.is_alive():
        pass
    current_tts_thread = threading.Thread(target=tts_job, daemon=True)
    current_tts_thread.start()


def detect_hand(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        h, w, _ = image.shape
        palm = [results.multi_hand_landmarks[0].landmark[i] for i in [0, 1, 5, 9, 13, 17]]
        x = int(sum(p.x for p in palm) / len(palm) * w)
        y = int(sum(p.y for p in palm) / len(palm) * h)
        return (x, y)
    return None

def find_object_position(image, label, min_conf=0.6):
    resized = cv2.resize(image, (320, 320))
    results = model(resized, verbose=False)
    scale_x = image.shape[1] / 320
    scale_y = image.shape[0] / 320
    for result in results:
        for box in result.boxes:
            cls = model.names[int(box.cls)].lower()
            conf = float(box.conf[0])
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

def ask_gpt_if_grabbed(image, target):
    question = (
        f"이미지를 보고 사람이 '{target}'을 손으로 잡고 있는지 판단해줘. "
        f"손이 '{target}' 위에 완전히 올라가 있고 일부 감싸고 있거나, 확실히 잡고 있다면 다음과 같이 응답해:\n"
        '{ "grabbed": "true" }\n\n'
        "손이 근처에 있지만 명확하게 잡지 않았으면:\n"
        '{ "grabbed": "false" }\n\n'
        f"중요: 이미지가 불분명하더라도 손이 '{target}'을 감싸고 있거나, '{target}' 대부분을 덮고 있다면 true로 판단해줘."
    )

    _, buffer = cv2.imencode('.jpg', image)
    img_bytes = buffer.tobytes()
    base64_img = base64.b64encode(img_bytes).decode("utf-8")

    print("🧠 GPT 판단 요청 중...")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                        }
                    ]
                }
            ],
            max_tokens=300,
        )

        result = response['choices'][0]['message']['content']
        print("🧠 GPT 응답:", result)

        # JSON 응답 파싱 시도
        try:
            parsed = json.loads(result)
            return str(parsed.get("grabbed", "")).strip().lower() == "true"
        except json.JSONDecodeError:
            print("⚠️ JSON 파싱 실패. 응답 내용 확인:", result)
            return False

    except Exception as e:
        print("[GPT 오류]", e)
        return False


# ------------------ 피드백 루프 ------------------

def feedback_loop():
    global step, target_intro_done, destination_intro_done
    global prev_distance, target_grabbed, last_close_to_target_time
    global initial_target_direction_given, last_hand_feedback_time
    global near_intro_done

    horizontal_aligned = False

    while True:
        time.sleep(FEEDBACK_INTERVAL)

        with frame_lock:
            hand = hand_pos
            target = target_pos or last_seen_target_pos
            dest = destination_pos or last_seen_destination_pos
            frame = frame_for_display.copy() if frame_for_display is not None else None

        if step == "find_target":
            if frame is None:
                continue

            if not target and not destination:
                speak_feedback("타겟과 목적지가 감지되지 않았습니다.")
                continue
            elif not target:
                speak_feedback("타겟이 감지되지 않았습니다.")
                continue
            elif not destination:
                speak_feedback("목적지가 감지되지 않았습니다.")
                continue

            if not initial_target_direction_given and target:
                msg = get_initial_direction_comment(target, (frame.shape[1], frame.shape[0]))
                speak_feedback(msg)
                initial_target_direction_given = True

            if not hand:
                now = time.time()
                if now - last_hand_feedback_time > HAND_FEEDBACK_INTERVAL:
                    speak_hand_feedback("손이 감지되지 않았습니다.")
                    last_hand_feedback_time = now
                continue

            hx, hy = hand
            tx, ty = target
            dx, dy = tx - hx, ty - hy
            distance = math.sqrt(dx ** 2 + dy ** 2)

            if distance < ARRIVE_THRESHOLD:
                if not last_close_to_target_time:
                    last_close_to_target_time = datetime.now()
                elif (datetime.now() - last_close_to_target_time).total_seconds() >= GRAB_HOLD_DURATION:
                    speak_feedback("도착했습니다. 손을 뻗어 잡으세요.")

                    # 3초 대기 후 프레임 캡처
                    time.sleep(3.0)

                    if frame is not None:
                        speak_feedback("손이 물체를 잡았는지 확인 중입니다.")
                        is_grabbed = ask_gpt_if_grabbed(frame, target)

                        if isinstance(is_grabbed, str):
                            is_grabbed = is_grabbed.strip().lower() == "true"

                        if is_grabbed:
                            speak_feedback("손이 물체를 잡은 것이 확인되었습니다.")
                            target_grabbed = True
                            step = "move_to_destination"
                            destination_intro_done = False
                            target_intro_done = False
                            time.sleep(1.5)
                            continue
                        else:
                            speak_feedback("아직 잡지 않은 것 같아요. 더 가까이 접근하세요.")
            else:
                last_close_to_target_time = None

            direction = (
                "오른쪽" if dx > 0 else "왼쪽"
            ) if abs(dx) > abs(dy) else (
                "아래" if dy > 0 else "위"
            )

            if distance < NEAR_THRESHOLD:
                if not near_intro_done:
                    speak_feedback(f"거의 도착했어요. {direction}으로 이동하세요.")
                    near_intro_done = True
                else:
                    speak_feedback(f"{direction}으로 이동하세요.")
            else:
                near_intro_done = False
                if not target_intro_done:
                    speak_feedback(f"타겟에 접근 중입니다. {direction}으로 이동하세요.")
                    target_intro_done = True
                else:
                    speak_feedback(f"{direction}으로 이동하세요.")
            prev_distance = distance

        elif step == "move_to_destination":
            if frame is None:
                continue

            if not dest:
                speak_feedback("목적지가 감지되지 않았습니다.")
                continue

            if not hand:
                now = time.time()
                if now - last_hand_feedback_time > HAND_FEEDBACK_INTERVAL:
                    speak_hand_feedback("손이 감지되지 않았습니다.")
                    last_hand_feedback_time = now
                continue

            hx, hy = hand
            dx, dy = dest[0] - hx, dest[1] - hy
            distance = math.sqrt(dx ** 2 + dy ** 2)

            if distance < ARRIVE_THRESHOLD:
                speak_feedback("목적지에 도착했습니다. 내려놓으세요.")
                step = "done"
                continue

            direction = (
                "오른쪽" if dx > 0 else "왼쪽"
            ) if abs(dx) > abs(dy) else (
                "아래" if dy > 0 else "위"
            )

            if not destination_intro_done:
                speak_feedback("목적지로 이동 중입니다.")
                destination_intro_done = True
            else:
                speak_feedback(f"{direction}으로 이동하세요.")

# ------------------ 객체 위치 갱신 루프 ------------------

def yolo_loop():
    global target_pos, destination_pos, last_seen_target_pos, last_seen_destination_pos
    while True:
        time.sleep(YOLO_INTERVAL)
        with frame_lock:
            frame = frame_for_display.copy() if frame_for_display is not None else None
        if frame is None:
            continue

        if target:
            pos = find_object_position(frame, target)
            if pos:
                target_pos = pos
                last_seen_target_pos = pos

        if destination:
            pos = find_object_position(frame, destination)
            if pos:
                destination_pos = pos
                last_seen_destination_pos = pos

# ------------------ 메인 ------------------

def load_target_info():
    try:
        with open(TASK_FILENAME, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("target"), data.get("destination")
    except:
        return None, None

if __name__ == "__main__":
    target, destination = load_target_info()
    if not target or not destination:
        print("❌ 타겟 또는 목적지 정보를 불러오지 못했습니다.")
        exit()

    print(f"📌 타겟: {target}, 목적지: {destination}")

    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
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
            if target_pos or destination_pos or last_seen_target_pos or last_seen_destination_pos:
                hand_pos = detect_hand(frame)
            else:
                hand_pos = None

        if hand_pos:
            cv2.circle(frame_for_display, hand_pos, 10, (0, 255, 0), -1)
        if target_pos or last_seen_target_pos:
            cv2.circle(frame_for_display, target_pos or last_seen_target_pos, 10, (0, 0, 255), -1)
        if destination_pos or last_seen_destination_pos:
            cv2.circle(frame_for_display, destination_pos or last_seen_destination_pos, 10, (255, 0, 0), -1)

        cv2.imshow("웹캠 미리보기", frame_for_display)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
