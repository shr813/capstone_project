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
from ultralytics import YOLO as _YOLO
from datetime import datetime
import openai
import base64
from voice_command import get_yes_no_response, get_command
from task_planner_dest import load_command, extract_target_with_gpt

# 설정
TASK_FILENAME = "task_plan_destination.json"
ARRIVE_THRESHOLD = 50
NEAR_THRESHOLD = 150
DISTANCE_DELTA = 30
FEEDBACK_INTERVAL = 2.0
YOLO_INTERVAL = 1.5
TTS_RATE = 220
MIN_FEEDBACK_INTERVAL = 5.0
HAND_FEEDBACK_INTERVAL = 7.0
api_key = os.getenv("OPENAI_API_KEY")

# 초기화
model = YOLO("yolov8n.pt")
tts = pyttsx3.init()
tts.setProperty("rate", TTS_RATE)
tts_lock = threading.Lock()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 전역 변수
target, destination = None, None
target_pos, destination_pos = None, None
last_seen_target_pos, last_seen_destination_pos = None, None
hand_pos, frame_for_display = None, None
frame_lock = threading.Lock()
last_hand_feedback_time = 0
near_intro_done = False
miss_count = 0  # 물체 미검출 시 카운트

# 피드백 상태
last_feedback_time, last_feedback_text = 0, None
current_tts_thread = None
step = "find_target"
prev_distance = None
target_intro_done, destination_intro_done = False, False
target_grabbed = False
last_close_to_target_time = None
initial_target_direction_given = False



def auto_panorama_scan(scan_duration=7.0, capture_interval=0.6, cam_index=2):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    frames = []
    start_t = time.time()
    next_capture = start_t

    print(f"▶ 파노라마 스캔 시작: {scan_duration}초 동안 주변을 천천히 돌려주세요...")
    tts.say("파노라마 스캔을 시작하겠습니다. 오른쪽으로 천천히 돌아봐주세요.")
    tts.runAndWait()
    while time.time() - start_t < scan_duration:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()
        if now >= next_capture:
            frames.append(frame.copy())
            next_capture += capture_interval
        cv2.waitKey(1)
    cap.release()
    # stitch
    stitcher = cv2.Stitcher_create()
    status, pano = stitcher.stitch(frames)
    if status != cv2.Stitcher_OK:
        print("⚠️ 파노라마 생성 실패:", status)
        return None
    print("✅ 파노라마 생성 완료")
    return pano

def detect_on_panorama(pano_img, target_label, dest_label=None, return_labels=False):
    model = _YOLO("yolov8n.pt")
    results = model(pano_img)
    tgt, dst = None, None
    labels = []
    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls)].lower()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = int((x1 + x2)/2)
            cy = int((y1 + y2)/2)
            if target_label.lower() in cls:
                tgt = (cx, cy)
            if dest_label and dest_label.lower() in cls:
                dst = (cx, cy)
    if return_labels:
        return tgt, dst, labels
    return tgt, dst


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
                cy = int(y2 * scale_y)  # 하단 부분
                return (cx, cy)
    return None

def get_initial_direction_comment(pos, frame_size):
    if pos is None:
        return "타겟 위치를 찾을 수 없습니다."

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
        f"이미지를 보고 사람이 '{target}'를 손으로 확실히 잡고 있는지 판단해줘. "
        "다음과 같은 경우에만 다음 형식으로 응답해:\n"
        '{ "grabbed": "true" }\n\n'
        "- 손이 물체를 감싸고 있거나\n"
        "- 손가락이 물체의 양쪽을 잡고 있으며\n"
        "- 손이 물체 위에 놓여 확실히 고정된 상태라면 true.\n\n"
        "손이 근처에 있거나 손가락이 물체 위에 없으면 false로 판단해줘:\n"
        '{ "grabbed": "false" }\n\n'
        "반드시 위 JSON 형식 중 하나로만 대답해줘."
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
            return str(parsed.get_yes_no_response("grabbed", "")).strip().lower() == "true"
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
    #global near_intro_done, miss_count

    while True:
        time.sleep(FEEDBACK_INTERVAL)

        with frame_lock:
            hand = hand_pos
            target = target_pos or last_seen_target_pos
            dest = destination_pos or last_seen_destination_pos
            frame = frame_for_display.copy() if frame_for_display is not None else None

        # if step == "find_target":
        #
        #     if target is None:
        #         miss_count += 1
        #         if miss_count == 3:
        #             speak_feedback("타겟이 보이지 않아요. 오른쪽으로 천천히 움직여 주세요.")
        #         elif miss_count == 6:
        #             speak_feedback("아직 못 찾았어요. 왼쪽으로도 비춰봐 주세요.")
        #         elif miss_count == 9:
        #             speak_feedback("안 보입니다. 위쪽도 봐주세요.")
        #         elif miss_count == 12:
        #             speak_feedback("아래쪽도 봐주세요.")
        #         elif miss_count >= 15:
        #             speak_feedback("가까운 곳에는 찾으시는 물건이 없는 것 같아요.")
        #             if current_tts_thread:
        #                 current_tts_thread.join()
        #
        #             if get_yes_no_response():
        #                 # ▶ 사용자가 “예”라고 하면, 새 명령어 받기
        #                 speak_feedback("찾으실 물건을 말씀해주세요.")
        #                 get_command()  # voice_command.py의 get_command()로 command.txt 업데이트
        #
        #                 # ▶ 새로 저장된 명령 불러와서 GPT로 target/destination 추출
        #                 cmd_text = load_command()
        #                 info = extract_target_with_gpt(cmd_text)
        #                 if info:
        #                     target = info.get("target")
        #                     destination = info.get("destination")  # 단순 탐색일 땐 None일 수도 있음
        #                     miss_count = 0
        #                     # 필요하면 화면에도 출력하거나 로그 남기기
        #                     print(f"새 target: {target}, destination: {destination}")
        #                     continue
        #                 else:
        #                     speak_feedback("목적지 정보를 추출하지 못했습니다. 프로그램을 종료합니다.")
        #                     sys.exit(1)
        #
        #             else:
        #                 # 사용자가 “아니오”라고 하면 종료
        #                 speak_feedback("프로그램을 종료합니다.")
        #                 if current_tts_thread:
        #                     current_tts_thread.join()
        #                 sys.exit(0)
        #         continue
        #     if target is not None:
        #         miss_count = 0  # 검출 성공 시 카운트 초기화
        #         if not target_intro_done:
        #             speak_feedback("타겟을 찾았습니다. 안내를 시작하겠습니다.")
        #             target_intro_done = True

        if step == "find_target":
            if target is None:
                speak_feedback("타겟이 감지되지 않았습니다. 카메라를 천천히 움직여 다시 보여주세요.")
                continue

            if not target_intro_done:
                speak_feedback("타겟을 찾았습니다. 안내를 시작하겠습니다.")
                target_intro_done = True

            if frame is None:
                continue
            #if not target and not dest:
            #    speak_feedback("타겟과 목적지가 감지되지 않았습니다.")
            #    continue
            #elif not target:
            #    speak_feedback("타겟이 감지되지 않았습니다.")
            #    continue
            elif not dest:
                speak_feedback("목적지가 감지되지 않았습니다.")
                continue

            if not initial_target_direction_given and target and isinstance(target, tuple):
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
                elif (datetime.now() - last_close_to_target_time).total_seconds() >= 3:
                    speak_feedback("손을 뻗어 잡으세요.")

                    # 3초 대기 후 프레임 캡처
                    time.sleep(3.0)

                    with frame_lock:
                        latest_frame = frame_for_display.copy() if frame_for_display is not None else None

                    if latest_frame is not None and hand is not None:
                        hx, hy = hand
                        tx, ty = target
                        h, w, _ = latest_frame.shape
                        mx, my = int((hx + tx) / 2), int((hy + ty) / 2)
                        x1, x2 = max(0, mx - 200), min(w, mx + 200)
                        y1, y2 = max(0, my - 200), min(h, my + 200)
                        crop = latest_frame[y1:y2, x1:x2]
                        cv2.imwrite("debug_crop.jpg", crop)

                        # speak_feedback("손이 물체를 잡았는지 확인 중입니다.")
                        is_grabbed = ask_gpt_if_grabbed(crop, target)

                        if isinstance(is_grabbed, str):
                            is_grabbed = is_grabbed.strip().lower() == "true"

                        if is_grabbed:
                            #speak_feedback("손이 물체를 잡은 것이 확인되었습니다.")
                            target_grabbed = True
                            step = "move_to_destination"
                            destination_intro_done = False
                            target_intro_done = False
                            time.sleep(1.5)
                            continue
                        else:
                            speak_feedback("아직 잡지 않은 것 같아요.")
            else:
                    last_close_to_target_time = None

            direction = "오른쪽" if dx > 0 else "왼쪽" if abs(dx) > abs(dy) else "아래" if dy > 0 else "위"
            speak_feedback(f"{direction}으로 이동하세요.")

            # if distance < NEAR_THRESHOLD:
            #     if not near_intro_done:
            #         speak_feedback(f"거의 도착했어요. {direction}으로 이동하세요.")
            #         near_intro_done = True
            #     else:
            #         speak_feedback(f"{direction}으로 이동하세요.")
            # else:
            #     near_intro_done = False
            #     if not target_intro_done:
            #         speak_feedback(f"타겟에 접근 중입니다. {direction}으로 이동하세요.")
            #         target_intro_done = True
            #     else:
            #         speak_feedback(f"{direction}으로 이동하세요.")
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

            direction = "오른쪽" if dx > 0 else "왼쪽" if abs(dx) > abs(dy) else "아래" if dy > 0 else "위"

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

    # 1) 파노라마 스캔 및 검출
    print("파노라마 스캔을 위해 10시 방향으로 몸을 돌려주세요.")
    tts.say("파노라마 스캔을 위해 10시 방향으로 몸을 돌려주세요.")
    tts.runAndWait()
    pano = auto_panorama_scan(scan_duration=7.0, capture_interval=0.6, cam_index=2)
    if pano is None:
        exit("파노라마 실패 — 카메라를 천천히 움직여 다시 시도하세요.")
    # 저장 또는 디버깅용
    cv2.imwrite("panorama.jpg", pano)
    tgt_pos, dst_pos, labels = detect_on_panorama(pano, target, destination, return_labels=True)

    # 둘 다 못 찾았으면 한 번 더 시도
    if not tgt_pos and not dst_pos:
        speak_feedback("파노라마에서 타겟과 목적지를 찾지 못했습니다. 다시 한 번 스캔하겠습니다.")
        speak_feedback("8시 방향으로 돌아주세요. 그리고 다시 천천히 오른쪽으로 돌아봐주세요.")
        pano = auto_panorama_scan(scan_duration=7.0, capture_interval=0.6, cam_index=2)
        if pano is None:
            sys.exit("파노라마 재스캔 실패 — 프로그램을 종료합니다.")
        cv2.imwrite("panorama.jpg", pano)
        if destination:
            tgt_pos, dst_pos, labels = detect_on_panorama(pano, target, destination, return_labels=True)
        else:
            tgt_pos, _, labels = detect_on_panorama(pano, target, None, return_labels=True)
            dst_pos = None

    # 두 번 시도 후에도 못 찾았을 때
    if not tgt_pos and not dst_pos:
        seen = ", ".join(sorted(set(labels))) or "없음"
        speak_feedback(
            f"찾으시는 {target}"
            + (f" / {destination}" if destination else "")
            + f" 물체는 보이지 않습니다. "
            f"현재 인식된 물체는 {seen}입니다. 이 중에 필요하신 물건이 있나요?"
        )
        if not get_yes_no_response():
            speak_feedback("프로그램을 종료합니다.")
            sys.exit(0)
        else:
            speak_feedback("찾으실 물건을 말씀해주세요.")
            get_command()
            # 새 명령어 로드 & GPT 추출
            cmd_text = load_command()
            info = extract_target_with_gpt(cmd_text)
            target = info.get("target")
            destination = info.get("destination")
            # 재초기화 후 계속 진행

    # 정상 안내 분기 (destination 유무에 따라 다른 안내)
    if destination:
        if tgt_pos and dst_pos:
            speak_feedback("타겟과 목적지 물체를 찾았습니다. 안내를 시작하겠습니다.")
        elif tgt_pos:
            speak_feedback("타겟 물체만 찾았습니다. 안내를 시작하겠습니다.")
        else:
            speak_feedback("목적지 물체만 찾았습니다. 안내를 시작하겠습니다.")
    else:
        if tgt_pos:
            speak_feedback("물체를 찾았습니다. 안내를 시작하겠습니다.")
        else:
            speak_feedback("물체를 찾지 못했습니다. 종료합니다.")
            sys.exit(1)

    # 파노라마 위치 결과로 전역 변수 초기화
    target_pos = last_seen_target_pos = tgt_pos
    destination_pos = last_seen_destination_pos = dst_pos

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
            if any([target_pos, destination_pos, last_seen_target_pos, last_seen_destination_pos]):
                hand_pos = detect_hand(frame)
            else:
                hand_pos = None

        if hand_pos:
            cv2.circle(frame_for_display, hand_pos, 10, (0, 255, 0), -1)
        tp = target_pos or last_seen_target_pos
        if tp and isinstance(tp, tuple):
            cv2.circle(frame_for_display, tp, 10, (0, 0, 255), -1)
        dp = destination_pos or last_seen_destination_pos
        if dp and isinstance(dp, tuple):
            cv2.circle(frame_for_display, dp, 10, (255, 0, 0), -1)

        cv2.imshow("웹캠 미리보기", frame_for_display)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
