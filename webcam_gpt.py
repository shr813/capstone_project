import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
import logging
logging.getLogger("absl").setLevel(logging.ERROR)
import sys
import cv2
if hasattr(cv2, 'utils') and hasattr(cv2.utils, 'logging'):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
import json
import math
import time
import threading
import pyttsx3
import mediapipe as mp
try:
    mp.logging.set_verbosity(mp.logging.ERROR)
except:
    # older versions of MediaPipe
    logging.getLogger('mediapipe').setLevel(logging.ERROR)
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
last_seen_target_time = 0.0
last_seen_destination_time = 0.0
EXPIRE_TIME = 3.0   #3초이상 감지 안 되면 점 지움
hand_pos, frame_for_display = None, None
frame_lock = threading.Lock()
last_hand_feedback_time = 0
near_intro_done = False
miss_count = 0  # 물체 미검출 시 카운트
pano_width, pano_height = None, None
pan_target_pos, pan_dest_pos = None, None
pan_target_prompted = False
pan_dest_prompted = False

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
    print("🗣 파노라마 스캔을 7초동안 진행하겠습니다. 오른쪽으로 천천히 돌아봐주세요.")
    tts.say("파노라마 스캔을 7초동안 진행하겠습니다. 오른쪽으로 천천히 돌아봐주세요.")
    tts.runAndWait()

    start_t = time.time()
    next_capture = start_t

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
            labels.append(cls)

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

    # def tts_job():
    #     with tts_lock:
    #         try:
    #             tts.stop()
    #             tts.say(text)
    #             tts.runAndWait()
    #         except:
    #             pass
    #
    # if current_tts_thread and current_tts_thread.is_alive():
    #     pass
    # current_tts_thread = threading.Thread(target=tts_job, daemon=True)
    # current_tts_thread.start()

    with tts_lock:
        tts.say(text)
        tts.runAndWait()

def speak_hand_feedback(text):
    global last_hand_feedback_time, current_tts_thread
    now = time.time()
    if now - last_hand_feedback_time < HAND_FEEDBACK_INTERVAL:
        return
    last_hand_feedback_time = now
    print("🗣", text)

    # def tts_job():
    #     with tts_lock:
    #         try:
    #             tts.stop()
    #             tts.say(text)
    #             tts.runAndWait()
    #         except:
    #             pass
    #
    # if current_tts_thread and current_tts_thread.is_alive():
    #     pass
    # current_tts_thread = threading.Thread(target=tts_job, daemon=True)
    # current_tts_thread.start()

    with tts_lock:
        tts.say(text)
        tts.runAndWait()

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
    # if pos is None:
    #     return "타겟 위치를 찾을 수 없습니다."

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
            return str(parsed.get("grabbed", "")).strip().lower() == "true"
        except json.JSONDecodeError:
            print("⚠️ JSON 파싱 실패. 응답 내용 확인:", result)
            return False

    except Exception as e:
        print("[GPT 오류]", e)
        return False


# ------------------ 피드백 루프 ------------------

def feedback_loop():
    global step, target_intro_done, destination_intro_done, target_grabbed, last_close_to_target_time
    global initial_target_direction_given, last_hand_feedback_time
    global pan_target_prompted, pan_dest_prompted

    while True:
        time.sleep(FEEDBACK_INTERVAL)

        with frame_lock:
            hand = hand_pos
            tgt = target_pos or last_seen_target_pos
            dst = destination_pos or last_seen_destination_pos
            frame = frame_for_display.copy() if frame_for_display is not None else None

        # 1) find_target 단계
        if step == "find_target":
            # 타겟이 처음 감지되었을 때, 딱 한 번만 방향 안내
            if tgt is not None and not initial_target_direction_given:
                # frame 크기: (width, height)
                msg = get_initial_direction_comment(tgt, (frame.shape[1], frame.shape[0]))
                speak_feedback(msg)
                initial_target_direction_given = True
            # 타겟이 없으면 파노라마 기준 좌/우 안내 (한 번만)
            if tgt is None:
                if pan_target_pos:
                    if not pan_target_prompted:
                        msg = "오른쪽을 봐주세요." if pan_target_pos[0] > pano_width / 2 else "왼쪽을 봐주세요."
                        speak_feedback(msg)
                        pan_target_prompted = True
                else:
                    speak_feedback("타겟 위치 정보를 알 수 없습니다.")
                continue

            # 타겟이 보이면 플래그 리셋
            pan_target_prompted = False

            # 타겟이 잡혔는지 확인하기 전까지는 거리 안내
            if not hand:
                speak_hand_feedback("손이 감지되지 않습니다.")
                continue

            dx, dy = tgt[0] - hand[0], tgt[1] - hand[1]
            dist = math.hypot(dx, dy)

            # 손이 타겟 가까이 왔을 때 잡기 유도 + 확인
            if dist < ARRIVE_THRESHOLD:
                if not last_close_to_target_time:
                    last_close_to_target_time = datetime.now()
                elif (datetime.now() - last_close_to_target_time).total_seconds() >= 3:
                    speak_feedback("손을 뻗어 잡으세요.")
                    # 잡기 확인 로직
                    time.sleep(4.0)
                    with frame_lock:
                        latest_frame = frame_for_display.copy() if frame_for_display is not None else None

                    if latest_frame is not None and hand is not None:
                        hx, hy = hand
                        tx, ty = tgt
                        h, w, _ = latest_frame.shape
                        mx, my = int((hx + tx) / 2), int((hy + ty) / 2)
                        x1, x2 = max(0, mx - 200), min(w, mx + 200)
                        y1, y2 = max(0, my - 200), min(h, my + 200)
                        crop = latest_frame[y1:y2, x1:x2]
                        cv2.imwrite("debug_crop.jpg", crop)

                        is_grabbed = ask_gpt_if_grabbed(crop, target)
                        if isinstance(is_grabbed, str):
                            is_grabbed = is_grabbed.strip().lower() == "true"

                        if is_grabbed:
                            speak_feedback("잘 잡았어요.")
                            target_grabbed = True
                            step = "move_to_destination"
                            target_intro_done = False
                            destination_intro_done = False
                            time.sleep(1.5)
                            continue
                        else:
                            speak_feedback("아직 잡지 않은 것 같아요.")
            else:
                last_close_to_target_time = None
                # 손↔타겟 거리 안내
                direction = "오른쪽" if dx > 0 else "왼쪽" if abs(dx) > abs(dy) else "위" if dy < 0 else "아래"
                speak_feedback(f"{direction}으로 이동하세요.")
            continue


        # 2) move_to_destination 단계
        elif step == "move_to_destination":
            # (a) 목적지가 화면에 보이지 않을 때 → 파노라마 기준 좌/우 안내
            if dst is None:
                if pan_dest_pos:
                    if not pan_dest_prompted:
                        msg = "오른쪽을 봐주세요." if pan_dest_pos[0] > pano_width / 2 else "왼쪽을 봐주세요."
                        speak_feedback(msg)
                        pan_dest_prompted = True
                else:
                    speak_feedback("목적지 위치 정보를 알 수 없습니다.")
                continue

            pan_dest_prompted = False

            # (b) 목적지가 보이면 손↔목적지 거리 안내
            if not hand:
                speak_hand_feedback("손이 감지되지 않습니다.")
                continue

            dx, dy = dst[0] - hand[0], dst[1] - hand[1]
            dist = math.hypot(dx, dy)

            if dist < ARRIVE_THRESHOLD:
                speak_feedback("목적지에 도착했습니다. 내려놓으세요.")
                step = "done"
            else:
                direction = "오른쪽" if dx > 0 else "왼쪽" if abs(dx) > abs(dy) else "위" if dy < 0 else "아래"
                speak_feedback(f"{direction}으로 이동하세요.")
            continue


# ------------------ 객체 위치 갱신 루프 ------------------

def yolo_loop():
    global target_pos, destination_pos, last_seen_target_pos, last_seen_destination_pos
    global last_seen_target_time, last_seen_destination_time
    while True:
        time.sleep(YOLO_INTERVAL)
        with frame_lock:
            frame = frame_for_display.copy() if frame_for_display is not None else None
        if frame is None:
            continue

        if target:
            pos = find_object_position(frame, target)
            if pos:
                target_pos = last_seen_target_pos = pos
                last_seen_target_time = time.time()
        if destination:
            pos = find_object_position(frame, destination)
            if pos:
                destination_pos = last_seen_destination_pos = pos
                last_seen_destination_time = time.time()


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
    print("🗣 파노라마 스캔을 위해 10시 방향으로 몸을 돌려주세요.")
    tts.say("파노라마 스캔을 위해 10시 방향으로 몸을 돌려주세요.")
    tts.runAndWait()
    pano = auto_panorama_scan(scan_duration=7.0, capture_interval=0.6, cam_index=2)

    if pano is None:
        # stitch 실패 시 한 번 더 재시도
        speak_feedback("파노라마 생성에 실패했습니다. 다시 한 번 스캔하겠습니다.")
        speak_feedback("8시 방향으로 돌아주세요.")
        pano = auto_panorama_scan(scan_duration=7.0, capture_interval=0.6, cam_index=2)

    if pano is None:
        sys.exit("파노라마 재시도에도 실패했습니다. 프로그램을 종료합니다.")

    cv2.imwrite("panorama.jpg", pano)
    tgt_pos, dst_pos, labels = detect_on_panorama(pano, target, destination, return_labels=True)
    pano_width, pano_height = pano.shape[1], pano.shape[0]
    pan_target_pos, pan_dest_pos = tgt_pos, dst_pos

    # 둘 다 못 찾았으면 한 번 더 시도
    if not tgt_pos and not dst_pos:
        speak_feedback("파노라마에서 타겟과 목적지를 찾지 못했습니다. 다시 한 번 스캔하겠습니다.")
        speak_feedback("8시 방향으로 돌아주세요.")
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
            f"현재 보이는 물체는 {seen}입니다."
        )
        if current_tts_thread and current_tts_thread.is_alive():
            current_tts_thread.join()
        answer = get_yes_no_response()
        if not answer:
            speak_feedback("프로그램을 종료하겠습니다.")
            sys.exit(0)
        else:
            speak_feedback("필요한 물건을 말씀해주세요.")
            get_command()
            # 새 명령어 로드 & GPT 추출
            cmd_text = load_command()
            info = extract_target_with_gpt(cmd_text)
            target = info.get("target")
            destination = info.get("destination")
            # 재초기화 후 계속 진행

    # 정상 안내 분기 (destination 유무에 따라 다른 안내)
    seen = ", ".join(sorted(set(labels))) or "없음"
    if destination:
        if tgt_pos and dst_pos:
            speak_feedback("타겟과 목적지 물체를 찾았습니다. 안내를 시작하겠습니다.")
        elif tgt_pos:
            speak_feedback(f"{target}만 찾았습니다. 현재 보이는 물체는 {seen}입니다.")
            if current_tts_thread and current_tts_thread.is_alive():
                current_tts_thread.join()
            answer = get_yes_no_response()
            if not answer:
                speak_feedback("프로그램을 종료하겠습니다.")
                sys.exit(0)
            else:
                speak_feedback("찾으실 물건을 말씀해주세요.")
                get_command()
                # 새 명령어 로드 & GPT 추출
                cmd_text = load_command()
                info = extract_target_with_gpt(cmd_text)
                target = info.get("target")
                destination = info.get("destination")
        else:
            speak_feedback(f"{destination}만 찾았습니다. 현재 보이는 물체는 {seen}입니다.")
            if current_tts_thread and current_tts_thread.is_alive():
                current_tts_thread.join()
            answer = get_yes_no_response()
            if not answer:
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
    else:
        if not tgt_pos:
            speak_feedback("물체를 찾지 못했습니다. 현재 보이는 물체는 {seen}입니다. ")
            if current_tts_thread and current_tts_thread.is_alive():
                current_tts_thread.join()
            answer = get_yes_no_response()
            if not answer:
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


    # 파노라마 위치 결과로 전역 변수 초기화
    target_pos = last_seen_target_pos = tgt_pos
    destination_pos = last_seen_destination_pos = dst_pos
    pano_width, pano_height = pano.shape[1], pano.shape[0]
    pan_target_pos, pan_dest_pos = tgt_pos, dst_pos

    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    threading.Thread(target=feedback_loop, daemon=True).start()
    threading.Thread(target=yolo_loop, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        with frame_lock:
            if target_pos and now - last_seen_target_time > EXPIRE_TIME:
                target_pos = None
                last_seen_target_pos = None
            if destination_pos and now - last_seen_destination_time > EXPIRE_TIME:
                destination_pos = None
                last_seen_destination_pos = None

            frame_for_display = frame.copy()
            if target_pos or destination_pos:
                hand_pos = detect_hand(frame)
            else:
                hand_pos = None

        if hand_pos:
            cv2.circle(frame_for_display, hand_pos, 10, (0, 255, 0), -1)

        now = time.time()
        tp = None
        if target_pos:
            tp = target_pos
        elif last_seen_target_pos and now - last_seen_target_time < EXPIRE_TIME:
            tp = last_seen_target_pos
        elif last_seen_target_pos and hand_pos:  # 손이 가려서 안 보인 경우: 손과 마지막 위치가 가까우면 유지
            if math.hypot(hand_pos[0] - last_seen_target_pos[0],
                          hand_pos[1] - last_seen_target_pos[1]) < ARRIVE_THRESHOLD:
                tp = last_seen_target_pos

        if tp and isinstance(tp, tuple):
            cv2.circle(frame_for_display, tp, 10, (0, 0, 255), -1)

        dp = None
        if destination_pos:
            dp = destination_pos
        elif last_seen_destination_pos and now - last_seen_destination_time < EXPIRE_TIME:
            dp = last_seen_destination_pos
        elif last_seen_destination_pos and hand_pos:
            if math.hypot(hand_pos[0] - last_seen_destination_pos[0],
                          hand_pos[1] - last_seen_destination_pos[1]) < ARRIVE_THRESHOLD:
                dp = last_seen_destination_pos

        if dp and isinstance(dp, tuple):
            cv2.circle(frame_for_display, dp, 10, (255, 0, 0), -1)

        cv2.imshow("웹캠 미리보기", frame_for_display)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
