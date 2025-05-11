import whisper
import pyaudio
import wave
import os
from pydub import AudioSegment
from pydub.effects import normalize
import pyttsx3
import sys
import openai
import warnings

warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU; using FP32 instead"
)

# 녹음 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
OUTPUT_FILENAME = "live_command.wav"
TEXT_FILENAME = "command.txt"

# Whisper 모델 로드
model = whisper.load_model("medium")

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# TTS 엔진 초기화 (전역)
tts = pyttsx3.init()
tts.setProperty("rate", 200)

def speak_prompt():
    prompt = (
        "안녕하세요, 저는 여러분의 눈이 되어 물건을 찾는 것을 도와드리겠습니다. "
        "'어떤 것을 찾아줘' 또는 '어떤 것을 어디에 두고 싶어' 등의 형식으로 말씀해주세요. 5초간 녹음됩니다."
    )
    tts.say(prompt)
    tts.runAndWait()

def record_audio():
    """마이크로 음성을 녹음하고 WAV 파일로 저장"""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("- 음성을 입력하세요... (5초 동안 녹음)")
    frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS))]

    print("녹음 완료! 파일 저장 중...")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    # 음량 정규화
    sound = AudioSegment.from_wav(OUTPUT_FILENAME)
    normalized_sound = normalize(sound)
    normalized_sound.export(OUTPUT_FILENAME, format="wav")


def is_move_intent(text: str) -> bool:
    """GPT에게 이 문장이 물건 탐색/이동 명령인지 yes/no로만 물어봄"""
    prompt = (
        "아래 문장이 물체 찾기 또는 물체 이동 명령인지 판단해줘. "
        "명령이면 ‘yes’, 아니면 ‘no’로만 대답해줘.\n\n"
        f"문장: \"{text.strip()}\""
    )
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",  "content": "당신은 문장 분류 전문가입니다."},
            {"role": "user",    "content": prompt}
        ],
        temperature=0
    )
    answer = resp.choices[0].message.content.strip().lower()
    print(f"입력 분류 결과: {answer}")
    return answer.startswith("yes")


def transcribe_audio():
    """Whisper로 오디오 파일을 텍스트로 변환하고 저장"""
    #print("음성을 텍스트로 변환 중...")

    # 한국어 명시
    result = model.transcribe(OUTPUT_FILENAME, language="ko")
    command_text = result["text"].strip()

    # 말 안 했을 때
    if len(command_text) < 3:
        tts.say("음성이 감지되지 않았습니다. 다시 말씀해주세요.")
        print("음성 감지 안 됨. 5초간 재녹음 시작...")
        tts.runAndWait()
        return None

    # GPT로 명령 의도 판별
    print("🧠 GPT로 음성 입력 검사 중…")
    try:
        if not is_move_intent(command_text):
            tts.say("'어떤 것을 찾아줘' 또는 '어떤 것을 어디에 두고 싶어' 등의 형식으로 말씀해주세요.")
            tts.runAndWait()
            return None
    except Exception as e:
        print("[GPT 오류]", e)
        tts.say("명령을 이해하지 못했습니다. 다시 시도해주세요.")
        tts.runAndWait()
        return None

    with open(TEXT_FILENAME, "w", encoding="utf-8") as f:
        f.write(command_text)
    print(f"변환된 명령어 저장됨: {command_text}")
    return command_text

def get_command():
    for attempt in range(3):
        record_audio()
        cmd = transcribe_audio()
        if cmd is not None:
            break # 정상 문장이 들어왔을 때 loop 탈출
    else:
        # 3회 모두 실패
        tts.say("알 수 없는 입력이 반복되어 시스템을 종료합니다.")
        tts.runAndWait()
        sys.exit(0)

# def get_yes_no_response(prompt_text: str) -> bool:
#     """
#     prompt_text 를 TTS로 읽고,
#     사용자의 응답을 녹음 → Whisper 전사 → GPT로 예/아니오 분류하여
#     긍정(yes)이면 True, 부정(no)이면 False 반환
#     """
#     # 1) 질문
#     tts.say(prompt_text)
#     tts.runAndWait()
#
#     # 2) 최대 3회 시도
#     for _ in range(3):
#         record_audio()
#         # Whisper 전사
#         result = model.transcribe(OUTPUT_FILENAME, language="ko")["text"].strip()
#         if len(result) < 1:
#             tts.say("응답이 잘 들리지 않았습니다. 다시 말씀해주세요.")
#             tts.runAndWait()
#             continue
#
#         # 3) GPT로 예/아니오 분류
#         classification_prompt = (
#             "다음 응답이 ‘네(긍정)’인지 ‘아니오(부정)’인지 판단해 주세요. "
#             "반드시 ‘yes’ 또는 ‘no’만 답해주세요.\n\n"
#             f"응답: \"{result}\""
#         )
#         resp = openai.ChatCompletion.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "당신은 예/아니오 분류 전문가입니다."},
#                 {"role": "user",   "content": classification_prompt}
#             ],
#             temperature=0
#         )
#         answer = resp.choices[0].message.content.strip().lower()
#         if answer.startswith("yes"):
#             return True
#         if answer.startswith("no"):
#             return False
#
#         # 그 외 응답일 때
#         tts.say("예 또는 아니오로만 대답해 주세요.")
#         tts.runAndWait()
#
#     # 3회 모두 이해 불가 시 종료
#     tts.say("입력을 이해하지 못해 시스템을 종료합니다.")
#     tts.runAndWait()
#     sys.exit(0)

def get_yes_no_response():
    """녹음→Whisper 전사→‘예/아니오’ 여부 판단 후 True/False 리턴"""
    tts.say("다른 물건을 찾으시겠습니까? 예 또는 아니오로 대답해주세요.")
    print("다른 물건을 찾으시겠습니까? 예/아니오")

    tts.runAndWait()

    for attempt in range(3):
        # 1) 녹음
        record_audio()
        # 2) Whisper 전사
        result = model.transcribe(OUTPUT_FILENAME, language="ko")
        answer = result["text"].strip()
        # 3) 음성 없으면 재시도
        if not answer:
            tts.say("음성이 감지되지 않았습니다. 다시 말씀해주세요.")
            tts.runAndWait()
            continue
        # 4) ‘예’ 계열
        if any(w in answer for w in ["네", "예", "응", "어"]):
            return True
        # 5) ‘아니오’ 계열
        if any(w in answer for w in ["아니", "아니요", "아니오", "아니야"]):
            return False
        # 6) 그 외 재입력 안내
        tts.say("잘 인식하지 못했어요. ‘예’ 또는 ‘아니오’로만 답해주세요.")
        tts.runAndWait()

    # 3회 모두 실패 시 종료
    tts.say("응답을 이해하지 못했습니다. 프로그램을 종료합니다.")
    tts.runAndWait()
    sys.exit(0)


if __name__ == "__main__":
    speak_prompt()
    #record_audio()
    #transcribe_audio()

    get_command()
