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
        "'어떤 것을 찾아줘' 또는 '어떤 것을 어디에 놓아줘' 등의 형식으로 말씀해주세요. 5초간 녹음됩니다."
    )
    print("🗣 안녕하세요, 저는 여러분의 눈이 되어 물건을 찾는 것을 도와드리겠습니다.")
    print("🗣 '어떤 것을 찾아줘' 또는 '어떤 것을 어디에 놓아줘' 등의 형식으로 말씀해주세요. 5초간 녹음됩니다.")
    tts.say(prompt)
    tts.runAndWait()

def record_audio():
    """마이크로 음성을 녹음하고 WAV 파일로 저장"""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("녹음 중...")
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

def transcribe_audio():
    """Whisper로 오디오 파일을 텍스트로 변환하고 저장"""
    #print("음성을 텍스트로 변환 중...")

    # 한국어 명시
    result = model.transcribe(OUTPUT_FILENAME, language="ko")
    command_text = result["text"].strip()

    # 말 안 했을 때
    if len(command_text) < 3:
        tts.say("음성이 감지되지 않았습니다. 다시 말씀해주세요.")
        print("🗣 음성이 감지되지 않았습니다. 다시 말씀해주세요.")
        tts.runAndWait()
        return None

    # GPT로 명령 의도 판별
    print("🧠 GPT로 음성 입력 검사 중…")
    try:
        if not is_move_intent(command_text):
            tts.say("'어떤 것을 찾아줘' 또는 '어떤 것을 어디에 놓아줘' 등의 형식으로 말씀해주세요.")
            print("🗣 '어떤 것을 찾아줘' 또는 '어떤 것을 어디에 놓아줘' 등의 형식으로 말씀해주세요.")
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


def is_move_intent(text: str) -> bool:
    prompt = (
        "아래 문장이 물체 찾기 또는 물체 이동 명령인지 판단해줘. "
        "명령이면 ‘yes’, 아니면 ‘no’로만 대답해줘.\n\n"
        "예시:\n"
        "  숟가락 어딨어? -> yes\n"
        "  리모컨 찾고 싶어 -> yes\n"
        "  마우스를 찾아서 노트북 옆에 놓고 싶어. -> yes\n"
        "  예쁜 컵을 사고 싶어. -> no\n"
        "  오늘 날씨 어때? -> no\n"
        "  음악 들려줘 -> no\n\n"
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


def get_yes_no_response():
    """녹음→Whisper 전사→‘예/아니오’ 여부 판단 후 True/False 리턴"""
    local_tts = pyttsx3.init()
    local_tts.setProperty("rate", 220)
    local_tts.say("이 중에 필요하신 물건이 있나요? 예 아니오로 대답해주세요.")
    print("🗣 이 중에 필요하신 물건이 있나요? 예 아니오로 대답해주세요.")
    local_tts.runAndWait()
    local_tts.stop()

    for attempt in range(3):
        # 1) 녹음
        record_audio()
        # 2) Whisper 전사
        result = model.transcribe(OUTPUT_FILENAME, language="ko")
        answer = result["text"].strip()
        # 3) 음성 없으면 재시도
        if not answer:
            local_tts.say("음성이 감지되지 않았습니다. 다시 말씀해주세요.")
            print("🗣 음성이 감지되지 않았습니다. 다시 말씀해주세요.")
            local_tts.runAndWait()
            continue
        # 4) ‘예’ 계열
        if any(w in answer for w in ["네", "예", "응", "어", "넵"]):
            return True
        # 5) ‘아니오’ 계열
        if any(w in answer for w in ["아니", "아니요", "아니오", "아니야", "아냐"]):
            return False
        # 6) 그 외 재입력 안내
        local_tts.say("잘 인식하지 못했어요. ‘예’ 또는 ‘아니오’로만 답해주세요.")
        print("🗣 잘 인식하지 못했어요. ‘예’ 또는 ‘아니오’로만 답해주세요.")
        local_tts.runAndWait()

    # 3회 모두 실패 시 종료
    local_tts = pyttsx3.init()
    local_tts.say("응답을 이해하지 못했습니다. 프로그램을 종료합니다.")
    print("🗣 응답을 이해하지 못했습니다. 프로그램을 종료합니다.")
    local_tts.runAndWait()
    sys.exit(0)


if __name__ == "__main__":
    speak_prompt()

    get_command()
