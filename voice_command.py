import whisper
import pyaudio
import wave
import os
from pydub import AudioSegment
from pydub.effects import normalize

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

def record_audio():
    """마이크로 음성을 녹음하고 WAV 파일로 저장"""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("음성을 입력하세요... (5초 동안 녹음)")
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

    # 🔊 음량 정규화
    sound = AudioSegment.from_wav(OUTPUT_FILENAME)
    normalized_sound = normalize(sound)
    normalized_sound.export(OUTPUT_FILENAME, format="wav")


def transcribe_audio():
    """Whisper로 오디오 파일을 텍스트로 변환하고 저장"""
    print("음성을 텍스트로 변환 중...")

    # 🗣 한국어 명시
    result = model.transcribe(OUTPUT_FILENAME, language="ko")
    command_text = result["text"]

    with open(TEXT_FILENAME, "w", encoding="utf-8") as f:
        f.write(command_text)

    print(f"변환된 명령어 저장됨: {command_text}")
    return command_text


if __name__ == "__main__":
    record_audio()
    transcribe_audio()
