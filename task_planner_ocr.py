import openai
import json
import os

# ✅ API 키 로딩
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수를 확인하세요!")

client = openai.OpenAI(api_key=api_key)

# ✅ 파일명 통일
TEXT_FILENAME = "command_ocr.txt"
TASK_FILENAME = "task_plan_ocr.json"

def load_command():
    """📥 저장된 텍스트 명령을 불러오기"""
    try:
        with open(TEXT_FILENAME, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("❌ 음성 명령 파일을 찾을 수 없습니다. 먼저 voice_command.py를 실행하세요.")
        return None

def extract_target_with_gpt(command_text):
    """🤖 GPT로 타겟 정보 추출"""
    system_prompt = """
    너는 AI 도우미야. 사용자의 명령에서 찾고자 하는 **물체 또는 책 제목**을 분석해서 아래 기준에 따라 JSON으로 추출해야 해.

    - 기준:
    1. 컵, 가위, 휴지통 등 일상 사물은 YOLO로 인식되므로 YOLO 클래스 이름을 영어로 'target_english'에 넣어줘.
    2. 책 제목, 문서, 포스터, 라벨처럼 글자가 보이는 대상은 OCR로 인식되므로, 텍스트 자체를 'target_original'에 그대로 넣어줘.
    3. 번역하거나 요약하지 말고, 텍스트는 원문 그대로 사용해야 해.
    4. 'target_english'와 'target_original' 둘 중 하나만 있어도 괜찮지만, 가능한 경우 둘 다 넣어줘.

    - 출력 형식 예시:
    {
      "target_original": "파이썬",
      "target_english": "book"
    }
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"사용자 명령: {command_text}"}
        ],
        max_tokens=100
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except json.JSONDecodeError:
        print("❌ GPT 응답을 JSON으로 파싱할 수 없습니다.")
        return None

if __name__ == "__main__":
    command_text = load_command()
    if command_text:
        print(f"📜 인식된 명령어: {command_text}")
        target_info = extract_target_with_gpt(command_text)

        if target_info:
            print(f"🎯 GPT가 추출한 타겟 정보: {target_info}")
            with open(TASK_FILENAME, "w", encoding="utf-8") as f:
                json.dump(target_info, f, indent=4, ensure_ascii=False)
            print("✅ task_plan_ocr.json 파일로 저장 완료!")
        else:
            print("⚠ 타겟 정보를 추출할 수 없습니다.")
