import openai
import json
import os

# API 키 로딩
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수를 확인하세요!")

openai.api_key = api_key

# 파일명
TEXT_FILENAME = "command.txt"
TASK_FILENAME = "task_plan_destination.json"

def load_command():
    """저장된 텍스트 명령을 불러오기"""
    try:
        with open(TEXT_FILENAME, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("❌ 음성 명령 파일을 찾을 수 없습니다. 먼저 voice_command.py를 실행하세요.")
        return None

def extract_target_with_gpt(command_text):
    system_prompt = """
    너는 AI 도우미야. 사용자의 명령에서 찾고자 하는 물체와 목적지를 분석해서 아래 기준에 따라 JSON으로 추출해야 해.

    - 기준:
    1. 컵, 휴지, 휴지통, 책상 등 YOLO로 인식 가능한 대상은 영어로 변환해서 'target' 또는 'destination'에 넣어줘.
    2. "컵을 물통 옆에 가져다줘", "휴지를 휴지통에 버려줘" 같은 **이동 명령**인 경우에는 'target'과 'destination' 모두 추출해야 해.
    3. "책상에서 휴지를 찾아줘", "컵을 찾아줘", "휴지 어디있어?" 같은 **단순 탐색 명령**인 경우에는 **무조건 'target'만 포함하고**, 'destination'은 **절대 넣지 마**.
    4. 번역은 꼭 YOLO에서 인식 가능한 클래스 이름으로 정확하게 넣어줘. 예: 컵 → cup, 가위 → scissors, 책상 → table

    - 출력 형식 예시:
    단순 탐색:
    { "target": "scissors" }

    복합 명령:
    { "target": "book", "destination": "table" }
    { "target": "cup", "destination": "bottle" }
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": f"사용자 명령: {command_text}"}
        ],
        max_tokens=150
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
        print(f" 인식된 명령어: {command_text}")
        target_info = extract_target_with_gpt(command_text)

        if target_info:
            print(f"GPT가 추출한 정보: {target_info}")
            with open(TASK_FILENAME, "w", encoding="utf-8") as f:
                json.dump(target_info, f, indent=4, ensure_ascii=False)
            print("✅ task_plan.json 파일로 저장 완료!")
        else:
            print("⚠ 타겟 정보를 추출할 수 없습니다.")
