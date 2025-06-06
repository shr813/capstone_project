# 👁️‍🗨️ 시각장애인을 위한 AI 음성 안내 서비스
**Team. Sido | Capstone Design 2025**

---

## 📌 프로젝트 개요

본 프로젝트는 시각장애인을 위한 **AI 기반 실시간 음성 안내 시스템**입니다.  
사용자의 음성 명령을 인식해 **물체 탐지**, **손 위치 추적**, **음성 피드백 제공**까지 자동으로 수행합니다.

- 명령 인식: Whisper → GPT-4o-mini
- 명령 분석: 대상/목적지 물체 이름 추출
- 객체 인식: YOLOv8 (ultralytics)
- 손 추적: MediaPipe Hands
- 파노라마 생성 및 실시간 웹캠 캡쳐 : OpenCV
- 실시간 피드백: pyttsx3 기반 TTS 출력

---

## 🛠️ 주요 기능

| 기능 | 설명 |
|------|------|
|  **음성 명령 인식** | 사용자의 자연어 명령을 Whisper로 텍스트로 변환 |
|  **명령 분석** | GPT-4o-mini로 명령 분석 및 타겟/목적지 물체 이름 추출 |
|  **물체 탐지** | YOLOv8으로 주변 사물의 위치 탐지 |
|  **손 위치 추적** | MediaPipe Hands로 손 위치 실시간 감지 |
|  **TTS 피드백** | 손과 물체 간 거리, 방향에 따라 음성 피드백 제공 |
|  **잡기 여부 판단** | 손이 물체를 잡았는지 GPT-4o-mini로 최종 판단 |

---

## 🖼️ 시스템 구조도

![Image](https://github.com/user-attachments/assets/02427961-a3ee-47be-be6d-456d73c5eca4)

---

## 🚀 사용 예시

```plaintext
🧑 사용자: "컵을 물통 옆에 놔줘"

① Whisper → 텍스트 변환
② GPT-4o-mini → {"target": "cup", "destination": "bottle"}
③ OpenCV Stitcher, YOLOv8 → 파노라마 촬영 후 컵, 물통 위치 탐지
④ MediaPipe → 손 위치 추적
⑤ TTS → "오른쪽으로 이동하세요", "위로 이동하세요" → "손을 뻗어 잡으세요"
⑥ 손이 올바른 물체를 잡으면 → "잘 잡았어요." → 목적지 물체가 있다면 목적지 탐지 단계로 전환하여 3번부터 실행
