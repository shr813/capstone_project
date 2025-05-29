import os
# TensorFlow INFO/WARNING 메시지 억제 (0=ALL, 1=INFO, 2=WARNING, 3=ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# oneDNN 최적화 메시지 비활성화
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# absl (MediaPipe 등에서 사용) 로깅 레벨 ERROR로 설정
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

# OpenCV 로깅 레벨 ERROR로 (opencv-python 4.7 이상)
try:
    import cv2
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except AttributeError:
    # 구버전 opencv: cv2.useOptimized(False) 등 대체 옵션
    pass

import sys
import subprocess


def run_step(script_name):
    # print(f"▶ Running {script_name}...")
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"❌ {script_name} exited with code {result.returncode}. Aborting.")
        sys.exit(result.returncode)
    # print(f"✅ {script_name} completed successfully.\n")

if __name__ == '__main__':
    # 1. 음성 명령 받기
    run_step('voice_command.py')

    # 2. 명령어 분석 및 task_plan.json 생성
    run_step('task_planner_dest.py')

    # 3. 웹캠 가이드 실행
    run_step('webcam_gpt.py')

    sys.exit(0)
