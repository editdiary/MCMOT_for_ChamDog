# 설정 값들 모은 파일

import cv2
#import pyzed.sl as sl

# ----- 저장 경로 설정 -----
# 이미지 저장 경로 (SCRIPT_DIR 아래 'output_images' 폴더)
OUTPUT_FOLDER_NAME = "output_images"

# YOLO 폴더 이름
WEIGHTS_FOLDER_NAME = "weights"


# ----- 카메라 하드웨어 설정 -----
# 카메라 속성 정의 (v4l2-ctl로 확인한 지원하는 값으로 설정해야 함)
# 명령어: v4l2-ctl -d /dev/arducam_left --list-formats-ext
CAM_CONFIG = {
    "left_cam": "/dev/arducam_left", "right_cam": "/dev/arducam_right",
    # Arducam 카메라에서 가능한 해상도: 1280x800, 800x600, 640x480, 320x240
    "frame_width": 800, "frame_height": 600, "fps": 15,
    "fourcc": cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
}

ZED_CONFIG = {
    "zed_cam": "/dev/zed",
    # zed에서 가능한 해상도: 4416x1242, 3840x1080, 2560x720, 1344x376
    "zed_width": 2560, "zed_height": 720, "zed_fps": 15
}

# ZED SDK용 설정
# ZED_CONFIG = {
#     "resolution": sl.RESOLUTION.HD720, "fps": 15,
#     "depth_mode": sl.DEPTH_MODE.NONE, "sdk_verbose": 1    
# }


# ----- 버퍼 설정 -----
BUFFER_SIZE = 3                     # 각 카메라 캡처 thread의 프레임 버퍼 크기
CAMERA_TIMEOUT_SEC = 3.0            # 카메라 grab 실패 시 스레드 종료까지 대기 시간 (초)
MAIN_LOOP_TIMEOUT = 0.2             # main thread가 카메라 큐를 기다리는 최대 시간 (초)
INFERENCE_BUFFER_SIZE = 5           # 추론 스레드 입력 큐의 버퍼 크기
SAVE_BUFFER_SIZE = 10               # [New!] 저장 스레드 입력 큐의 버퍼 크기
INFERENCE_WORKER_TIMEOUT = 1.0      # 추론 스레드가 입력 큐를 기다리는 최대 시간 (초)
CPU_WORKER_TIMEOUT = 1.0            # 추론 스레드가 CPU 작업 결과를 기다리는 최대 시간 (초)
SAVE_WORKER_TIMEOUT = 1.0           # [New!] 저장 스레드가 입력 큐를 기다리는 최대 시간 (초
THREAD_JOIN_TIMEOUT = 5.0           # 프로그램 종료 시 스레드 종료를 기다리는 최대 시간 (초)


# ----- 동기화 설정 -----
# 프레임 간격 (초)
FRAME_INTERVAL_SEC = 1 / ZED_CONFIG['zed_fps']
# 동기화 허용 오차 (초), 예: 1.1 프레임 간격
MAX_ALLOWED_DIFF_SEC = FRAME_INTERVAL_SEC * 1.1


# ----- 파이프 탐지 설정 -----
# ROI(관심 영역) 좌표 (x, y, width, height)
PIPE_ROI_COORS = {'x': 630, 'y': 350, 'width': 200, 'height': 300}
# 파이프 탐지에 사용할 ZED 프레임 (Flase: 왼쪽, True: 오른쪽)
USE_ZED_RIGHT_FOR_PIPE = False


# ----- 실행 옵션 -----
GUI_DEBUG = False       # True일 경우 OpenCV 창으로 실시간 영상 표시
IMG_SAVE = True        # True일 경우 동기화된 프레임 이미지 파일로 저장