# 설정 값들 모은 파일

import cv2
#import pyzed.sl as sl

# 카메라 속성 정의 (v4l2-ctl로 확인한 지원하는 값으로 설정)
# 명령어: v4l2-ctl -d /dev/arducam_left --list-formats-ext
CAM_CONFIG = {
    "left_cam": "/dev/arducam_left", "right_cam": "/dev/arducam_right",
    # Arducam 카메라에서 가능한 해상도: 1280x800, 800x600, 640x480, 320x240
    "frame_width": 800, "frame_height": 600, "fps": 15,
    "fourcc": cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')    # MJPEG 코덱
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

# 각 카메라가 frame을 저장할 버퍼(Queue)의 크기
BUFFER_SIZE = 3

# 카메라 연결 실패 처리 설정
CAMERA_TIMEOUT_SEC = 5.0  # 2초 동안 연속 실패 시 스레드 종료

# 메인 루프 타임아웃 설정
#MAIN_LOOP_TIMEOUT = (1 / CAM_CONFIG['fps']) * 3
MAIN_LOOP_TIMEOUT = 5.0

# 안정적인 매칭을 위해 허용되는 최대 시간 차이 (초)
MAX_ALLOWED_DIFF_SEC = (1 / CAM_CONFIG['fps']) * 1

# --- 실행 옵션 ---
GUI_DEBUG = False
IMG_SAVE = False