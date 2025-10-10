import cv2
import pyzed.sl as sl
import threading
import time
from queue import Queue

# --- 설정 값들은 상단에 유지 ---
# 카메라 속성 정의
# v4l2-ctl로 확인한 지원하는 값으로 설정해야 한다.
    # 명령어: v4l2-ctl -d /dev/arducam_left --list-formats-ext
CAM_CONFIG = {
    "left_cam": "/dev/arducam_left", "right_cam": "/dev/arducam_right",
    "frame_width": 640, "frame_height": 480, "fps": 30,
    "fourcc": cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')    # MJPEG 코덱 설정 (압축 포맷 사용 시)
}

ZED_CONFIG = {
    "resolution": sl.RESOLUTION.HD720, "fps": 30,   # 원하는 해상도 & 프레임률
    "depth_mode": sl.DEPTH_MODE.NONE, "sdk_verbose": 1    
}

def arducam_worker(cap, q):
    """Arducam 프레임을 읽어 큐에 넣는 워커 함수"""
    while True:
        ret, frame = cap.read()
        if ret:
            # OpenCV는 별도의 타임스탬프 기능이 없으므로, 읽은 직후 시스템 시간 사용
            timestamp = time.time()
            q.put((timestamp, frame))

def zed_worker(zed, q):
    """ZED 카메라 프레임을 읽어 큐에 넣는 워커 함수"""
    runtime = sl.RuntimeParameters()
    zed_image = sl.Mat()
    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # ZED SDK는 이미지 캡처 시점의 타임스탬프를 제공하므로 더 정확함
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            frame = zed_image.get_data()
            q.put((timestamp, frame))

def initialize_arducam_cameras():
    # udev 규칙으로 만든 카메라 별명 사용
    left_cam = CAM_CONFIG['left_cam']
    right_cam = CAM_CONFIG['right_cam']

    # 카메라 열기
    left_cap = cv2.VideoCapture(left_cam, cv2.CAP_V4L2)
    right_cap = cv2.VideoCapture(right_cam, cv2.CAP_V4L2)

    if not left_cap.isOpened():
        print(f"오류: 카메라를 열 수 없습니다. 경로: {left_cam}")
    if not right_cap.isOpened():
        print(f"오류: 카메라를 열 수 없습니다. 경로: {right_cam}")
    
    # 카메라 속성 설정
    left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_CONFIG['frame_width'])
    left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_CONFIG['frame_height'])
    left_cap.set(cv2.CAP_PROP_FPS, CAM_CONFIG['fps'])
    left_cap.set(cv2.CAP_PROP_FOURCC, CAM_CONFIG['fourcc'])

    print(f"왼쪽 카메라 '{left_cam}'를 성공적으로 열었습니다.")
    print(f"해상도: {left_cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {left_cap.get(cv2.CAP_PROP_FPS)}")

    right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_CONFIG['frame_width'])
    right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_CONFIG['frame_height'])
    right_cap.set(cv2.CAP_PROP_FPS, CAM_CONFIG['fps'])
    right_cap.set(cv2.CAP_PROP_FOURCC, CAM_CONFIG['fourcc'])

    print(f"오른쪽 카메라 '{right_cam}'를 성공적으로 열었습니다.")
    print(f"해상도: {right_cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{right_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {right_cap.get(cv2.CAP_PROP_FPS)}")

    return left_cap, right_cap

def initialize_zed_camera():
    zed = sl.Camera()   # ZED 카메라 객체 생성

    # 초기화 파라미터 설정
    init = sl.InitParameters()
    init.camera_resolution = ZED_CONFIG['resolution']
    init.camera_fps = ZED_CONFIG['fps']
    init.depth_mode = ZED_CONFIG['depth_mode']
    init.sdk_verbose = ZED_CONFIG['sdk_verbose']

    # 카메라 열기
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS: # Ensure the camera has opened successfully
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()
    
    # Runtime 파라미터 설정 (여기서는 기본값 사용)
    runtime = sl.RuntimeParameters()
    zed_image = sl.Mat()    # 실시간 프레임을 받아올 객체 생성
    
    return zed, runtime, zed_image


def main():
    # Arducam 카메라 초기화
    left_cap, right_cap = initialize_arducam_cameras()

    # ZED 카메라 초기화
    zed, runtime, zed_image = initialize_zed_camera()

    # 각 카메라의 프레임을 담을 큐 생성
    q_zed = Queue(maxsize=1)
    q_left = Queue(maxsize=1)
    q_right = Queue(maxsize=1)

    # 스레드 생성
    thread_zed = threading.Thread(target=zed_worker, args=(zed, q_zed))
    thread_left = threading.Thread(target=arducam_worker, args=(left_cap, q_left))
    thread_right = threading.Thread(target=arducam_worker, args=(right_cap, q_right))

    # 스레드 시작 (데몬 스레드로 설정하면 메인 스레드 종료 시 함께 종료됨)
    thread_zed.daemon = True
    thread_left.daemon = True
    thread_right.daemon = True
    thread_zed.start()
    thread_left.start()
    thread_right.start()

    # 메인 루프
    # 동기화 허용 오차 (e.g. 30fps의 절반 시간, 약 16ms)
    SYNC_THRESHOLD_SEC = (1 / CAM_CONFIG['fps'] / 2)  # e.g., (1 / 30 / 2) = 약 0.0167초

    exit_app = False
    GUI_DEBUG = False
    try:
        while not exit_app:
            # 각 큐에서 최신 프레임 가져오기
            ts_zed, frame_z = q_zed.get()
            ts_left, frame_l = q_left.get()
            ts_right, frame_r = q_right.get()

            # ZED 타임스탬프는 나노초 단위일 수 있으므로 단위를 통일해야 함 (OpenCV의 time.time()은 초 단위)
            ts_zed_sec = ts_zed.get_nanoseconds() * 1e-9

            # 타임스탬프 차이 계산
            diff_zl = abs(ts_zed_sec - ts_left)
            diff_zr = abs(ts_zed_sec - ts_right)

            # 모든 카메라의 프레임이 허용 오차 내에 있는지 확인
            if diff_zl < SYNC_THRESHOLD_SEC and diff_zr < SYNC_THRESHOLD_SEC:
                print(f"Synced frames found! Timestamps: ZED={ts_zed_sec:.3f}, L={ts_left:.3f}, R={ts_right:.3f}")
                
                # 이 프레임 세트(frame_z, frame_l, frame_r)를 가지고 다음 처리 수행
                # ZED는 4채널(RGBA) 이미지를 반환하므로, RGB로 변환
                frame_z_rgb = cv2.cvtColor(frame_z, cv2.COLOR_RGBA2RGB)

                # GUI 디버깅 할 때 사용
                if GUI_DEBUG:
                    cv2.imshow("ZED Camera", frame_z_rgb)
                    cv2.imshow("Arducam LEFT", frame_l)
                    cv2.imshow("Arducam RIGHT", frame_r)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        exit_app = True
                        break
            else:
                # 동기화가 맞지 않으면 가장 오래된 프레임을 버리고 다음 루프에서 새 프레임을 기다리는 전략 사용 가능
                print("Frames out of sync, skipping...")
                # 추가로 디버깅을 위해 현재 시간 차이를 출력
                print(f"Frames out of sync! Time diffs: Z-L={diff_zl:.4f}s, Z-R={diff_zr:.4f}s")

    except KeyboardInterrupt:
        print("KeyboardInterrupt 발생. 프로그램을 종료합니다.")
        exit_app = True

    finally:
        if GUI_DEBUG:
            cv2.destroyAllWindows()
        left_cap.release()
        right_cap.release()
        zed.close()
        print("카메라가 안전하게 닫혔습니다.")

    print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()