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

class Camera:
    """모든 카메라 클래스가 상속할 기본 클래스 (향후 확장용)"""
    def __init__(self):
        self.thread = None
        self.queue = Queue(maxsize=1)
    
    def start(self):
        """캡처 스레드 시작"""
        self.thread.start()
    
    def read(self):
        """큐에서 최신 프레임 가져오기"""
        return self.queue.get()
    
    def stop(self):
        """카메라 리소스 해제 (자식 클래스에서 구현)"""
        raise NotImplementedError

class ArduCam(Camera):
    """Arducam을 제어하는 클래스"""
    def __init__(self, device_path, config):
        super().__init__()
        self.device_path = device_path
        self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise IOError(f"Arducam을 열 수 없습니다: {self.device_path}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['frame_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['frame_height'])
        self.cap.set(cv2.CAP_PROP_FPS, config['fps'])
        self.cap.set(cv2.CAP_PROP_FOURCC, config['fourcc'])
        print(f"Arducam '{self.device_path}' 초기화 완료.")
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
    
    def _worker(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.queue.put((time.time(), frame))
    
    def stop(self):
        self.cap.release()
        print(f"Arducam '{self.device_path}'가 닫혔습니다.")

class ZedCam(Camera):
    """ZED 카메라를 제어하는 클래스"""
    def __init__(self, config):
        super().__init__()
        self.zed = sl.Camera()
        init_params = sl.InitParameters(
            camera_resolution=config['resolution'],
            camera_fps=config['fps'],
            depth_mode=config['depth_mode'],
            sdk_verbose=config['sdk_verbose']
        )
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise IOError(f"ZED 카메라를 열 수 없습니다: {repr(status)}")

        print(f"ZED 카메라 초기화 완료.")
        self.zed_image = sl.Mat()
        self.thread = threading.Thread(target=self._worker, daemon=True)
    
    def _worker(self):
        runtime = sl.RuntimeParameters()
        while True:
            if self.zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                self.zed.retrieve_image(self.zed_image, sl.VIEW.LEFT)
                frame = self.zed_image.get_data()
                self.queue.put((timestamp, frame))
    
    def stop(self):
        self.zed.close()
        print("ZED 카메라가 닫혔습니다.")


def main():
    cameras = {
        'left': ArduCam(CAM_CONFIG['left_cam'], CAM_CONFIG),
        'right': ArduCam(CAM_CONFIG['right_cam'], CAM_CONFIG),
        'zed': ZedCam(ZED_CONFIG)
    }
    
    # 모든 카메라 스레드 시작
    for cam in cameras.values():
        cam.start()

    SYNC_THRESHOLD_SEC = (1 / CAM_CONFIG['fps'] / 2)  # e.g., (1 / 30 / 2) = 약 0.0167초
    GUI_DEBUG = False
    exit_app = False

    try:
        while not exit_app:
            # 모든 카메라에서 프레임 읽기
            ts_left, frame_l = cameras['left'].read()
            ts_right, frame_r = cameras['right'].read()
            ts_zed, frame_z = cameras['zed'].read()

            # ZED 타임스탬프는 나노초 단위일 수 있으므로 단위를 통일해야 함 (OpenCV의 time.time()은 초 단위)
            ts_zed_sec = ts_zed.get_nanoseconds() * 1e-9
            
            # 타임스탬프 비교 (모든 카메라의 프레임이 허용 오차 내에 있는지 확인)
            diff_zl = abs(ts_zed_sec - ts_left)
            diff_zr = abs(ts_zed_sec - ts_right)
            if diff_zl < SYNC_THRESHOLD_SEC and diff_zr < SYNC_THRESHOLD_SEC:
                print(f"Synced frames found! Timestamps: ZED={ts_zed_sec:.3f}, L={ts_left:.3f}, R={ts_right:.3f}")
                
                # sync 이미지를 기반으로 다음 로직 처리리...
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
                print("Frames out of sync, skipping... | Time diffs: Z-L={diff_zl:.4f}s, Z-R={diff_zr:.4f}s")

    except KeyboardInterrupt:
        print("KeyboardInterrupt 발생. 프로그램을 종료합니다.")
        exit_app = True

    finally:
        # 모든 카메라 리소스 해제
        for cam in cameras.values():
            cam.stop()
        if GUI_DEBUG:
            cv2.destroyAllWindows()
        print("모든 리소스가 안전하게 해제되었습니다.")

    print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()