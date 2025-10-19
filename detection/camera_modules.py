# 카메라 관련 클래스들 모은 파일

import cv2
import pyzed.sl as sl
import threading
import time
from queue import Queue, Empty

# 설정 파일 및 글로벌 종료 이벤트 임포트
from config import CAM_CONFIG, ZED_CONFIG, CAMERA_TIMEOUT_SEC

# 글로벌 종료 플래그 (main_sync.py에서 생성하여 전달받을 수 있지만,
# 모듈 수준에서 공유하는 것도 간단한 방법입니다. 여기서는 main에서 생성하는 것으로 변경하겠습니다.)
# shutdown_event = threading.Event() # -> main_sync.py로 이동

class Camera:
    """모든 카메라 클래스가 상속할 기본 클래스 (향후 확장용)"""
    def __init__(self, buffer_size=1):
        self.thread = None
        self.queue = Queue(maxsize=buffer_size)
        self.shutdown_event = None  # 외부에서 shutdown_event 전달받음
    
    def start(self, shutdown_event):
        """캡처 스레드 시작 (shutdown_event 전달)"""
        self.shutdown_event = shutdown_event
        self.thread.start()
    
    def read(self, timeout=None):
        """큐에서 최신 프레임 가져오기"""
        return self.queue.get(timeout=timeout)
    
    def stop(self):
        """카메라 리소스 해제 (자식 클래스에서 구현)"""
        raise NotImplementedError


class ArduCam(Camera):
    """Arducam을 제어하는 클래스"""
    def __init__(self, device_path, config, buffer_size=1):
        super().__init__(buffer_size)
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
        consecutive_failures = 0
        max_consecutive_failures = int(CAMERA_TIMEOUT_SEC * CAM_CONFIG['fps'])  # 시간 기반 계산
        
        while not self.shutdown_event.is_set():
            try:
                if self.cap.grab():                     # grab()으로 프레임 캡처
                    consecutive_failures = 0            # 성공 시 카운터 리셋
                    system_timestamp = time.time()      # grab 직후 정확한 촬영 시점 timestamp 측정
                    ret, frame = self.cap.retrieve()    # retrieve()로 실제 frame data 가져오기
                    if ret:                             # retrieve 성공 확인
                        if self.queue.full():           # 큐가 가득 찬 경우 가장 오래된 프레임 제거
                            try: self.queue.get_nowait()
                            except Empty: pass
                        self.queue.put((system_timestamp, frame))    # 큐에 프레임 추가
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"ArduCam '{self.device_path}' 연결 실패가 {CAMERA_TIMEOUT_SEC}초 동안 연속 발생했습니다. 스레드를 종료합니다.")
                        break

            except Exception as e:
                if self.shutdown_event.is_set():
                    break
                else:
                    consecutive_failures += 1
                    print(f"ArduCam '{self.device_path}' worker 오류 (연속 {consecutive_failures}회): {e}")
                    if consecutive_failures >= max_consecutive_failures:
                        break

    def stop(self):
        self.cap.release()
        print(f"Arducam '{self.device_path}'가 닫혔습니다.")


class ZedCam(Camera):
    """ZED 카메라를 제어하는 클래스"""
    def __init__(self, config, buffer_size=1):
        super().__init__(buffer_size)
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
        consecutive_failures = 0
        max_consecutive_failures = int(CAMERA_TIMEOUT_SEC * ZED_CONFIG['fps'])  # 시간 기반 계산
        
        while not self.shutdown_event.is_set():
            try:
                if self.zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                    consecutive_failures = 0
                    system_timestamp = time.time()
                    self.zed.retrieve_image(self.zed_image, sl.VIEW.LEFT)
                    frame = self.zed_image.get_data()
                    
                    if self.queue.full():
                        try: self.queue.get_nowait()
                        except Empty: pass
                    self.queue.put((system_timestamp, frame))
                else:
                    consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"ZED 카메라 연결 실패가 {CAMERA_TIMEOUT_SEC}초 동안 연속 발생했습니다. 스레드를 종료합니다.")
                    break

            except Exception as e:
                if self.shutdown_event.is_set():
                    break
                else:
                    consecutive_failures += 1
                    print(f"ZED worker 오류 (연속 {consecutive_failures}회): {e}")
                    if consecutive_failures >= max_consecutive_failures:
                        break

    def stop(self):
        self.zed.close()
        print("ZED 카메라가 닫혔습니다.")