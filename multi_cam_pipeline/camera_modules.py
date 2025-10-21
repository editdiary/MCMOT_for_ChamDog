# 카메라 관련 클래스들 모은 파일

import cv2
#import pyzed.sl as sl
import threading
import time
from queue import Queue, Empty

# 설정 파일 및 글로벌 종료 이벤트 임포트
from config import CAM_CONFIG, ZED_CONFIG, CAMERA_TIMEOUT_SEC


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

        # [New!] 실제 프레임 촬영을 위한 코드 임시 추가
        frame_count = 0
        last_log_time = time.time()
        
        while not self.shutdown_event.is_set():
            try:
                if self.cap.grab():
                    consecutive_failures = 0
                    #system_timestamp = time.time()
                    system_timestamp = cv2.getTickCount() / cv2.getTickFrequency()
                    ret, frame = self.cap.retrieve()
                    if ret:
                        # [New!] 실제 프레임 촬영을 위한 코드 임시 추가
                        frame_count += 1
                        current_time = time.time()
                        if current_time - last_log_time >= 1.0:
                            print(f"ArduCam '{self.device_path}' 프레임 캡처 성공: {frame_count}프레임")
                            last_log_time = current_time
                            frame_count = 0

                        if self.queue.full():
                            try: self.queue.get_nowait()
                            except Empty: pass
                        self.queue.put((system_timestamp, frame))
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
    """ZED를 제어하는 클래스(V4L2)"""
    def __init__(self, device_path, config, buffer_size=1):
        super().__init__(buffer_size)
        self.device_path = device_path
        self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise IOError(f"ZED을 열 수 없습니다: {self.device_path}")

        self.w = config['zed_width']
        self.h = config['zed_height']
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['zed_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['zed_height'])
        self.cap.set(cv2.CAP_PROP_FPS, config['zed_fps'])

        print(f"ZED '{self.device_path}' 초기화 완료.")
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
    
    def _worker(self):
        consecutive_failures = 0
        max_consecutive_failures = int(CAMERA_TIMEOUT_SEC * ZED_CONFIG['zed_fps'])  # 시간 기반 계산

        # [New!] 실제 프레임 촬영을 위한 코드 임시 추가
        frame_count = 0
        last_log_time = time.time()

        # V4L2로 촬영한 ZED 스테레오 이미지를 분할하기 위한 너비
        crop_width = self.w // 2
        
        while not self.shutdown_event.is_set():
            try:
                if self.cap.grab():
                    consecutive_failures = 0
                    #system_timestamp = time.time()
                    system_timestamp = cv2.getTickCount() / cv2.getTickFrequency()

                    ret, stereo_frame = self.cap.retrieve()
                    if ret:
                        # [New!] 실제 프레임 촬영을 위한 코드 임시 추가
                        frame_count += 1
                        current_time = time.time()
                        if current_time - last_log_time >= 1.0:
                            print(f"ZED '{self.device_path}' 프레임 캡처 성공: {frame_count}프레임")
                            last_log_time = current_time
                            frame_count = 0

                        # V4L2로 촬영한 ZED는 스테레오 프레임이므로 좌우 프레임으로 분리
                        try:
                            frame_zl = stereo_frame[:, 0:crop_width]
                            frame_zr = stereo_frame[:, crop_width:self.w]
                        except Exception as e:
                            print(f"ZED '{self.device_path}' 스테레오 프레임 자르기 실패: {e}")
                            continue

                        if self.queue.full():
                            try: self.queue.get_nowait()
                            except Empty: pass
                        self.queue.put((system_timestamp, (frame_zl.copy(), frame_zr.copy())))
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"ZED '{self.device_path}' 연결 실패가 {CAMERA_TIMEOUT_SEC}초 동안 연속 발생했습니다. 스레드를 종료합니다.")
                        break

            except Exception as e:
                if self.shutdown_event.is_set():
                    break
                else:
                    consecutive_failures += 1
                    print(f"ZED '{self.device_path}' worker 오류 (연속 {consecutive_failures}회): {e}")
                    if consecutive_failures >= max_consecutive_failures:
                        break

    def stop(self):
        self.cap.release()
        print(f"ZED '{self.device_path}'가 닫혔습니다.")