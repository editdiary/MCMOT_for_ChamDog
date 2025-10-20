import cv2
import threading
import time
from queue import Queue, Empty

# --- 설정 값들은 상단에 유지 ---
# 카메라 속성 정의
# v4l2-ctl로 확인한 지원하는 값으로 설정해야 한다.
    # 명령어: v4l2-ctl -d /dev/arducam_left --list-formats-ext
CAM_CONFIG = {
    "left_cam": "/dev/arducam_left", "right_cam": "/dev/arducam_right",
    "frame_width": 800, "frame_height": 600, "fps": 15,
    "fourcc": cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')    # MJPEG 코덱 설정 (압축 포맷 사용 시)
}
ZED_CONFIG = {
    "zed": "/dev/zed",
    "zed_width": 4416, "zed_height": 1242, "zed_fps": 15
}
# [New] 각 카메라가 frame을 저장할 버퍼(Queue)의 크기
BUFFER_SIZE = 5

# 카메라 연결 실패 처리 설정
CAMERA_TIMEOUT_SEC = 2.0  # 2초 동안 연속 실패 시 스레드 종료

# 메인 루프 타임아웃 설정
MAIN_LOOP_TIMEOUT = 5.0  # 5초 - 메인 루프에서 프레임 대기 시간

# 글로벌 종료 플래그
shutdown_event = threading.Event()

class Camera:
    """모든 카메라 클래스가 상속할 기본 클래스 (향후 확장용)"""
    def __init__(self, buffer_size=1):
        self.thread = None
        self.queue = Queue(maxsize=buffer_size)
    
    def start(self):
        """캡처 스레드 시작"""
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
        frame_count = 0
        last_log_time = time.time()
        while not shutdown_event.is_set():
            try:
                if self.cap.grab():
                    consecutive_failures = 0
                    system_timestamp = time.time()
                    ret, frame = self.cap.retrieve()
                    if ret:
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
                        break  # 스레드 종료

            except Exception as e:
                # KeyboardInterrupt 관련 예외는 조용히 처리
                if shutdown_event.is_set():
                    break   # 종료 중이면 조용히 종료
                else:
                    consecutive_failures += 1
                    print(f"ArduCam '{self.device_path}' worker 오류 (연속 {consecutive_failures}회): {e}")
                    if consecutive_failures >= max_consecutive_failures:
                        break  # 스레드 종료
    
    def stop(self):
        self.cap.release()
        print(f"Arducam '{self.device_path}'가 닫혔습니다.")

class ZedCam(Camera):
    """ZED를 제어하는 클래스"""
    def __init__(self, device_path, config, buffer_size=1):
        super().__init__(buffer_size)
        self.device_path = device_path
        self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise IOError(f"ZED를를 열 수 없습니다: {self.device_path}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['zed_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['zed_height'])
        self.cap.set(cv2.CAP_PROP_FPS, config['zed_fps'])

        print(f"ZED '{self.device_path}' 초기화 완료.")
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
    
    def _worker(self):
        consecutive_failures = 0
        max_consecutive_failures = int(CAMERA_TIMEOUT_SEC * ZED_CONFIG['zed_fps'])
        frame_count = 0
        last_log_time = time.time()
        while not shutdown_event.is_set():
            try:
                if self.cap.grab():
                    consecutive_failures = 0
                    system_timestamp = time.time()
                    ret, frame = self.cap.retrieve()
                    if ret:
                        frame_count += 1
                        current_time = time.time()
                        if current_time - last_log_time >= 1.0:
                            print(f"ZED '{self.device_path}' 프레임 캡처 성공: {frame_count}프레임")
                            last_log_time = current_time
                            frame_count = 0

                        if self.queue.full():
                            try: self.queue.get_nowait()
                            except Empty: pass
                        self.queue.put((system_timestamp, frame))

                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"ZED '{self.device_path}' 연결 실패가 {CAMERA_TIMEOUT_SEC}초 동안 연속 발생했습니다. 스레드를 종료합니다.")
                        break  # 스레드 종료

            except Exception as e:
                # KeyboardInterrupt 관련 예외는 조용히 처리
                if shutdown_event.is_set():
                    break   # 종료 중이면 조용히 종료
                else:
                    consecutive_failures += 1
                    print(f"ZED '{self.device_path}' worker 오류 (연속 {consecutive_failures}회): {e}")
                    if consecutive_failures >= max_consecutive_failures:
                        break  # 스레드 종료
    
    def stop(self):
        self.cap.release()
        print(f"ZED '{self.device_path}'가 닫혔습니다.")

def save_sync_frames(master_frame, frame_l, frame_r):
    """동기화된 프레임들을 저장 (리소스 부하 테스트용)"""
    try:
        # ZED 프레임 저장 (RGBA -> RGB 변환)
        cv2.imwrite("test_zed.jpg", master_frame)
        
        # Arducam 프레임들 저장
        cv2.imwrite("test_left.jpg", frame_l)
        cv2.imwrite("test_right.jpg", frame_r)
        
        return True
    except Exception as e:
        print(f"이미지 저장 실패: {e}")
        return False

def print_camera_settings(cameras):
    """모든 카메라의 실제 설정값을 간단히 출력"""
    print("=== 실제 카메라 설정값 요약 ===")
    
    # ArduCam Left
    left_cap = cameras['left'].cap
    print(f"ArduCam Left:  {int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(left_cap.get(cv2.CAP_PROP_FPS))}fps")
    
    # ArduCam Right  
    right_cap = cameras['right'].cap
    print(f"ArduCam Right: {int(right_cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(right_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(right_cap.get(cv2.CAP_PROP_FPS))}fps")
    
    # ZED Camera
    zed_cam = cameras['zed'].cap
    print(f"ZED Camera:    {int(zed_cam.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(zed_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(zed_cam.get(cv2.CAP_PROP_FPS))}fps")
    
    print("==========================\n")

# Slave Queue에서 Master Timestamp와 가장 가까운 frame을 찾는 함수
def find_best_match_in_queue(slave_queue, master_ts):
    if slave_queue.empty():
        return None

    buffer_list = []
    # 큐에서 모든 아이템을 안전하게 가져오기
    while not slave_queue.empty():
        try: 
            item = slave_queue.get_nowait()
            buffer_list.append(item)
        except Empty: 
            break
    
    if not buffer_list:
        return None
    
    # 가장 가까운 타임스탬프를 가진 아이템 찾기
    best_match = min(buffer_list, key=lambda x: abs(x[0] - master_ts))
    
    # 나머지 아이템들을 다시 큐에 넣기 (최신 것만 유지)
    for item in buffer_list:
        if item != best_match:  # 선택된 것 제외하고 나머지 다시 넣기
            try:
                slave_queue.put_nowait(item)
            except:
                # 큐가 가득 찬 경우 가장 오래된 것 버리기
                break
    
    return best_match


def main():
    try:
        cameras = {
            'left': ArduCam(CAM_CONFIG['left_cam'], CAM_CONFIG, buffer_size=BUFFER_SIZE),
            'right': ArduCam(CAM_CONFIG['right_cam'], CAM_CONFIG, buffer_size=BUFFER_SIZE),
            'zed': ZedCam(ZED_CONFIG['zed'], ZED_CONFIG, buffer_size=BUFFER_SIZE)
        }
    except IOError as e:
        print(e)
        return
    
    # 모든 카메라 스레드 시작
    for cam in cameras.values():
        cam.start()

    # 모든 카메라 스레드 시작 후, 워밍업 전에 추가
    print("\n=== 설정한 카메라 파라미터 요약 ===")
    print(f"ZED Camera:    {ZED_CONFIG['zed_width']}x{ZED_CONFIG['zed_height']} @ {ZED_CONFIG['zed_fps']}fps")
    print(f"ArduCam Left:  {CAM_CONFIG['frame_width']}x{CAM_CONFIG['frame_height']} @ {CAM_CONFIG['fps']}fps")
    print(f"ArduCam Right: {CAM_CONFIG['frame_width']}x{CAM_CONFIG['frame_height']} @ {CAM_CONFIG['fps']}fps")
    print(f"버퍼 크기: {BUFFER_SIZE} 프레임")
    print("========================\n")
    
    # 실제 설정값들 간단히 출력
    print_camera_settings(cameras)

    # ---- 카메라 안정화를 위해 워밍업 과정 추가 ----
    WARMUP_FRAMES = CAM_CONFIG['fps'] * 2
    print(f"카메라 안정화를 위해 {WARMUP_FRAMES} 프레임 동안 워밍업을 시작합니다...")
    for _ in range(WARMUP_FRAMES):
        for cam in cameras.values():
            cam.read()  # frame을 읽기만 하고 사용하지 않음
    print("워밍업 완료.")
    # ------------------------------------------

    # 안정적인 매칭을 위해, 베스트 매치의 시간 차이도 일정 수준 이하여야 함
    #MAX_ALLOWED_DIFF_SEC = (1 / CAM_CONFIG['fps'] / 2)  # e.g., (1 / 30 / 2) = 약 0.0167초
    MAX_ALLOWED_DIFF_SEC = (1 / CAM_CONFIG['fps']) * 3    # 약 3frame 차이까지 허용
    GUI_DEBUG = False
    IMG_SAVE = True
    exit_app = False

    try:
        while not exit_app:
            # 1. 마스터 카메라(ZED)에서 최신 프레임을 가져옴
            try:
                master_ts, master_frame = cameras['zed'].read(timeout=MAIN_LOOP_TIMEOUT)  # 50ms 타임아웃
            except Empty:
                # ZED 큐가 비어있으면 다음 루프로 (sleep 없이)
                continue
        
            # 2. 각 슬레이브 카메라의 버퍼에서 최적의 짝을 찾음
            left_match = find_best_match_in_queue(cameras['left'].queue, master_ts)
            right_match = find_best_match_in_queue(cameras['right'].queue, master_ts)

            # 3. 모든 카메라에서 성공적으로 짝을 찾았는지 확인
            if left_match is None or right_match is None:
                # 슬레이브 카메라 버퍼가 비어있으면 다음 루프로 (sleep 없이)
                continue

            ts_left, frame_l = left_match
            ts_right, frame_r = right_match

            # 4. 찾아낸 베스트 매치도 허용 오차 내에 있는지 최종 확인
            diff_zl = abs(master_ts - ts_left)
            diff_zr = abs(master_ts - ts_right)
            
            if diff_zl < MAX_ALLOWED_DIFF_SEC and diff_zr < MAX_ALLOWED_DIFF_SEC:
                print(f"Synced! Time Diffs: Z-L={diff_zl:.4f}, Z-R={diff_zr:.4f}")

                # 이미지 저장 기능 (리소스 부하 테스트용)
                if IMG_SAVE:
                    if save_sync_frames(master_frame, frame_l, frame_r):
                        print("이미지 저장 완료")
                    else:
                        print("이미지 저장 실패")

                # GUI 디버깅 할 때 사용
                if GUI_DEBUG:
                    cv2.imshow("ZED Camera", master_frame)
                    cv2.imshow("Arducam LEFT", frame_l)
                    cv2.imshow("Arducam RIGHT", frame_r)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        shutdown_event.set()    # Worker 스레드들에 종료 신호
                        exit_app = True         # 메인 루프 종료
                
            else:
                print(f"Frames out of sync, skipping... | Time diffs: Z-L={diff_zl:.4f}s, Z-R={diff_zr:.4f}s")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt 발생. 프로그램을 종료합니다.")
        shutdown_event.set()    # Worker 스레드들에 종료 신호
        exit_app = True         # 메인 루프 종료

    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
        shutdown_event.set()    # Worker 스레드들에 종료 신호
        exit_app = True         # 메인 루프 종료

    finally:
        # 모든 카메라 리소스 해제
        try:
            for cam in cameras.values():
                cam.stop()
        except Exception as e:
            print(f"카메라 종료 중 오류: {e}")
        
        if GUI_DEBUG:
            cv2.destroyAllWindows()
        print("모든 리소스가 안전하게 해제되었습니다.")

    print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()