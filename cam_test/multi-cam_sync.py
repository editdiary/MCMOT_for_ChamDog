import cv2
import pyzed.sl as sl
import threading
import time
from queue import Queue, Empty

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

# [New] 각 카메라가 frame을 저장할 버퍼(Queue)의 크기
BUFFER_SIZE = 10

def arducam_worker(cap, q):
    """Arducam 프레임을 읽어 큐에 넣는 워커 함수"""
    while True:
        #ret, frame = cap.read()
        if cap.grab():
            # grab 직후 정확한 촬영 시점 timestamp 측정
            system_timestamp = time.time()
            # retrieve()로 실제 frame data 가져오기
            ret, frame = cap.retrieve()
            if ret:     # retrieve 성공 확인
                # 큐가 가득 찬 경우 가장 오래된 프레임 제거
                if q.full():
                    try: q.get_nowait()
                    except: pass
                q.put((system_timestamp, frame))

def zed_worker(zed, q):
    """ZED 카메라 프레임을 읽어 큐에 넣는 워커 함수"""
    runtime = sl.RuntimeParameters()
    zed_image = sl.Mat()
    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # ZED SDK는 이미지 캡처 시점의 타임스탬프를 제공
            system_timestamp = time.time()
            #camera_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)

            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            frame = zed_image.get_data()

            # 큐가 가득 찬 경우 가장 오래된 프레임 제거
            if q.full():
                try: q.get_nowait()
                except: pass
            q.put((system_timestamp, frame))

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

# [New] Slave Queue에서 Master Timestamp와 가장 가까운 frame을 찾는 함수
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
    # Arducam 카메라 초기화
    left_cap, right_cap = initialize_arducam_cameras()

    # ZED 카메라 초기화
    zed, runtime, zed_image = initialize_zed_camera()

    # 각 카메라의 프레임을 담을 큐 생성 (버퍼 크기 적용)
    q_zed = Queue(maxsize=BUFFER_SIZE)
    q_left = Queue(maxsize=BUFFER_SIZE)
    q_right = Queue(maxsize=BUFFER_SIZE)

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

    # ---- 카메라 안정화를 위해 워밍업 과정 추가 ----
    WARMUP_FRAMES = CAM_CONFIG['fps'] * 2
    print(f"카메라 안정화를 위해 {WARMUP_FRAMES} 프레임 동안 워밍업을 시작합니다...")
    for _ in range(WARMUP_FRAMES):
        try:
            q_zed.get(timeout=1)
            q_left.get(timeout=1)
            q_right.get(timeout=1)
        except:
            pass
    print("워밍업 완료.")
    # ------------------------------------------

    # 안정적인 매칭을 위해, 베스트 매치의 시간 차이도 일정 수준 이하여야 함
    MAX_ALLOWED_DIFF_SEC = 0.04  # 동기화 허용 오차
    GUI_DEBUG = True
    exit_app = False

    try:
        while not exit_app:
            # 1. 마스터 카메라(ZED)에서 최신 프레임을 가져옴
            try:
                master_ts, master_frame = q_zed.get(timeout=1)
            except:
                print("마스터 카메라(ZED)에서 1초 이상 프레임이 없습니다. 확인이 필요합니다.")
                continue
        
            # 2. 각 슬레이브 카메라의 버퍼에서 최적의 짝을 찾음
            left_match = find_best_match_in_queue(q_left, master_ts)
            right_match = find_best_match_in_queue(q_right, master_ts)

            # 3. 모든 카메라에서 성공적으로 짝을 찾았는지 확인
            if left_match is None or right_match is None:
                print("슬레이브 카메라(Arducam) 버퍼가 비어있습니다. 잠시 기다립니다...")
                continue

            ts_left, frame_l = left_match
            ts_right, frame_r = right_match

            # 4. 찾아낸 베스트 매치도 허용 오차 내에 있는지 최종 확인
            diff_zl = abs(master_ts - ts_left)
            diff_zr = abs(master_ts - ts_right)
            
            if diff_zl < MAX_ALLOWED_DIFF_SEC and diff_zr < MAX_ALLOWED_DIFF_SEC:
                print(f"Synced! Time Diffs: Z-L={diff_zl:.4f}, Z-R={diff_zr:.4f}")

                # sync 이미지를 기반으로 다음 로직 처리...
                # ZED는 4채널(RGBA) 이미지를 반환하므로, RGB로 변환
                frame_z_rgb = cv2.cvtColor(master_frame, cv2.COLOR_RGBA2RGB)

                # GUI 디버깅 할 때 사용
                if GUI_DEBUG:
                    cv2.imshow("ZED Camera", frame_z_rgb)
                    cv2.imshow("Arducam LEFT", frame_l)
                    cv2.imshow("Arducam RIGHT", frame_r)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        exit_app = True
                        break
            else:
                print(f"Frames out of sync, skipping... | Time diffs: Z-L={diff_zl:.4f}s, Z-R={diff_zr:.4f}s")

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