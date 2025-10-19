# 카메라 촬영 main 함수

import threading
import cv2
from queue import Empty, Queue, Full
import time

# 분리된 모듈들 입포트
from config import (
    CAM_CONFIG, ZED_CONFIG, BUFFER_SIZE,
    MAIN_LOOP_TIMEOUT, MAX_ALLOWED_DIFF_SEC,
    GUI_DEBUG, IMG_SAVE
)
from camera_modules import ArduCam, ZedCam
from sync_utils import (
    save_sync_frames, print_camera_settings
    # find_best_match_in_queue
)

# 글로벌 종료 플래그
shutdown_event = threading.Event()

# [New!] 후처리(추론)를 위한 큐
INFERENCE_BUFFER_SIZE = 5   # 추론 스레드가 밀릴 경우를 대비한 버퍼
inference_queue = Queue(maxsize=INFERENCE_BUFFER_SIZE)

# [New!] YOLO 추론을 담당할 소비자(Consumer) 스레드 함수
def inference_worker(queue, shutdown_event):
    """
    inference_queue에서 동기화된 프레임 세트를 가져와
    YOLO 추론, 저장, GUI 표시 등 '느린' 작업을 처리한다.
    """
    print("[Inference Thread] 추론 스레드 시작...")
    while not shutdown_event.is_set():
        try:
            # 1. 동기화된 프레임 세트를 큐에서 가져옴 (타임아웃 1초)
            (ts, frame_z, frame_l, frame_r) = queue.get(timeout=1.0)

            # 2. 여기에 YOLO 추론 코드를 넣는다 (지금은 100ms 작업 시뮬레이션)
            print(f"[Inference Thread] {ts:.3f} 프레임 세트 추론 시작...")
            time.sleep(0.1) # 100ms (0.1초)가 걸리는 작업 시뮬레이션
            print(f"[Inference Thread] ...추론 완료.")

            # (YOLO 결과물이 적용된 프레임이라고 가정)
            # 3. 이미지 저장 (느린 작업)
            if IMG_SAVE:
                if save_sync_frames(frame_z, frame_l, frame_r):
                    print("이미지 저장 완료")

            # 4. GUI 표시 (느린 작업)
            if GUI_DEBUG:
                zed_display = cv2.cvtColor(frame_z, cv2.COLOR_RGBA2RGB)
                cv2.imshow("ZED Camera", zed_display)
                cv2.imshow("Arducam LEFT", frame_l)
                cv2.imshow("Arducam RIGHT", frame_r)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    shutdown_event.set()
        except Empty:
            continue    # 큐가 1초 동안 비어있으면 그냥 계속 대기
        except Exception as e:
            if shutdown_event.is_set():
                break
            print(f"[Inference Thread] 오류 발생: {e}")
    
    # 종료 시 GUI 창 닫기
    if GUI_DEBUG:
        cv2.destroyAllWindows()
    print("[Inference Thread] 추론 스레드 종료.")

def main():
    try:
        cameras = {
            'left': ArduCam(CAM_CONFIG['left_cam'], CAM_CONFIG, buffer_size=BUFFER_SIZE),
            'right': ArduCam(CAM_CONFIG['right_cam'], CAM_CONFIG, buffer_size=BUFFER_SIZE),
            'zed': ZedCam(ZED_CONFIG, buffer_size=BUFFER_SIZE)
        }
    except IOError as e:
        print(f"카메라 초기화 중 오류: {e}")
        return
    
    # 모든 카메라 스레드 시작 (shutdown_event 전달)
    for cam in cameras.values():
        cam.start(shutdown_event)
    
    # [New!] 추론 스레드 시작
    yolo_thread = threading.Thread(
        target=inference_worker,
        args=(inference_queue, shutdown_event),
        daemon=True
    )
    yolo_thread.start()

    print("\n=== 설정한 카메라 파라미터 요약 ===")
    print(f"ZED Camera:    {ZED_CONFIG['resolution'].name} @ {ZED_CONFIG['fps']}fps")
    print(f"ArduCam Left:  {CAM_CONFIG['frame_width']}x{CAM_CONFIG['frame_height']} @ {CAM_CONFIG['fps']}fps")
    print(f"ArduCam Right: {CAM_CONFIG['frame_width']}x{CAM_CONFIG['frame_height']} @ {CAM_CONFIG['fps']}fps")
    print(f"버퍼 크기: {BUFFER_SIZE} 프레임")
    print("========================\n")
    
    print_camera_settings(cameras)    # 실제 시스템에 설정된 값들을 간단히 출력

    # ---- 카메라 안정화를 위해 워밍업 과정 추가 ----
    WARMUP_FRAMES = CAM_CONFIG['fps'] * 2
    print(f"카메라 안정화를 위해 {WARMUP_FRAMES} 프레임 동안 워밍업을 시작합니다...")
    try:
        for _ in range(WARMUP_FRAMES):
            for cam in cameras.values():
                cam.read(timeout=1.0)
    except Empty:
        print("워밍업 중 카메라 대기 시간 초과. 큐가 비어있습니다. 프로그램을 종료합니다.")
        shutdown_event.set()
        return
        
    print("워밍업 완료.")
    # ------------------------------------------

    exit_app = False

    # [수정] 3개의 큐에서 프레임을 하나씩 미리 가져옴
    try:
        master_ts, master_frame = cameras['zed'].read(timeout=MAIN_LOOP_TIMEOUT)
        left_ts, left_frame = cameras['left'].read(timeout=MAIN_LOOP_TIMEOUT)
        right_ts, right_frame = cameras['right'].read(timeout=MAIN_LOOP_TIMEOUT)
    except Empty:
        print("초기 프레임 로드 실패. 카메라 연결을 확인하세요.")
        shutdown_event.set()
        exit_app = True

    try:
        while not exit_app and not shutdown_event.is_set():
            # [새 로직] 타임스탬프 리스트와 최대/최소값 계산
            ts_list = [master_ts, left_ts, right_ts]
            ts_min = min(ts_list)
            ts_max = max(ts_list)
            time_diff = ts_max - ts_min

            # [새 로직] 1. 동기화 성공 케이스
            if time_diff < MAX_ALLOWED_DIFF_SEC:
                print(f"[Sync Thread] Synced! Time Diff: {time_diff:.4f}s")

                # [새 로직] 동기화 성공 시, 프레임 세트를 'inference_queue'에 넣음
                try:
                    inference_queue.put_nowait(
                        (master_ts, master_frame, left_frame, right_frame)
                    )
                except Full:
                    # Queue가 가득 찼을 경우 (YOLO 추론이 너무 밀림)
                    # 경고를 출력하고 이 프레임 세트는 버림
                    print("[Sync Thread] 경고: 추론 큐가 가득 찼습니다. 프레임 세트를 버립니다.")
                
                # 동기화 성공 시, 3개 카메라 모두에서 새 프레임을 가져옴
                try:
                    master_ts, master_frame = cameras['zed'].read(timeout=MAIN_LOOP_TIMEOUT)
                    left_ts, left_frame = cameras['left'].read(timeout=MAIN_LOOP_TIMEOUT)
                    right_ts, right_frame = cameras['right'].read(timeout=MAIN_LOOP_TIMEOUT)
                except Empty:
                    print("[Sync Thread] 동기화 처리 중 큐가 비었습니다. 루프를 종료합니다.")
                    break
            
            # 2. 동기화 실패 케이스 (가장 오래된 프레임 버리기)
            else:
                oldest_cam_index = ts_list.index(ts_min)
                
                try:
                    if oldest_cam_index == 0: # ZED가 가장 오래됨
                        master_ts, master_frame = cameras['zed'].read(timeout=MAIN_LOOP_TIMEOUT)
                    elif oldest_cam_index == 1: # Left가 가장 오래됨
                        left_ts, left_frame = cameras['left'].read(timeout=MAIN_LOOP_TIMEOUT)
                    else: # Right가 가장 오래됨
                        right_ts, right_frame = cameras['right'].read(timeout=MAIN_LOOP_TIMEOUT)
                except Empty:
                    print("동기화 대기 중 큐가 비었습니다. 루프를 종료합니다.")
                    break # while 루프 종료

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt 발생. 프로그램을 종료합니다.")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
    finally:
        shutdown_event.set()
        for cam in cameras.values():
            cam.stop()
        print("[Sync Thread] 모든 리소스가 안전하게 해제되었습니다.")
    print("[Sync Thread] 프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()