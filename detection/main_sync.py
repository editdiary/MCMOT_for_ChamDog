# 카메라 촬영 main 함수

import threading
import cv2
from queue import Empty, Queue, Full
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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

def find_pipe_line(frame_z):
    """
    ZED 이미지에서 수직선을 찾아 추적점의 좌표와 결과 데이터를 반환하는 함수
    """

   # [추가] 이미지 유효성 검사
    if frame_z is None or frame_z.size == 0:
        print("find_pipe_line: 입력 이미지가 비어있습니다.")
        return None, None, None, None
    
    # 0. roi 영역 설정
    # ROI(관심 영역) 좌표
    roi_x, roi_y, roi_w, roi_h = 630, 350, 200, 300
    roi_coords = (roi_x, roi_y, roi_w, roi_h)   # 좌표 튜블 저장

    # ROI(관심 영역) 자르기
    roi = frame_z[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # 1. 이미지 전처리
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(equalized, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 2. 허프 변환으로 수직선 필터링
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=30, maxLineGap=30)
    
    pipe_line_x_coords = []
    vertical_lines = []     # 추후 시각화를 위해 라인 자체를 저장
    track_point = None

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.rad2deg(np.arctan2(y2 - y1, x2 - x1))) if x2 - x1 != 0 else 90.0
            if angle > 70:
                vertical_lines.append(line[0])
                pipe_line_x_coords.extend([x1, x2])

    # 3. 추적점 계산
    if pipe_line_x_coords:
        avg_x = int(np.mean(pipe_line_x_coords))
        center_y = roi.shape[0] // 2
        track_point = (avg_x, center_y)

    return roi, roi_coords, vertical_lines, track_point

# [New!] 파이프 탐지 및 YOLO 추론을 담당할 소비자(Consumer) 스레드 함수
def inference_worker(queue, shutdown_event):
    """
    inference_queue에서 동기화된 프레임 세트를 가져와
    CPU 작업(Pipe 탐지)과 GPU 작업(YOLO 추론)을 병렬로 처리
    """
    print("[Inference Thread] 추론 스레드 시작...")

    USE_ZED_RIGHT_FOR_PIPE = False  # True로 설정하면 frame_zr을 사용하여 파이프 탐지

    # [New!] CPU 작업을 처리할 1개의 워커 스레드를 가진 thread pool 생성
    # (CPU 코어 수에 맞춰 조절 가능)
    cpu_thread_pool = ThreadPoolExecutor(max_workers=2)

    while not shutdown_event.is_set():
        try:
            # 1. 동기화된 프레임 세트를 큐에서 가져옴 (타임아웃 1초)
            (ts, (frame_zl, frame_zr), frame_l, frame_r) = queue.get(timeout=1.0)

            # [추가] 프레임 유효성 검사 (큰 문제 없으면 뺴도 될 듯)
            if frame_zl is None or frame_zl.size == 0:
                print("[Inference Thread] ZED Left 프레임이 비어있습니다.")
                continue
            if frame_zr is None or frame_zr.size == 0:
                print("[Inference Thread] ZED Right 프레임이 비어있습니다.")
                continue
            if frame_l is None or frame_l.size == 0:
                print("[Inference Thread] Left 프레임이 비어있습니다.")
                continue
            if frame_r is None or frame_r.size == 0:
                print("[Inference Thread] Right 프레임이 비어있습니다.")
                continue

            # ----- [핵심] 병렬 처리 -----
            # 2. (병렬 작업 1) CPU 스레드에 파이프 탐지 작업을 비동기로 "제출"
            # 파이프 탐지를 위한 이미지 선택
            if USE_ZED_RIGHT_FOR_PIPE:
                PIPE_IMAGE = frame_zr
            else:
                PIPE_IMAGE = frame_zl

            pipe_future = cpu_thread_pool.submit(find_pipe_line, PIPE_IMAGE)

            # 3. (병렬 작업 2) 메인 스레드(GPU 담당)는 YOLO 추론을 바로 실행
            # GPU 작업을 위해 Left/Right 이미지를 '배치'로 묶음
            # yolo_batch = [frame_l, frame_r]

            # GPU에서 "배치"를 한 번에 처리
            # (가상의 YOLO 함수 호출)
            # yolo_results_batch = yolo_model(yolo_batch)
            
            # 지금은 시뮬레이션
            time.sleep(0.05) # 50ms (0.05초)가 걸리는 GPU 작업 시뮬레이션

            # (결과 분리)
            # yolo_results_l = yolo_results_batch[0]
            # yolo_results_r = yolo_results_batch[1]

            # 4. CPU 스레드의 파이프 탐지 결과가 끝날 때까지 "대기"
            # (YOLO가 50ms, 파이프 탐지가 15ms 걸렸다면,
            #  YOLO가 끝난 시점엔 이미 파이프 탐지는 끝났으므로, .result()는 즉시 반환됨)
            roi_view, roi_coords, vertical_lines, track_point = pipe_future.result(timeout=1.0)

            # (여기서 yolo_resultsl, yolo_resultsr, detection_result를
            #  모두 사용하여 이미지에 그리거나 저장)

            # 5. 결과 취합 및 시각화 준비
            if (IMG_SAVE or GUI_DEBUG):
                # 저장 또는 GUI 표시가 필요하면, 원본의 복사본을 만듦
                annotated_image = PIPE_IMAGE.copy()

                if roi_coords:
                    # (1) ROI 영역 좌표 가져오기
                    (roi_x, roi_y, roi_w, roi_h) = roi_coords

                    # (2) Pipe를 탐지한 원본 ZED 프레임에 ROI 영역 그리기 (빨간색 네모)
                    cv2.rectangle(annotated_image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)

                    # (3) 탐지된 라인 그리기 (좌표 변환 필요)
                    if vertical_lines:
                        for x1, y1, x2, y2 in vertical_lines:
                            global_x1 = x1 + roi_x
                            global_y1 = y1 + roi_y
                            global_x2 = x2 + roi_x
                            global_y2 = y2 + roi_y
                            cv2.line(annotated_image, (global_x1, global_y1), (global_x2, global_y2), (0, 255, 0), 2)

                    # (4) 트랙 포인트 그리기
                    if track_point:
                        tp_x_roi, tp_y_roi = track_point    # ROI 기준 좌표
                        # 전체 프레임 좌표로 변환
                        tp_x_global = tp_x_roi + roi_x
                        tp_y_global = tp_y_roi + roi_y

                        print(f"Track Point (Global): (x={tp_x_global}, y={tp_y_global})")
                        cv2.circle(annotated_image, (tp_x_global, tp_y_global), 5, (0, 0, 255), -1)
                        cv2.putText(annotated_image, "Track Point", (tp_x_global - 55, tp_y_global - 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    else:
                        print("Track Point: Not found")
            
            else:
                # 시각화를 안 할 경우 원본 zl 프레임을 사용
                annotated_image = None

            # 6. 이미지 저장 (느린 작업)
            if IMG_SAVE:
                if save_sync_frames((frame_zl, frame_zr), frame_l, frame_r, frame_roi=annotated_image):
                    print("이미지 저장 완료")

            # 7. GUI 표시 (느린 작업)
            if GUI_DEBUG:
                cv2.imshow("ZED Camera Left", frame_zl)
                #cv2.imshow("ZED Camera Right", frame_zr)
                cv2.imshow("Arducam LEFT", frame_l)
                cv2.imshow("Arducam RIGHT", frame_r)

                # 시각화된 ROI가 있다면 별도 창으로 표시
                if annotated_image is not None:
                    cv2.imshow("ROI View", annotated_image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    shutdown_event.set()

        except Empty:
            continue    # 큐가 1초 동안 비어있으면 그냥 계속 대기
        except Exception as e:
            if shutdown_event.is_set():
                break
            print(f"[Inference Thread] 오류 발생: {e}")
    
    cpu_thread_pool.shutdown(wait=True)     # 스레드 풀 종료 추가
    
    # 종료 시 GUI 창 닫기
    if GUI_DEBUG:
        cv2.destroyAllWindows()
    print("[Inference Thread] 추론 스레드 종료.")

def main():
    try:
        cameras = {
            'left': ArduCam(CAM_CONFIG['left_cam'], CAM_CONFIG, buffer_size=BUFFER_SIZE),
            'right': ArduCam(CAM_CONFIG['right_cam'], CAM_CONFIG, buffer_size=BUFFER_SIZE),
            'zed': ZedCam(ZED_CONFIG['zed_cam'], ZED_CONFIG, buffer_size=BUFFER_SIZE)
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

    ## 이것도 print_camera_settings 함수에 통합해서 argument로 조절하는게 좋을듯
    print("\n=== 설정한 카메라 파라미터 요약 ===")
    print(f"ZED Camera:    {ZED_CONFIG['zed_width']}x{ZED_CONFIG['zed_height']} @ {ZED_CONFIG['zed_fps']}fps")
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
                cam.read(timeout=2.0)
    except Empty:
        print("워밍업 중 카메라 대기 시간 초과. 큐가 비어있습니다. 프로그램을 종료합니다.")
        shutdown_event.set()
        return
        
    print("워밍업 완료.")
    # ------------------------------------------

    exit_app = False

    # [수정] 3개의 큐에서 프레임을 하나씩 미리 가져옴
    try:
        zed_ts, (frame_zl, frame_zr) = cameras['zed'].read(timeout=MAIN_LOOP_TIMEOUT)
        left_ts, frame_l = cameras['left'].read(timeout=MAIN_LOOP_TIMEOUT)
        right_ts, frame_r = cameras['right'].read(timeout=MAIN_LOOP_TIMEOUT)
    except Empty:
        print("초기 프레임 로드 실패. 카메라 연결을 확인하세요.")
        shutdown_event.set()
        exit_app = True

    try:
        while not exit_app and not shutdown_event.is_set():
            # [새 로직] 타임스탬프 리스트와 최대/최소값 계산
            ts_list = [zed_ts, left_ts, right_ts]
            ts_min = min(ts_list)
            ts_max = max(ts_list)
            time_diff = ts_max - ts_min

            # [새 로직] 1. 동기화 성공 케이스
            if time_diff < MAX_ALLOWED_DIFF_SEC:
                print(f"[Sync Thread] Synced! Time Diff: {time_diff:.4f}s")

                # [새 로직] 동기화 성공 시, 프레임 세트를 'inference_queue'에 넣음
                try:
                    inference_queue.put_nowait(
                        (zed_ts, (frame_zl, frame_zr), frame_l, frame_r)
                    )
                except Full:
                    # Queue가 가득 찼을 경우 (YOLO 추론이 너무 밀림)
                    # 경고를 출력하고 이 프레임 세트는 버림
                    #print("[Sync Thread] 경고: 추론 큐가 가득 찼습니다. 프레임 세트를 버립니다.")
                    pass
                
                # 동기화 성공 시, 3개 카메라 모두에서 새 프레임을 가져옴
                try:
                    zed_ts, (frame_zl, frame_zr) = cameras['zed'].read(timeout=MAIN_LOOP_TIMEOUT)
                    left_ts, frame_l = cameras['left'].read(timeout=MAIN_LOOP_TIMEOUT)
                    right_ts, frame_r = cameras['right'].read(timeout=MAIN_LOOP_TIMEOUT)
                except Empty:
                    print("[Sync Thread] 동기화 처리 중 큐가 비었습니다. 루프를 종료합니다.")
                    break
            
            # 2. 동기화 실패 케이스 (가장 오래된 프레임 버리기)
            else:
                oldest_cam_index = ts_list.index(ts_min)
                
                try:
                    if oldest_cam_index == 0: # ZED가 가장 오래됨
                        zed_ts, (frame_zl, frame_zr) = cameras['zed'].read(timeout=MAIN_LOOP_TIMEOUT)
                    elif oldest_cam_index == 1: # Left가 가장 오래됨
                        left_ts, frame_l = cameras['left'].read(timeout=MAIN_LOOP_TIMEOUT)
                    else: # Right가 가장 오래됨
                        right_ts, frame_r = cameras['right'].read(timeout=MAIN_LOOP_TIMEOUT)
                except Empty:
                    print("동기화 대기 중 큐가 비었습니다. 루프를 종료합니다.")
                    break # while 루프 종료
            
            # [New!] 1초 대기 추가
            #time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt 발생. 프로그램을 종료합니다.")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
    finally:
        shutdown_event.set()

        # 추론 스레드가 종료될 때까지 대기
        if 'yolo_thread' in locals():
            yolo_thread.join(timeout=5.0)

        for cam in cameras.values():
            cam.stop()
        print("[Sync Thread] 모든 리소스가 안전하게 해제되었습니다.")
    print("[Sync Thread] 프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()