# 후처리(추론) 스레드 관리 파일

import cv2
from queue import Empty, Full, Queue
import time
from concurrent.futures import ThreadPoolExecutor

from pipe_detection import find_pipe_line
#from sync_utils import save_sync_frames  # 이제 저장 함수는 save_worker가 사용
from config import (
    GUI_DEBUG, IMG_SAVE,
    USE_ZED_RIGHT_FOR_PIPE,
    INFERENCE_WORKER_TIMEOUT,
    CPU_WORKER_TIMEOUT,
    SAVE_BUFFER_SIZE
)

# [New!] 저장 스레드로 데이터를 전달할 큐
save_queue = Queue(maxsize=SAVE_BUFFER_SIZE)


# [New!] 파이프 탐지 및 YOLO 추론을 담당할 소비자(Consumer) 스레드 함수
def inference_worker(queue, shutdown_event):
    """
    inference_queue에서 동기화된 프레임 세트를 가져와
    CPU 작업(Pipe 탐지)과 GPU 작업(YOLO 추론)을 병렬로 처리
    """
    print("[Inference Thread] 추론 스레드 시작...")

    # [New!] CPU 작업을 처리할 1개의 워커 스레드를 가진 thread pool 생성
    # (CPU 코어 수에 맞춰 조절 가능)
    cpu_thread_pool = ThreadPoolExecutor(max_workers=1)
    pipe_future = None      # 안정적인 운영을 위한 변수 초기화

    while not shutdown_event.is_set():
        try:
            # 1. 동기화된 프레임 세트를 큐에서 가져옴 (타임아웃 1초)
            (ts, (frame_zl, frame_zr), frame_l, frame_r) = queue.get(INFERENCE_WORKER_TIMEOUT)

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
            
            # 파이프 탐지를 위한 이미지 선택
            if USE_ZED_RIGHT_FOR_PIPE:
                PIPE_IMAGE = frame_zr
            else:
                PIPE_IMAGE = frame_zl

            # ----- [핵심] 병렬 처리 -----
            # 2. (병렬 작업 1) CPU 스레드에 파이프 탐지 작업을 비동기로 "제출"
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
            roi_view, roi_coords, vertical_lines, track_point = pipe_future.result(timeout=CPU_WORKER_TIMEOUT)

            # (여기서 yolo_resultsl, yolo_resultsr, detection_result를
            #  모두 사용하여 이미지에 그리거나 저장)

            # ------- 결과 취합 및 시각화 -------
            annotated_image = None  # 초기화

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

            # ----- [핵심 수정] 저장 큐에 데이터 넣기 -----
            if IMG_SAVE:
                try:
                    # 저장 스레드에 필요한 모든 데이터를 튜플로 묶어 전달
                    save_data = (frame_zl, frame_zr, frame_l, frame_r, annotated_image)
                    save_queue.put_nowait(save_data)
                except Full:
                    print("[Inference Thread] 경고: 저장 큐가 가득 찼습니다. 이미지 저장을 건너뜁니다.")
                
                # [삭제] 직접 save_sync_frames 호출 및 print 제거
                #if save_sync_frames((frame_zl, frame_zr), frame_l, frame_r, frame_roi=annotated_image):
                #    print("이미지 저장 완료")

            if GUI_DEBUG:
                cv2.imshow("ZED Camera Left", frame_zl)
                #cv2.imshow("ZED Camera Right", frame_zr)
                cv2.imshow("Arducam LEFT", frame_l)
                cv2.imshow("Arducam RIGHT", frame_r)

                # 시각화된 ROI가 있다면 별도 창으로 표시
                if annotated_image is not None:
                    cv2.imshow("Pipe Detection View", annotated_image)
                
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


# --- [New!] 저장 스레드 함수 ---
from sync_utils import save_sync_frames
from config import SAVE_WORKER_TIMEOUT

def save_worker(queue, shutdown_event):
    """
    save_queue에서 데이터를 가져와 이미지 파일로 저장하는 스레드
    """
    print("[Save Thread] 저장 스레드 시작...")
    while not shutdown_event.is_set():
        try:
            # 1. 저장할 데이터 세트를 큐에서 가져옴
            (frame_zl, frame_zr, frame_l, frame_r, annotated_image) = queue.get(SAVE_WORKER_TIMEOUT)

            # 2. sync_utils의 저장 함수 호출
            if save_sync_frames(frame_zl, frame_zr, frame_l, frame_r, frame_roi=annotated_image):
                print("[Save Thread] 이미지 저장 완료")
            else:
                print("[Save Thread] 이미지 저장 실패")
        
        except Empty:
            # 큐가 비어있으면 계속 대기
            continue
        except Exception as e:
            if shutdown_event.is_set():
                break   # 종료 신호면 조용히 종료
            print(f"[Save Thread] 저장 중 오류 발생: {e}")
    
    print("[Save Thread] 저장 스레드 종료.")