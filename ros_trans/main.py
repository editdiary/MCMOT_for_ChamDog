# 카메라 촬영 main 함수

import os
import threading
from queue import Empty, Queue, Full
import numpy as np
import cv2

# 분리된 모듈들 임포트
from config import (
    CAM_CONFIG, ZED_CONFIG, BUFFER_SIZE,
    MAIN_LOOP_TIMEOUT, MAX_ALLOWED_DIFF_SEC,
    INFERENCE_BUFFER_SIZE,
    THREAD_JOIN_TIMEOUT,
    OUTPUT_FOLDER_NAME, WEIGHTS_FOLDER_NAME,
    SAVE_BUFFER_SIZE
)
from camera_modules import ArduCam, ZedCam
from sync_utils import print_camera_settings
from processing import inference_worker, save_worker


# --- [New!] 경로 설정 ---
# main.py 파일이 있는 디렉토리의 절대 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 절대 경로 생성
OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_FOLDER_NAME)
WEIGHTS_DIR = os.path.join(BASE_DIR, WEIGHTS_FOLDER_NAME)

# 경로 존재 확인
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)

# 글로벌 종료 플래그
shutdown_event = threading.Event()

# [New!] 후처리(추론)를 위한 큐
inference_queue = Queue(maxsize=INFERENCE_BUFFER_SIZE)
save_queue = Queue(maxsize=SAVE_BUFFER_SIZE)

# [New!] 샘플 저장 설정 (몇 프레임만 저장할지)
SAMPLE_SAVE_COUNT = 5
if '_sample_saved_count' not in globals():
    _sample_saved_count = 0

def warmup_cameras(cameras, warmup_frames, timeout, shutdown_event):
    """카메라 안정화를 위해 지정된 프레임 수만큼 읽기를 시도합니다."""
    try:
        for i in range(warmup_frames):
            for cam_name, cam in cameras.items():
                try:
                    # 각 프레임 읽기 시도 (타임아웃 적용)
                    cam.read(timeout=timeout)
                except Empty:
                    # 개별 카메라 타임아웃 발생 시 경고 출력 (프로그램 중단은 아님)
                    print(f"워밍업 중 {cam_name} 카메라 타임아웃 ({i+1}/{warmup_frames} 프레임)")
                    pass
            # 워밍업 중에도 종료 신호 확인
            if shutdown_event.is_set():
                print("워밍업 중 종료 신호 감지.")
                return False
        
        # 모든 프레임 처리 완료
        print("워밍업 완료.")
        # [Option] 워밍업 후 큐를 비워 최신 상태에서 시작
        for cam in cameras.values():
            while not cam.queue.empty():
                try: cam.queue.get_nowait()
                except Empty: break
        return True     # 워밍업 성공
    
    except Exception as e:
        print(f"워밍업 중 오류 발생: {e}")
        shutdown_event.set()
        return False    # 워밍업 실패

def main():
    global _sample_saved_count
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
    
    # [New!] 추론 스레드 시작 (경로 전달)
    inference_thread = threading.Thread(
        target=inference_worker,
        args=(inference_queue, save_queue, shutdown_event, OUTPUT_DIR, WEIGHTS_DIR),
        daemon=True
    )
    inference_thread.start()

    # [New!] 저장 스레드 시작
    save_thread = threading.Thread(
        target=save_worker,
        args=(save_queue, shutdown_event, OUTPUT_DIR),
        daemon=True
    )
    save_thread.start()

    # 카메라 세팅값 요약 출력    
    print_camera_settings(cameras, CAM_CONFIG, ZED_CONFIG)

    exit_app = False

    # 메인 try 블록 시작
    try:
        # 카메라 안정화를 위한 워밍업 함수 호출
        WARMUP_FRAMES = CAM_CONFIG['fps'] * 2
        # 워밍업 read 타임아웃 설정 (MAIN_LOOP_TIMEOUT과 같거나 약간 길게)
        WARMUP_TIMEOUT = MAIN_LOOP_TIMEOUT * 2

        if not warmup_cameras(cameras, WARMUP_FRAMES, WARMUP_TIMEOUT, shutdown_event):
            print("카메라 안정화 실패. 프로그램을 종료합니다.")
            return
        
        # ----- 워밍업 로직 끝 -----

        # [수정] 3개의 큐에서 프레임을 하나씩 미리 가져옴
        try:
            zed_ts, (frame_zl, frame_zr) = cameras['zed'].read(timeout=MAIN_LOOP_TIMEOUT)
            left_ts, frame_l = cameras['left'].read(timeout=MAIN_LOOP_TIMEOUT)
            right_ts, frame_r = cameras['right'].read(timeout=MAIN_LOOP_TIMEOUT)
        except Empty:
            print("초기 프레임 로드 실패. 카메라 연결을 확인하세요.")
            exit_app = True

        while not exit_app and not shutdown_event.is_set():
            # [새 로직] 타임스탬프 리스트와 최대/최소값 계산
            ts_list = [zed_ts, left_ts, right_ts]
            ts_min = min(ts_list)
            ts_max = max(ts_list)
            time_diff = ts_max - ts_min

            # [새 로직] 1. 동기화 성공 케이스
            if time_diff < MAX_ALLOWED_DIFF_SEC:
                print(f"[Main Thread] Synced! Time Diff: {time_diff:.4f}s")

                # [새 로직] 동기화 성공 시, 프레임 세트를 'inference_queue'에 넣음
                try:
                    # -------------------------------------------------------------------------------------------------------------
                    # !!!!!!!! 이 부분에서 queue에 데이터 넣기 전에 ROS 변환 및 송신 + 전송된 데이터 수신 로직 추가하면 될 거 같습니다 !!!!!!!!
                    
                    # ROS 대신 CvBridge만 사용하는 직접 변환
                    try:
                        from cv_bridge import CvBridge
                        if 'bridge' not in globals():
                            bridge = CvBridge()
                        
                        # 원본 복사
                        frame_l_orig = None if frame_l is None else frame_l.copy()
                        frame_r_orig = None if frame_r is None else frame_r.copy()
                        frame_zl_orig = None if frame_zl is None else frame_zl.copy()
                        frame_zr_orig = None if frame_zr is None else frame_zr.copy()

                        def simple_roundtrip(img, encoding='bgr8'):
                            """CvBridge로 직접 변환→복원"""
                            if img is None:
                                return None
                            if not isinstance(img, np.ndarray):
                                return None
                            if img.size == 0:
                                return None
                                
                            try:
                                # ROS 메시지로 변환했다가 다시 이미지로
                                msg = bridge.cv2_to_imgmsg(img, encoding=encoding)
                                converted = bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
                                return converted
                            except Exception as e:
                                print(f"변환 오류: {e}")
                                return None

                        # 각 이미지 변환 전에 타입/내용 체크
                        for frame in [frame_l, frame_r, frame_zl, frame_zr]:
                            if frame is not None:
                                if not isinstance(frame, np.ndarray):
                                    print(f"경고: 잘못된 이미지 타입 - {type(frame)}")
                                    continue
                                if frame.size == 0:
                                    print("경고: 빈 이미지")
                                    continue
                                    
                        # 이미지 변환
                        converted_l = simple_roundtrip(frame_l, 'bgr8')
                        converted_r = simple_roundtrip(frame_r, 'bgr8')
                        converted_zl = simple_roundtrip(frame_zl, 'bgr8')
                        converted_zr = simple_roundtrip(frame_zr, 'bgr8')

                        # None인 경우 원본 사용
                        frame_l = converted_l if converted_l is not None else frame_l
                        frame_r = converted_r if converted_r is not None else frame_r
                        frame_zl = converted_zl if converted_zl is not None else frame_zl
                        frame_zr = converted_zr if converted_zr is not None else frame_zr

                        # --- 샘플 저장: 지정 수만큼만 저장 ---
                        if '_sample_saved_count' not in globals():
                            _sample_saved_count = 0

                        if _sample_saved_count < SAMPLE_SAVE_COUNT:
                            try:
                                import json, pickle, time

                                idx = _sample_saved_count + 1
                                root_sample_dir = os.path.join(OUTPUT_DIR, "rt_samples")
                                sample_dir = os.path.join(root_sample_dir, f"sample_{idx}")
                                os.makedirs(sample_dir, exist_ok=True)

                                frames_to_save = [
                                    ('left', frame_l_orig, frame_l),
                                    ('right', frame_r_orig, frame_r),
                                    ('zed_left', frame_zl_orig, frame_zl),
                                    ('zed_right', frame_zr_orig, frame_zr),
                                ]

                                # bridge 확인 (있으면 msg도 저장)
                                bridge_local = None
                                try:
                                    if 'ros_rt' in globals() and hasattr(ros_rt, 'bridge'):
                                        bridge_local = ros_rt.bridge
                                    else:
                                        from cv_bridge import CvBridge
                                        bridge_local = CvBridge()
                                except Exception:
                                    bridge_local = None

                                def _detect_encoding(img):
                                    if img is None: return 'bgr8'
                                    if img.ndim == 2: return 'mono8'
                                    if img.ndim == 3 and img.shape[2] == 3: return 'bgr8'
                                    if img.ndim == 3 and img.shape[2] == 4: return 'rgba8'
                                    return 'bgr8'

                                def compute_stats(a, b):
                                    if a is None or b is None:
                                        return None
                                    if a.shape != b.shape:
                                        return {'same_shape': False}
                                    da = a.astype(np.int32) - b.astype(np.int32)
                                    mse = float(np.mean(da * da))
                                    max_abs = int(np.max(np.abs(da)))
                                    mean_abs = float(np.mean(np.abs(da)))
                                    psnr = float('inf') if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)
                                    return {'same_shape': True, 'mse': mse, 'max_abs': max_abs, 'mean_abs': mean_abs, 'psnr': psnr}

                                metadata = {'sample_index': idx, 'timestamp': time.time(), 'items': []}
                                for name, orig_img, restored_img in frames_to_save:
                                    item = {'name': name}
                                    # save original
                                    if orig_img is not None:
                                        p_orig = os.path.join(sample_dir, f"{name}_orig.png")
                                        cv2.imwrite(p_orig, orig_img)
                                        item['orig_path'] = p_orig
                                    else:
                                        item['orig_path'] = None

                                    # save msg (pickle) if bridge available
                                    if bridge_local is not None and orig_img is not None:
                                        try:
                                            enc = _detect_encoding(orig_img)
                                            img_msg = bridge_local.cv2_to_imgmsg(orig_img, encoding=enc)
                                            p_msg = os.path.join(sample_dir, f"{name}_msg.pkl")
                                            with open(p_msg, 'wb') as mf:
                                                pickle.dump(img_msg, mf)
                                            item['msg_path'] = p_msg
                                            item['msg_meta'] = {
                                                'encoding': getattr(img_msg, 'encoding', None),
                                                'height': getattr(img_msg, 'height', None),
                                                'width': getattr(img_msg, 'width', None)
                                            }
                                        except Exception as _e:
                                            item['msg_path'] = None
                                            item['msg_meta'] = {'error': str(_e)}
                                    else:
                                        item['msg_path'] = None
                                        item['msg_meta'] = None

                                    # save restored
                                    if restored_img is not None:
                                        p_rest = os.path.join(sample_dir, f"{name}_restored.png")
                                        cv2.imwrite(p_rest, restored_img)
                                        item['restored_path'] = p_rest
                                    else:
                                        item['restored_path'] = None

                                    item['stats'] = compute_stats(orig_img, restored_img)
                                    metadata['items'].append(item)

                                # write metrics.json
                                meta_path = os.path.join(sample_dir, "metrics.json")
                                with open(meta_path, 'w') as mf:
                                    json.dump(metadata, mf, indent=2, ensure_ascii=False)

                                _sample_saved_count += 1
                                print(f"[RT SAMPLE] saved sample {idx} -> {sample_dir}")
                            except Exception as e:
                                print(f"[RT SAMPLE] 저장 중 예외: {e}")
                            
                    except ImportError:
                        print("cv_bridge를 찾을 수 없습니다. 원본 이미지를 그대로 사용합니다.")
                    except Exception as e:
                        print(f"이미지 변환 중 오류: {e}")

                    # 변환 여부와 관계없이 큐에 넣기
                    inference_queue.put_nowait(
                        (zed_ts, (frame_zl, frame_zr), frame_l, frame_r)
                    )
                except Full:
                    # Queue가 가득 찼을 경우 (YOLO 추론이 너무 밀림)
                    # 경고를 출력하고 이 프레임 세트는 버림
                    #print("[Main Thread] 경고: 추론 큐가 가득 찼습니다. 프레임 세트를 버립니다.")
                    pass
                
                # 동기화 성공 시, 3개 카메라 모두에서 새 프레임을 가져옴
                try:
                    zed_ts, (frame_zl, frame_zr) = cameras['zed'].read(timeout=MAIN_LOOP_TIMEOUT)
                    left_ts, frame_l = cameras['left'].read(timeout=MAIN_LOOP_TIMEOUT)
                    right_ts, frame_r = cameras['right'].read(timeout=MAIN_LOOP_TIMEOUT)
                except Empty:
                    print("[Main Thread] 동기화 처리 중 큐가 비었습니다. 루프를 종료합니다.")
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

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt 발생. 프로그램을 종료합니다.")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
    finally:
        shutdown_event.set()

        # 추론 스레드 종료 대기
        if 'inference_thread' in locals():
            inference_thread.join(timeout=THREAD_JOIN_TIMEOUT)

        # [New!] 저장 스레드 종료 대기
        if 'save_thread' in locals():
            save_thread.join(timeout=THREAD_JOIN_TIMEOUT)

        # 카메라 리소스 해제
        for cam in cameras.values():
            cam.stop()
        print("[Main Thread] 모든 리소스가 안전하게 해제되었습니다.")
    print("[Main Thread] 프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()