# 카메라 촬영 main 함수

import os
import threading
from queue import Empty, Queue, Full

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
                    
                    try:
                        # Lazy init: 루프 내에서 최초 호출 시 한 번만 rclpy/CvBridge 노드 생성
                        if 'ros_rt' not in globals():
                            import rclpy
                            from rclpy.node import Node
                            from rclpy.qos import QoSProfile
                            from sensor_msgs.msg import Image
                            from cv_bridge import CvBridge

                            class RosImageRoundtrip:
                                def __init__(self):
                                    if not rclpy.ok():
                                        rclpy.init(args=None)
                                    self.node = rclpy.create_node('image_roundtrip_node')
                                    qos = QoSProfile(depth=1)
                                    self.pub = self.node.create_publisher(Image, 'image_roundtrip', qos)
                                    self.bridge = CvBridge()
                                    self._event = threading.Event()
                                    self.last_msg = None
                                    # callback: 수신 시 마지막 메시지를 저장하고 이벤트 셋
                                    def _cb(msg):
                                        self.last_msg = msg
                                        self._event.set()
                                    self.sub = self.node.create_subscription(Image, 'image_roundtrip', _cb, qos)
                                    # spin을 별도 스레드에서 실행하여 콜백이 처리되도록 함
                                    self._spin_thread = threading.Thread(target=lambda: rclpy.spin(self.node), daemon=True)
                                    self._spin_thread.start()

                                def roundtrip(self, cv_img, timeout=0.05, encoding='bgr8'):
                                    """이미지를 ROS msg로 변환해 publish한 뒤 수신된 msg를 다시 cv2 이미지로 변환해 반환합니다.
                                       실패하면 None을 반환합니다."""
                                    self._event.clear()
                                    img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding=encoding)
                                    self.pub.publish(img_msg)
                                    if self._event.wait(timeout):
                                        try:
                                            return self.bridge.imgmsg_to_cv2(self.last_msg, desired_encoding=encoding)
                                        except Exception:
                                            return None
                                    return None

                                def destroy(self):
                                    try:
                                        self.node.destroy_node()
                                    except Exception:
                                        pass

                            ros_rt = RosImageRoundtrip()

                        # 라운드트립 수행: 필요한 카메라 프레임들에 대해 적용
                        # encoding은 카메라 프레임의 포맷에 맞춰 변경 (예: 'bgr8', 'rgb8', 'mono8', 'rgba8' 등)
                        # --- 원본을 복사하여 비교용으로 보존 ---
                        frame_l_orig = None if frame_l is None else frame_l.copy()
                        frame_r_orig = None if frame_r is None else frame_r.copy()
                        frame_zl_orig = None if frame_zl is None else frame_zl.copy()
                        frame_zr_orig = None if frame_zr is None else frame_zr.copy()

                        rt_left = ros_rt.roundtrip(frame_l, timeout=0.05, encoding='bgr8')
                        if rt_left is not None:
                            frame_l = rt_left

                        rt_right = ros_rt.roundtrip(frame_r, timeout=0.05, encoding='bgr8')
                        if rt_right is not None:
                            frame_r = rt_right

                        # ZED 쌍(frames)도 동일 방식으로 처리 (프레임 타입에 맞는 encoding 사용)
                        rt_zl = ros_rt.roundtrip(frame_zl, timeout=0.05, encoding='bgr8')
                        if rt_zl is not None:
                            frame_zl = rt_zl
                        rt_zr = ros_rt.roundtrip(frame_zr, timeout=0.05, encoding='bgr8')
                        if rt_zr is not None:
                            frame_zr = rt_zr

                        # --- 간단한 검증 / 샘플 저장: 해시/shape/type 및 픽셀 차 통계 처리 ---
                        try:
                            import numpy as np, hashlib, math, time, cv2, pickle, json, os
                            
                            def _diff_stats(a, b):
                                if a is None or b is None:
                                    return None
                                if a.shape != b.shape:
                                    return {'same_shape': False}
                                da = a.astype(np.int32) - b.astype(np.int32)
                                mse = float(np.mean(da * da))
                                max_abs = int(np.max(np.abs(da)))
                                psnr = float('inf') if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)
                                return {'same_shape': True, 'mse': mse, 'max_abs': max_abs, 'psnr': psnr}

                            # 기존 통계/경고 로직 (간단히 유지)
                            checks = [
                                ('left', frame_l_orig, frame_l),
                                ('right', frame_r_orig, frame_r),
                                ('zed_left', frame_zl_orig, frame_zl),
                                ('zed_right', frame_zr_orig, frame_zr),
                            ]
                            for name, orig, new in checks:
                                stats = _diff_stats(orig, new)
                                # (원래의 경고/요약 로직을 여기서 처리)
                                # ...

                            # --- 샘플 저장: 지정 수만큼만 저장 ---
                            try:
                                # _sample_saved_count는 전역 카운터
                                if _sample_saved_count < SAMPLE_SAVE_COUNT:
                                    idx = _sample_saved_count + 1
                                    sample_dir = os.path.join(OUTPUT_DIR, "rt_samples")
                                    os.makedirs(sample_dir, exist_ok=True)

                                    # helper: 인코딩 추정
                                    def _detect_encoding(img):
                                        if img is None: return 'bgr8'
                                        if img.ndim == 2: return 'mono8'
                                        if img.ndim == 3 and img.shape[2] == 3: return 'bgr8'
                                        if img.ndim == 3 and img.shape[2] == 4: return 'rgba8'
                                        return 'bgr8'

                                    # 각 이미지 타입별로 저장
                                    frames_to_save = [
                                        ('left', frame_l_orig, frame_l),
                                        ('right', frame_r_orig, frame_r),
                                        ('zed_left', frame_zl_orig, frame_zl),
                                        ('zed_right', frame_zr_orig, frame_zr),
                                    ]
                                    # cv_bridge 객체 확보 (ros_rt가 있으면 그 bridge 사용)
                                    bridge = None
                                    try:
                                        if 'ros_rt' in globals() and hasattr(ros_rt, 'bridge'):
                                            bridge = ros_rt.bridge
                                        else:
                                            from cv_bridge import CvBridge
                                            bridge = CvBridge()
                                    except Exception:
                                        bridge = None

                                    metadata = {'sample_index': idx, 'timestamp': time.time(), 'items': []}
                                    for name, orig_img, restored_img in frames_to_save:
                                        entry = {'name': name}
                                        # 저장: 원본
                                        if orig_img is not None:
                                            orig_path = os.path.join(sample_dir, f"sample_{idx}_{name}_orig.png")
                                            cv2.imwrite(orig_path, orig_img)
                                            entry['orig_path'] = orig_path
                                        else:
                                            entry['orig_path'] = None

                                        # 변환된 메시지 (cv2_to_imgmsg 결과)를 pickle로 저장
                                        img_msg = None
                                        if bridge is not None and orig_img is not None:
                                            try:
                                                enc = _detect_encoding(orig_img)
                                                img_msg = bridge.cv2_to_imgmsg(orig_img, encoding=enc)
                                                msg_path = os.path.join(sample_dir, f"sample_{idx}_{name}_msg.pkl")
                                                with open(msg_path, 'wb') as f:
                                                    pickle.dump(img_msg, f)
                                                entry['msg_path'] = msg_path
                                            except Exception as _e:
                                                entry['msg_path'] = None
                                        else:
                                            entry['msg_path'] = None

                                        # 복원 이미지(이미 rt를 통해 대체되어 있으면 restored_img 사용)
                                        if restored_img is not None:
                                            rest_path = os.path.join(sample_dir, f"sample_{idx}_{name}_restored.png")
                                            cv2.imwrite(rest_path, restored_img)
                                            entry['restored_path'] = rest_path
                                        else:
                                            entry['restored_path'] = None

                                        metadata['items'].append(entry)

                                    # 메타데이터 파일 저장
                                    meta_path = os.path.join(sample_dir, f"sample_{idx}_meta.json")
                                    with open(meta_path, 'w') as mf:
                                        json.dump(metadata, mf, indent=2)

                                    _sample_saved_count += 1
                                    print(f"[RT SAMPLE] saved sample {idx} -> {sample_dir}")

                            except Exception as _e:
                                print(f"[RT SAMPLE] 저장 중 예외: {_e}")

                        except Exception as _e:
                            print(f"[RT CHECK] 예외: {_e}")

                    except Exception as e:
                        print(f"[ROS roundtrip] 오류: {e}")

                    # -------------------------------------------------------------------------------------------------------------
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