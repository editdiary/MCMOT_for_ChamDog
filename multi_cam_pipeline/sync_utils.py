# 유틸리티 함수들 모은 파일

import cv2
import os

def save_sync_frames(frame_set, frame_l, frame_r, output_dir, frame_roi=None, yolo_detections=None):
    """동기화된 프레임들을 저장 (YOLO bbox 그리기 포함)"""
    try:
        # ZED 프레임 저장
        cv2.imwrite(os.path.join(output_dir, "test_zed_left.jpg"), frame_set[0])
        cv2.imwrite(os.path.join(output_dir, "test_zed_right.jpg"), frame_set[1])

        # [New!] frame_l에 YOLO_bbox 그리기
        if yolo_detections:
            frame_l_annotated = frame_l.copy()
            for det in yolo_detections:
                label = det['label']
                conf = det['conf']
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(frame_l_annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label_text = f"{label}: {conf:.2f}"
                cv2.putText(frame_l_annotated, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # bbox가 그려진 이미지로 저장할 이미지 교체
            frame_l_to_save = frame_l_annotated
        else:
            # 탐지된 객체가 없으면 원본 이미지 저장
            frame_l_to_save = frame_l
        # --------------------------------------------

        # Arducam 프레임들 저장
        cv2.imwrite(os.path.join(output_dir, "test_left.jpg"), frame_l_to_save)
        cv2.imwrite(os.path.join(output_dir, "test_right.jpg"), frame_r)

        # frame_roi가 None이 아닐 때만 저장
        if frame_roi is not None:
            # 승운님 작성해주신 roi 영역 저장
            cv2.imwrite(os.path.join(output_dir, "test_roi.jpg"), frame_roi)

        return True
    except Exception as e:
        print(f"이미지 저장 실패: {e}")
        return False

def print_camera_settings(cameras, cam_config, zed_config):
    """설정한 카메라 파라미터와 실제 적용된 설정값을 출력"""

    print("\n=== 설정한 카메라 파라미터 요약 ===")
    print(f"ZED Camera:    {zed_config['zed_width']}x{zed_config['zed_height']} @ {zed_config['zed_fps']}fps")
    print(f"ArduCam Left:  {cam_config['frame_width']}x{cam_config['frame_height']} @ {cam_config['fps']}fps")
    print(f"ArduCam Right: {cam_config['frame_width']}x{cam_config['frame_height']} @ {cam_config['fps']}fps")
    print("========================\n")

    print("=== 실제 카메라 설정값 요약 ===")
    try:
        # ZED Camera (V4L2)
        zed_cam = cameras['zed'].cap
        print(f"ZED Camera:    {int(zed_cam.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(zed_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(zed_cam.get(cv2.CAP_PROP_FPS))}fps")

        # ArduCam Left
        left_cap = cameras['left'].cap
        print(f"ArduCam Left:  {int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(left_cap.get(cv2.CAP_PROP_FPS))}fps")
        
        # ArduCam Right  
        right_cap = cameras['right'].cap
        print(f"ArduCam Right: {int(right_cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(right_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(right_cap.get(cv2.CAP_PROP_FPS))}fps")
    except Exception as e:
        print(f"카메라 설정값 읽기 실패: {e}")
    finally:
        print("==========================\n")