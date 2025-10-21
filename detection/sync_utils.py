# 유틸리티 함수들 모은 파일

import cv2

def save_sync_frames(frame_set, frame_l, frame_r, frame_roi=None):
    """동기화된 프레임들을 저장 (리소스 부하 테스트용)"""
    try:
        # ZED 프레임 저장
        cv2.imwrite("test_zed_left.jpg", frame_set[0])
        cv2.imwrite("test_zed_right.jpg", frame_set[1])
        
        # Arducam 프레임들 저장
        cv2.imwrite("test_left.jpg", frame_l)
        cv2.imwrite("test_right.jpg", frame_r)
        
        # frame_roi가 None이 아닐 때만 저장
        if frame_roi is not None:
            # 승운님 작성해주신 roi 영역 저장
            cv2.imwrite("test_roi.jpg", frame_roi)

        return True
    except Exception as e:
        print(f"이미지 저장 실패: {e}")
        return False

def print_camera_settings(cameras):
    """모든 카메라의 실제 설정값을 간단히 출력"""
    print("=== 실제 카메라 설정값 요약 ===")
    try:
        # ZED Camera
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