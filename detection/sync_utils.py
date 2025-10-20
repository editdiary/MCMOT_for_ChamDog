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

# # [Warning!] 이 함수는 현재 로직의 문제 원인일 수 있음
# def find_best_match_in_queue(slave_queue, master_ts):
#     """Slave Queue에서 Master Timestamp와 가장 가까운 frame을 찾는 함수"""
#     if slave_queue.empty():
#         return None

#     buffer_list = []
#     # 큐에서 모든 아이템을 안전하게 가져오기
#     while not slave_queue.empty():
#         try: 
#             item = slave_queue.get_nowait()
#             buffer_list.append(item)
#         except Empty: 
#             break
    
#     if not buffer_list:
#         return None
    
#     # 가장 가까운 타임스탬프를 가진 아이템 찾기
#     best_match = min(buffer_list, key=lambda x: abs(x[0] - master_ts))
    
#     # 나머지 아이템들을 다시 큐에 넣기 (최신 것만 유지)
#     for item in buffer_list:
#         if item != best_match:  # 선택된 것 제외하고 나머지 다시 넣기
#             try:
#                 slave_queue.put_nowait(item)
#             except:
#                 # 큐가 가득 찬 경우 가장 오래된 것 버리기
#                 break
    
#     return best_match