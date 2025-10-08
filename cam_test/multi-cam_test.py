import cv2
import pyzed.sl as sl

# 카메라 속성 정의
# v4l2-ctl로 확인한 지원하는 값으로 설정해야 한다.
    # 명령어: v4l2-ctl -d /dev/arducam_left --list-formats-ext
CAM_CONFIG = {
    "left_cam": "/dev/arducam_left",
    "right_cam": "/dev/arducam_right",
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
    "fourcc": cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')    # MJPEG 코덱 설정 (압축 포맷 사용 시)
}
ZED_CONFIG = {
    "resolution": sl.RESOLUTION.HD720,     # 원하는 해상도
    "fps": 30,   # 원하는 프레임률
    "depth_mode": sl.DEPTH_MODE.NONE,   # GPU 가속을 위한 NEURAL 모드
    "sdk_verbose": 1    
}

def main():
    ######### Arducam 카메라 설정 #########
    # udev 규칙으로 만든 카메라 별명 사용
    left_cam = CAM_CONFIG.left_cam
    right_cam = CAM_CONFIG.right_cam

    # 카메라 열기
    # cv2.CAP_V4L2를 추가하여 Video4Linux2 백엔드를 사용하도록 명시하면 더 안정적일 수 있음
    ## OpenCV는 다양한 운영체제에서 동작하는 라이브러리이기 때문에, 다른 방법은 시도하지 말고 리눅스에서 사용하는 V4L2 백엔드를 직접 사용하라고 명시해주는 것 
    left_cap = cv2.VideoCapture(left_cam, cv2.CAP_V4L2)
    right_cap = cv2.VideoCapture(right_cam, cv2.CAP_V4L2)

    if not left_cap.isOpened():
        print(f"오류: 카메라를 열 수 없습니다. 경로: {left_cam}")
    if not right_cap.isOpened():
        print(f"오류: 카메라를 열 수 없습니다. 경로: {right_cam}")
    
    # 카메라 속성 설정
    left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_CONFIG.frame_width)
    left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_CONFIG.frame_height)
    left_cap.set(cv2.CAP_PROP_FPS, CAM_CONFIG.fps)
    left_cap.set(cv2.CAP_PROP_FOURCC, CAM_CONFIG.fourcc)

    print(f"왼쪽 카메라 '{left_cam}'를 성공적으로 열었습니다.")
    print(f"해상도: {left_cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {left_cap.get(cv2.CAP_PROP_FPS)}")

    right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_CONFIG.frame_width)
    right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_CONFIG.frame_height)
    right_cap.set(cv2.CAP_PROP_FPS, CAM_CONFIG.fps)    
    right_cap.set(cv2.CAP_PROP_FOURCC, CAM_CONFIG.fourcc)
    print(f"오른쪽 카메라 '{right_cam}'를 성공적으로 열었습니다.")
    print(f"해상도: {right_cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{right_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {right_cap.get(cv2.CAP_PROP_FPS)}")

    
    ######### ZED 카메라 설정 #########
    zed = sl.Camera()

    # 초기화 파라미터 설정
    init = sl.InitParameters()
    init.camera_resolution = ZED_CONFIG.resolution
    init.camera_fps = ZED_CONFIG.fps
    init.depth_mode = ZED_CONFIG.depth_mode
    init.sdk_verbose = ZED_CONFIG.sdk_verbose

    # 카메라 열기
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS: # Ensure the camera has opened successfully
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()

    # Runtime 파라미터 설정 (여기서는 기본값 사용)
    runtime = sl.RuntimeParameters()
    zed_image = sl.Mat()    # 실시간 프레임을 받아올 객체 생성

    # 메인 루프: 프레임 읽기 및 화면 표시
    exit_app = False
    GUI_DEBUG = False
    try:
        while not exit_app:
            ######### ZED 카메라 프레임 읽기 #########
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(zed_image, sl.VIEW.LEFT)
                frame_z = zed_image.get_data()      # 이미지를 메모리 상의 변수로 가져오기 (파일 저장 X)
                frame_z_rgb = cv2.cvtColor(frame_z, cv2.COLOR_RGBA2RGB)

            ######### Arducam 카메라 프레임 읽기 #########
            # frame 한 장씩 읽기
            ret_l, frame_l = left_cap.read()
            ret_r, frame_r = right_cap.read()

            if not ret_l:
                print("왼쪽 카메라 프레임 읽기 실패")
            if not ret_r:
                print("오른쪽 카메라 프레임 읽기 실패")
            

            ######### GUI로 디버깅 할 때 사용 #########
            if GUI_DEBUG:
                if ret_l:
                    cv2.imshow("Arducam LEFT", frame_l)
                if ret_r:
                    cv2.imshow("Arducam RIGHT", frame_r)
                if err == sl.ERROR_CODE.SUCCESS:
                    cv2.imshow("ZED Camera", frame_z_rgb)

                # 종료 조건: 'q' 키 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    exit_app = True
                    break

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