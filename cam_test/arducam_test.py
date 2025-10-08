import cv2

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

def main():
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
    right_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    print(f"오른쪽 카메라 '{right_cam}'를 성공적으로 열었습니다.")
    print(f"해상도: {right_cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{right_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {right_cap.get(cv2.CAP_PROP_FPS)}")

    
    # 메인 루프: 프레임 읽기 및 화면 표시
    exit_app = False
    GUI_DEBUG = False
    try:
        while not exit_app:
            # frame 한 장씩 읽기
            ret_l, frame_l = left_cap.read()
            ret_r, frame_r = right_cap.read()

            if not ret_l:
                print("왼쪽 카메라 프레임 읽기 실패")
            if not ret_r:
                print("오른쪽 카메라 프레임 읽기 실패")
            

            ######### GUI로 디버깅 할 때 사용 #########
            # 읽은 프레임을 "Arducam"이라는 창에 표시
            if GUI_DEBUG:
                if ret_l:
                    cv2.imshow("Arducam LEFT", frame_l)
                if ret_r:
                    cv2.imshow("Arducam RIGHT", frame_r)

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
        print("카메라가 안전하게 닫혔습니다.")

    print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()