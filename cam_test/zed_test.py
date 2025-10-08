import pyzed.sl as sl
import cv2

def main():
    # ZED 카메라 객체 생성
    zed = sl.Camera()

    # 초기화 파라미터 설정
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720     # 원하는 해상도 설정
    init.camera_fps = 15    # 원하는 프레임률
    init.depth_mode = sl.DEPTH_MODE.NONE     # GPU 가속을 위한 NEURAL 모드 (현재는 depth 정보 사용 X)
    init.sdk_verbose = 1    # 디버깅 정보 출력

    # 카메라 열기
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS: # Ensure the camera has opened successfully
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()

    # Runtime 파라미터 설정 (여기서는 기본값 사용)
    runtime = sl.RuntimeParameters()
    image = sl.Mat()    # 실시간 프레임을 받아올 객체 생성

    # 무한루프에서 프레임 가져오기 (스트리밍)
    exit_app = False
    GUI_DEBUG = False   # 촬영 디버깅 여부 (모니터 출력)
    try :
        while not exit_app:
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                ######### 이미지 촬영 코드 #########
                # ZED의 왼쪽 카메라 이미지 가져오기
                zed.retrieve_image(image, sl.VIEW.LEFT) 
                frame_to_detect = image.get_data()      # 이미지를 메모리 상의 변수로 가져오기 (파일 저장 X)


                ######### 전처리 코드 #########
                # ZED는 4채널(RGBA) 이미지를 반환하므로, RGB로 변환
                frame_to_detect_rgb = cv2.cvtColor(frame_to_detect, cv2.COLOR_RGBA2RGB)


                ######### 촬영 디버깅 시 활용 #########
                if GUI_DEBUG:
                    # OpenCV로 화면에 표시
                    cv2.imshow("ZED Camera", image.get_data())

                    # 종료 조건: 'q' 키 누르면 종료
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        exit_app = True
                        break

                    # # 이미지를 파일로 저장
                    # file_name = "zed_capture.png"
                    # cv2.imwrite(file_name, image.get_data())
                    # print(f"이미지가 {file_name}에 저장되었습니다")

    except KeyboardInterrupt:
        print("KeyboardInterrupt 발생. 프로그램을 종료합니다.")
        exit_app = True

    finally:
        if GUI_DEBUG:
            cv2.destroyAllWindows()
        # close the Camera
        zed.close()
        print("카메라가 안전하게 닫혔습니다.")

    print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()