import pyzed.sl as sl
import cv2
from time import sleep
import subprocess

# 동적 USB 경로 탐색 및 리셋
def find_and_reset_zed():
    # ZED 카메라의 Vendor ID와 Product ID
    zed_vendor_id, zed_product_id = "2b03", "f880"

    # 1. lsusb 명령어로 ZED 카메라 정보 찾기
    command = f"lsusb -d {zed_vendor_id}:{zed_product_id}"
    try:
        # 셸 명령 실행
        result = subprocess.check_output(command.split()).decode('utf-8')
        
        # 2. 결과 파싱하여 버스 경로 만들기 (예: "Bus 001 Device 004: ...")
        parts = result.strip().split()
        bus = parts[1]
        device = parts[3].replace(':', '')
        zed_usb_path = f"/dev/bus/usb/{bus}/{device}"
        
        # 3. usbreset 실행
        reset_command = f"sudo usbreset {zed_usb_path}"
        print(f"ZED 카메라를 리셋합니다: {reset_command}")
        
        return_code = subprocess.call(reset_command.split())
        if return_code == 0:
            print("USB 포트 리셋 성공. 3초 후 카메라 초기화를 시작합니다.")
            sleep(3)
            return True
        else:
            print("USB 포트 리셋 실패.")
            return False

    except subprocess.CalledProcessError:
        print("ZED 카메라를 찾을 수 없습니다. 연결을 확인해주세요.")
        return False

def main():
    # ZED 카메라 객체 생성
    zed = sl.Camera()

    # 초기화 파라미터 설정
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720     # 원하는 해상도 설정
    init.camera_fps = 30    # 원하는 프레임률
    init.depth_mode = sl.DEPTH_MODE.NONE     # GPU 가속을 위한 NEURAL 모드 (현재는 depth 정보 사용 X)
    init.sdk_verbose = 1    # 디버깅 정보 출력

    # -------- 재시도 루프 시작 --------
    # 카메라 오픈에 실패할 경우를 대비해서 재시도 루프 시작
    max_retries = 5
    retry_count = 0
    status = sl.ERROR_CODE.FAILURE

    while retry_count < max_retries and status != sl.ERROR_CODE.SUCCESS:
        # 카메라 열기
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            retry_count += 1
            print(f"카메라 열기 실패 ({retry_count}/{max_retries}): {repr(status)}. 5초 후 재시도합니다.")
            # 실패 시 ZED 내부 상태를 리셋하기 위해 close() 호출
            zed.close()
            sleep(5)    # ZED 카메라가 부팅될 충분한 시간을 줍니다.

            # 초기에 카메라 열기에 실패했을 경우 최초에 한해 usbreset 실행
            if retry_count == 0:
                if not find_and_reset_zed():
                    print("초기화 실패. 프로그램을 종료합니다.")
                    exit()
    # -------- 재시도 루프 종료 --------

    if status != sl.ERROR_CODE.SUCCESS:
        print("최대 재시도 횟수를 초과했습니다. 프로그램을 종료합니다.")
        exit()
    
    print("카메라를 성공적으로 열었습니다.")

    # Runtime 파라미터 설정 (여기서는 기본값 사용)
    runtime = sl.RuntimeParameters()
    image = sl.Mat()    # 실시간 프레임을 받아올 객체 생성

    # 무한루프에서 프레임 가져오기 (스트리밍)
    exit_app = False
    GUI_DEBUG = False   # 촬영 디버깅 여부 (모니터 출력)
    SAVE_IMG = False
    try :
        while not exit_app:
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                #--------- 이미지 촬영 ---------
                # ZED의 왼쪽 카메라 이미지 가져오기
                zed.retrieve_image(image, sl.VIEW.LEFT) 
                frame_to_detect = image.get_data()      # 이미지를 메모리 상의 변수로 가져오기 (파일 저장 X)

                #--------- 촬영 디버깅 시 활용 ---------
                # GUI를 통한 실시간 디버깅을 위한 코드
                if GUI_DEBUG:
                    # ZED는 4채널(RGBA) 이미지를 반환하므로, RGB로 변환
                    frame_to_detect_rgb = cv2.cvtColor(frame_to_detect, cv2.COLOR_RGBA2RGB)
                    # OpenCV로 화면에 표시
                    cv2.imshow("ZED Camera", frame_to_detect_rgb)

                    # 종료 조건: 'q' 키 누르면 종료
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        exit_app = True
                        break

                # 이미지를 파일로 저장하며 확인할 용도의 코드
                if SAVE_IMG:
                    # ZED는 4채널(RGBA) 이미지를 반환하므로, RGB로 변환
                    frame_to_detect_rgb = cv2.cvtColor(frame_to_detect, cv2.COLOR_RGBA2RGB)

                    file_name = "zed_capture.png"
                    cv2.imwrite(file_name, frame_to_detect_rgb)

                    print(f"이미지가 {file_name}에 저장되었습니다")

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