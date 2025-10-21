# 파이프 탐지 알고리즘 함수 (by 승운님)

import cv2
import numpy as np
from config import PIPE_ROI_COORS

def find_pipe_line(frame_z):
    """
    ZED 이미지에서 수직선을 찾아 추적점의 좌표와 결과 데이터를 반환하는 함수
    """

    # 이미지 유효성 검사
    if frame_z is None or frame_z.size == 0:
        print("find_pipe_line: 입력 이미지가 비어있습니다.")
        return None, None, None, None
    
    # ROI(관심 영역) 좌표
    roi_x = PIPE_ROI_COORS['x']
    roi_y = PIPE_ROI_COORS['y']
    roi_w = PIPE_ROI_COORS['w']
    roi_h = PIPE_ROI_COORS['h']
    roi_coords = (roi_x, roi_y, roi_w, roi_h)   # 좌표 튜블 저장

    # ROI(관심 영역) 자르기
    roi = frame_z[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # 이미지 전처리
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(equalized, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 허프 변환으로 수직선 필터링
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=30, maxLineGap=30)
    
    pipe_line_x_coords = []
    vertical_lines = []
    track_point = None

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.rad2deg(np.arctan2(y2 - y1, x2 - x1))) if x2 - x1 != 0 else 90.0
            if angle > 70:
                vertical_lines.append(line[0])
                pipe_line_x_coords.extend([x1, x2])

    # 추적점 계산
    if pipe_line_x_coords:
        avg_x = int(np.mean(pipe_line_x_coords))
        center_y = roi.shape[0] // 2
        track_point = (avg_x, center_y)

    return roi, roi_coords, vertical_lines, track_point