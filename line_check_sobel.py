#line check.ipynb 에 있는 함수중 필요한 것만 가지고 옴


# Imports
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
# Dont show warnings
warnings.filterwarnings("ignore")

src = np.array([[598, 448], [684, 448], [1026, 668], [278, 668]], np.float32)
dst = np.array([[300, 0], [980, 0], [980, 720], [300, 720]], np.float32)

kernel_small = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], 'uint8')

# Convert image to yellow and white color space
def color_space(img):
    # Convert image to HSV
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Colorspace "yellow" in HSV: (15-40, 80-255, 160-255)
    mask_yellow = cv.inRange(img_hsv, (15, 100, 160), (40, 255, 255))
    # Colorspace "white" in HSV: (0-255, 0-20, 200-255)
    mask_white = cv.inRange(img_hsv, (0, 0, 200), (255, 70, 255))
    # Merge white and yellow masks
    masks = cv.bitwise_or(mask_yellow, mask_white)
    # Return image in gray
    
    return cv.cvtColor(cv.bitwise_and(img, img, mask=masks), cv.COLOR_BGR2GRAY)


def color_space_hls(img):
    # Convert image to HSV
    img_hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    # Colorspace "yellow" in HSV: (15-40, 80-255, 160-255)
    mask_yellow = cv.inRange(img_hls, (20, 0, 0), (27, 255, 255))
    # Colorspace "white" in HSV: (0-255, 0-20, 200-255)
    mask_white = cv.inRange(img_hls, (0, 0, 200), (180, 255, 255))


    """
    차선 색이 흐리거나 하면 이걸 써봐야 할 수도 있음
    mask_yellow = cv.inRange(img_hls, (10, 80, 80), (40, 255, 255))
    mask_white = cv.inRange(img_hls, (0, 200, 0), (180, 255, 150))
    """

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # 초록색 제거용 마스크 (선택 사항)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv.inRange(hsv, lower_green, upper_green)


    # 노랑에서 초록 제거
    yellow_clean = cv.bitwise_and(mask_yellow, cv.bitwise_not(green_mask))

    yellow_clean_masks = cv.bitwise_and(mask_yellow, yellow_clean)

    # Merge white and yellow masks
    masks = cv.bitwise_or(yellow_clean_masks, mask_white)
    # Return image in gray

    #cv.imshow('hls', cv.cvtColor(cv.bitwise_and(img, img, mask=masks), cv.COLOR_HLS2BGR))
    #cv.imshow('hls', cv.bitwise_and(img, img, mask=masks))
    
    return cv.cvtColor(cv.bitwise_and(img, img, mask=masks), cv.COLOR_BGR2GRAY)


def open_img(img, iterations):
    return cv.morphologyEx(img, cv.MORPH_OPEN, kernel_small, iterations=iterations)


import numpy as np


class LaneTracker:
    def __init__(self, nwindows=9, margin=200, minimum=30):
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.nwindows = nwindows
        self.margin = margin
        self.minimum = minimum
        

    def reset(self):
        self.prev_left_fit = None
        self.prev_right_fit = None

    #다항식 리셋용
    #후술할 추적 방식에 리셋시 필요할 경우를 위해 왼쪽 차선이 오른쪽 차선과 교차하면 리셋되게 만듦
    def should_reset(self, left_fit, right_fit, warped_img):
        if left_fit is None or right_fit is None:
            #print("차선 인식 안됨")
            return True

        ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
        left_fitx = np.polyval(left_fit, ploty)
        right_fitx = np.polyval(right_fit, ploty)

        
        y_eval = np.max(ploty)  # 가장 아래 지점에서의 곡률 평가
        threshold = 2000
        def calc_curvature(fit):
            A = fit[0]
            B = fit[1]
            return ((1 + (2*A*y_eval + B)**2)**1.5) / np.abs(2*A)

        left_curve = calc_curvature(left_fit)
        right_curve = calc_curvature(right_fit)

        #diff = np.abs(left_curve - right_curve)


        curvature_ratio = max(left_curve, right_curve) / min(left_curve, right_curve)
        curvature_diff = abs(left_curve - right_curve)
        """
        if abs(self.prev_left_fit[0]) > 5.0e-04 and abs(self.prev_right_fit[0]) > 5.0e-04:
            if curvature_ratio > 2.5 and np.sign(self.prev_left_fit[0]) != np.sign(self.prev_right_fit[0]):
                return True
        """
        """
        if curvature_diff > threshold and abs(self.prev_left_fit[0]) > 1.0e-03 and abs(self.prev_right_fit[0]) > 1.0e-03:
            return True
            """

        if abs(left_fit[0] - right_fit[0]) > 5.0e-03:
            return True
        

        # 1. 좌우 교차 판단
        if np.any(left_fitx >= right_fitx):
            return True

        # 2. 거리 기반 판단
        #영상별로 차선간 거리가 달라져서 쓰기에는 힘들것으로 보임
        """
        lane_width = right_fitx - left_fitx
        if np.any(lane_width < 200) or np.any(lane_width > 800):
            return True
        """
        # 3. 곡률 차이 판단 (선택사항)
        # left_curv = calc_curvature(left_fit, ploty)
        # right_curv = calc_curvature(right_fit, ploty)
        # if abs(left_curv - right_curv) > threshold:
        #     return True

    def update(self, warped_img, draw=True):
        if self.should_reset(self.prev_left_fit, self.prev_right_fit, warped_img):
            #result = self.sliding_windows_visual(warped_img, draw)
            """
            if self.dummy is not None:
                plt.imshow(cv.cvtColor(self.dummy, cv.COLOR_BGR2RGB))
                plt.show()
            """
            result = self.sliding_windows_visual_central(warped_img, draw)
        else:
            result = self.quick_search(warped_img, draw)

        if self.should_reset(result["left"]["fit"], result["right"]["fit"], warped_img):
            self.reset_F = True
            result["left"]["fit"] = None
            result["right"]["fit"] = None
        else:
            self.reset_F = False
        # 상태 갱신
        self.prev_left_fit = result["left"]["fit"]
        self.prev_right_fit = result["right"]["fit"]
        self.dummy = result["image"]
        return result
    #sliding window
    def sliding_windows_visual(self, warped_img, draw):
        # ▶ ROI 마스킹: 잘못된 영역 제거
        mask = np.ones_like(warped_img, dtype=np.uint8) * 255
        height, width = warped_img.shape
        cv.rectangle(mask, (0, height - 20), (width, height), 0, -1)

        
        warped_img = cv.bitwise_and(warped_img, mask)

        
        """
        plt.imshow(cv.cvtColor(warped_img, cv.COLOR_BGR2RGB))
        plt.show()
        """
        # ▶ 모폴로지 연산으로 노이즈 제거
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        warped_img = cv.morphologyEx(warped_img, cv.MORPH_OPEN, kernel)

        # ▶ 시각화용 이미지 생성
        out_img = np.dstack((warped_img, warped_img, warped_img)) * 255

        # ▶ 히스토그램 기반 시작점 계산
        histogram = np.sum(warped_img[warped_img.shape[0]//2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        """
        plt.plot(histogram)
        plt.title("Lane pixel histogram (bottom half)")
        plt.show()
        """
        # ▶ 슬라이딩 윈도우 초기화
        window_height = warped_img.shape[0] // self.nwindows
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):
            win_y_low = warped_img.shape[0] - (window + 1) * window_height
            win_y_high = warped_img.shape[0] - window * window_height

            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            if draw:
                cv.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
                cv.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,255), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minimum:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minimum:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # ▶ 이상점 제거
        left_fit, right_fit = None, None
        if len(leftx) > 20:
            left_fit = np.polyfit(lefty, leftx, 2)
            residuals = np.abs(np.polyval(left_fit, lefty) - leftx)
            mask = residuals < 30
            leftx = leftx[mask]
            lefty = lefty[mask]
        if len(rightx) > 20:
            right_fit = np.polyfit(righty, rightx, 2)
            residuals = np.abs(np.polyval(right_fit, righty) - rightx)
            mask = residuals < 30
            rightx = rightx[mask]
            righty = righty[mask]

        # ▶ 픽셀 색상 표시
        out_img[lefty, leftx] = [100, 0, 0]
        out_img[righty, rightx] = [0, 0, 100]

        # ▶ 보간 곡선 시각화
        if draw:
            ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
            if left_fit is not None:
                left_fitx = np.polyval(left_fit, ploty)
                for i in range(len(ploty)-1):
                    cv.line(out_img, (int(left_fitx[i]), int(ploty[i])), (int(left_fitx[i+1]), int(ploty[i+1])), (255, 255, 0), 2)
            if right_fit is not None:
                right_fitx = np.polyval(right_fit, ploty)
                for i in range(len(ploty)-1):
                    cv.line(out_img, (int(right_fitx[i]), int(ploty[i])), (int(right_fitx[i+1]), int(ploty[i+1])), (0, 255, 255), 2)
        return {
            "image": out_img,
            "left": {
                "fit": left_fit,
                "x": leftx,
                "y": lefty,
            },
            "right": {
                "fit": right_fit,
                "x": rightx,
                "y": righty,
            }
        }
    #sliding window 중앙 기준으로
    #기본은 화면 끝에서부터 측정해서 멀리있는 차선이 인식되거나 할 경우 있음
    #중앙 기준의 경우 커브가 있어 중앙을 넘거나 중앙에 교통 마크가 있으면 문제 발생 가능성 있음
    def sliding_windows_visual_central(self, warped_img_ori, draw):

        warped_img = warped_img_ori.copy()
        height, width = warped_img.shape

        # ROI 마스킹
        mask = np.ones_like(warped_img, dtype=np.uint8) * 255
        cv.rectangle(mask, (0, height - 50), (width, height), 0, -1)
        warped_img = cv.bitwise_and(warped_img, mask)
        """
        plt.imshow(cv.cvtColor(warped_img, cv.COLOR_BGR2RGB))
        plt.show()
        """
        # 모폴로지 연산
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        warped_img = cv.morphologyEx(warped_img, cv.MORPH_OPEN, kernel)

        out_img = np.dstack((warped_img, warped_img, warped_img)) * 255

        # 중앙 기준 peak 검출
        histogram = np.sum(warped_img[height - height // 3:, :], axis=0)
        midpoint = width // 2
        threshold = 10000
        leftx_current = None
        rightx_current = None
        for i in range(midpoint, 0, -1):
            if histogram[i] > threshold:
                leftx_current = i
                break

        # 우측: 중앙에서 오른쪽으로 이동
        for i in range(midpoint, width):
            if histogram[i] > threshold:
                rightx_current = i
                break

        if  leftx_current == rightx_current:
            leftx_current = None
            rightx_current = None
            
        
        if  leftx_current == None:
            leftx_current = np.argmax(histogram[:midpoint])
        if rightx_current == None:
            rightx_current = np.argmax(histogram[midpoint:]) + midpoint

        """
        print(leftx_current)
        print(rightx_current)
        """
        """
        central_peak = np.argmax(histogram[midpoint - 100: midpoint + 100]) + (midpoint - 100)
        leftx_current = central_peak - (self.margin * 4)
        rightx_current = central_peak + (self.margin * 4)
        """
        """
        plt.plot(histogram)
        plt.title("Lane pixel histogram (bottom half)")
        plt.show()
        """
        window_height = height // self.nwindows
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            if window == 0:
                win_xleft_low = leftx_current - self.margin * 3
                win_xleft_high = leftx_current + self.margin * 3
                win_xright_low = rightx_current - self.margin * 3
                win_xright_high = rightx_current + self.margin * 3
            else:
                win_xleft_low = leftx_current - self.margin * 2
                win_xleft_high = leftx_current + self.margin * 2
                win_xright_low = rightx_current - self.margin * 2
                win_xright_high = rightx_current + self.margin * 2

            if draw:
                cv.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
                cv.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,255), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minimum:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minimum:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # 이상점 제거
        left_fit, right_fit = None, None
        if len(leftx) > 20:
            left_fit = np.polyfit(lefty, leftx, 2)
            residuals = np.abs(np.polyval(left_fit, lefty) - leftx)
            mask = residuals < 30
            leftx = leftx[mask]
            lefty = lefty[mask]
        if len(rightx) > 20:
            right_fit = np.polyfit(righty, rightx, 2)
            residuals = np.abs(np.polyval(right_fit, righty) - rightx)
            mask = residuals < 30
            rightx = rightx[mask]
            righty = righty[mask]

        # 시각화
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        
        ploty = np.linspace(0, height - 1, height)
        if left_fit is not None:
            left_fitx = np.polyval(left_fit, ploty)
            if draw:
                for i in range(len(ploty)-1):
                    cv.line(out_img, (int(left_fitx[i]), int(ploty[i])), (int(left_fitx[i+1]), int(ploty[i+1])), (255, 255, 0), 2)
        if right_fit is not None:
            right_fitx = np.polyval(right_fit, ploty)
            if draw:
                for i in range(len(ploty)-1):
                    cv.line(out_img, (int(right_fitx[i]), int(ploty[i])), (int(right_fitx[i+1]), int(ploty[i+1])), (0, 255, 255), 2)
        if left_fit is None or right_fit is None:
            return self.sliding_windows_visual(warped_img_ori, draw)
        else:
            if np.any(left_fitx >= right_fitx):
                return self.sliding_windows_visual(warped_img_ori, draw)
        return {
            "image": out_img,
            "left": {
                "fit": left_fit,
                "x": leftx,
                "y": lefty,
            },
            "right": {
                "fit": right_fit,
                "x": rightx,
                "y": righty,
            }
        }
    #위에있는 sliding window를 이용해 계산한 다항식을 기반으로 차선을 추적
    #처음 sliding window의 다항식을 쓰고 다음부터는 이 함수 스스로 계산한 다항식을 추적
    #속도가 빠른 대신 스스로 찾은 다항식을 추적하다보니 한번 엇나가면 복구가 힘들어 리셋 필요
    def quick_search(self, warped_img, draw):
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (np.polyval(self.prev_left_fit, nonzeroy) - self.margin)) &
                          (nonzerox < (np.polyval(self.prev_left_fit, nonzeroy) + self.margin)))

        right_lane_inds = ((nonzerox > (np.polyval(self.prev_right_fit, nonzeroy) - self.margin)) &
                           (nonzerox < (np.polyval(self.prev_right_fit, nonzeroy) + self.margin)))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        out_img = np.dstack((warped_img, warped_img, warped_img)) * 255
        left_fit, right_fit = None, None

        if len(leftx) > 20:
            left_fit = np.polyfit(lefty, leftx, 2)
            out_img[lefty, leftx] = [255, 0, 0]
        if len(rightx) > 20:
            right_fit = np.polyfit(righty, rightx, 2)
            out_img[righty, rightx] = [0, 0, 255]

        if draw:
            ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
            if left_fit is not None:
                left_fitx = np.polyval(left_fit, ploty)
                for i in range(len(ploty)-1):
                    cv.line(out_img, (int(left_fitx[i]), int(ploty[i])), (int(left_fitx[i+1]), int(ploty[i+1])), (0, 255, 255), 2)
            if right_fit is not None:
                right_fitx = np.polyval(right_fit, ploty)
                for i in range(len(ploty)-1):
                    cv.line(out_img, (int(right_fitx[i]), int(ploty[i])), (int(right_fitx[i+1]), int(ploty[i+1])), (255, 255, 0), 2)

        return {
            "image": out_img,
            "left": {
                "fit": left_fit,
                "x": leftx,
                "y": lefty,
            },
            "right": {
                "fit": right_fit,
                "x": rightx,
                "y": righty,
            }
        }


    # Warp image perspective
def warp(img, src=src, dst=dst):
    M = cv.getPerspectiveTransform(src, dst)
    return cv.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv.INTER_NEAREST)

def Re_warp(src=src, dst=dst):
    Minv = cv.getPerspectiveTransform(dst, src)
    return Minv

# Crop image for region of interest
def crop(img, ROI):
    # Create blank img with same size as input img
    blank = np.zeros(img.shape[:2], np.uint8)

    # Fill region of interest
    region_of_interest = cv.fillPoly(blank, ROI, 255)

    # Create image of interest with region (resize)
    return cv.bitwise_and(img, img, mask=region_of_interest)

# Merge to masks
def merge(frame, img1, img2):
    both = frame.copy()
    both[np.where(np.logical_and(img1==0, img2==0))] = 0
    return both

# Overlay two images
def overlay(img, overlay):
    img[np.where(overlay!=0)] = [0,255,0]
    return img

# Overlay two images with alpha
def overlay_alpha(img, overlay):
    return cv.addWeighted(img, 1, overlay.astype(np.uint8), 0.5, 0.0)


def detect_dash_line_along_curve(binary_img, fit, ploty, threshold_gap=50, threshold_segment=50):
    """
    다항 곡선을 따라 점선인지 실선인지 분석.
    - binary_img: 흑백 이미지 (차선만 흰색)
    - fit: np.polyfit으로 얻은 다항 계수
    - ploty: y좌표 배열
    - threshold_gap: 점선 판단 기준이 되는 최소 gap
    - threshold_segment: 실선 판단 기준이 되는 최소 선의 길이
    """
    gaps = []
    segments = []
    last_seen = None
    current_segment = 0
    
    for i, y in enumerate(ploty.astype(int)):
        x = int(np.polyval(fit, y))
        if 0 <= y < binary_img.shape[0] and 0 <= x < binary_img.shape[1]:
            radius = 20
            if np.any(binary_img[y, max(x - radius, 0): min(x + radius + 1, binary_img.shape[1])] > 0) or i == len(ploty) - 1:
                if last_seen is not None:
                    gap = y - last_seen
                    if gap > threshold_gap:
                        gaps.append(gap)
                        if current_segment > 20:
                            segments.append(current_segment)
                        current_segment = 0
                    else:
                        current_segment += gap
                last_seen = y

    if current_segment > 0:
        segments.append(current_segment)

    # 디버깅 출력
    """
    print("평균 gap:", np.mean(gaps) if gaps else 0)
    print("gap 표준편차:", np.std(gaps) if gaps else 0)
    print("gap 개수:", len(gaps))
    print("평균 segment 길이:", np.mean(segments) if segments else 0)
    print("segment 개수:", len(segments))
    """
    # 판단 조건
    if len(gaps) > 2 and np.mean(gaps) > threshold_gap and len(segments) >= 2:
        return "dashed"
    else:
        return "solid"


#결과인 다항식을 바탕으로 선을 시각화
def draw_lane_curve(binary_img, fit, ploty, line_type):
    color = (0, 255, 0) if line_type == "solid" else (0, 0, 255)
    curve_img = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)
    x_vals = np.polyval(fit, ploty).astype(np.int32)
    for i in range(len(ploty)-1):
        pt1 = (x_vals[i], int(ploty[i]))
        pt2 = (x_vals[i+1], int(ploty[i+1]))
        if 0 <= pt1[0] < binary_img.shape[1] and 0 <= pt2[0] < binary_img.shape[1]:
            cv.line(curve_img, pt1, pt2, color, 3)
    return curve_img


def blend_transparent_overlay(base_img, overlay_mask, color=(0, 255, 255), alpha=0.4):
    """
    base_img: 원본 BGR 이미지
    overlay_mask: 채워질 영역 (uint8 마스크, 255가 채워질 부분)
    color: 덧씌울 색상 (BGR)
    alpha: 투명도 (0: 완전 투명, 1: 불투명)
    """
    overlay = np.zeros_like(base_img, dtype=np.uint8)
    overlay[:] = color
    mask_3ch = cv.merge([overlay_mask] * 3)
    blended = cv.addWeighted(base_img, 1, cv.bitwise_and(overlay, mask_3ch), alpha, 0)
    return blended

def draw_lane_area_with_labels(original_img, left_fit, right_fit, warped_shape, Minv,
                               left_type="unknown", right_type="unknown",
                               left_color=(0, 255, 0), right_color=(255, 0, 0), fill_color=(0, 255, 255)):

    ploty = np.linspace(0, warped_shape[0] - 1, warped_shape[0])

    if left_fit is not None:
        left_fitx = np.polyval(left_fit, ploty)
        left_pts = np.array([[left_fitx[i], ploty[i]] for i in range(len(ploty))], dtype=np.float32).reshape(-1, 1, 2)
        left_unwarped = cv.perspectiveTransform(left_pts, Minv)

    if right_fit is not None:
        right_fitx = np.polyval(right_fit, ploty)
        right_pts = np.array([[right_fitx[i], ploty[i]] for i in range(len(ploty))], dtype=np.float32).reshape(-1, 1, 2)
        right_unwarped = cv.perspectiveTransform(right_pts, Minv)

    # 역투영
    
    

    # 차선 영역 그리기

    

    result = original_img.copy()

    if left_fit is not None and right_fit is not None:
        lane_poly = np.vstack((left_unwarped, np.flipud(right_unwarped)))
    #cv.fillPoly(result, [np.int32(lane_poly)], fill_color)

        lane_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
        cv.fillPoly(lane_mask, [np.int32(lane_poly)], 255)
        result = blend_transparent_overlay(original_img, lane_mask, color=fill_color, alpha=0.4)

        # 좌우 선 그리기
        for i in range(len(ploty) - 1):
            pt1_l = tuple(np.int32(left_unwarped[i][0]))
            pt2_l = tuple(np.int32(left_unwarped[i + 1][0]))
            pt1_r = tuple(np.int32(right_unwarped[i][0]))
            pt2_r = tuple(np.int32(right_unwarped[i + 1][0]))

            cv.line(result, pt1_l, pt2_l, left_color, 3)
            cv.line(result, pt1_r, pt2_r, right_color, 3)

    # 박스 그리기: 좌우 차선 경계 사각형
        def draw_label_box(unwarped_pts, color, label):
            xs = [pt[0][0] for pt in unwarped_pts]
            ys = [pt[0][1] for pt in unwarped_pts]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            cv.rectangle(result, (x_min, y_min), (x_max, y_max), color, 2)
            cv.putText(result, label, (x_min, y_min - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)

    # 박스 + 라벨 추가
        draw_label_box(left_unwarped, left_color, f"Left: {left_type}")
        draw_label_box(right_unwarped, right_color, f"Right: {right_type}")

    return result


def clahe(img):

    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv.merge((l_clahe, a, b))
    img_clahe = cv.cvtColor(lab_clahe, cv.COLOR_LAB2BGR)

    return img_clahe


def hls_clahe(img):
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    h, l, s = cv.split(hls)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    hls_eq = cv.merge((h, l_eq, s))
    img_clahe = cv.cvtColor(hls_eq, cv.COLOR_HLS2BGR)
    return img_clahe


def get_region_brightness(image, x_ratio=0.5, y_ratio=0.6, region_size=0.2):
    height, width = image.shape[:2]
    x_start = int(width * (x_ratio - region_size / 2))
    x_end = int(width * (x_ratio + region_size / 2))
    y_start = int(height * y_ratio)
    y_end = int(height * (y_ratio + region_size))

    region = image[y_start:y_end, x_start:x_end]


    brightness = np.mean(region)
    return brightness

def combined_threshold(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 소벨 에지 검출
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    #cv.imshow('scaled_sobel1', scaled_sobel)
    # 임계값 적용
    thresh_min = 15
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    """
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(sxbinary == 1)] = 255
    cv.imshow('sxbinary', combined_binary)
    """
    # 색상 임계값
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_thresh_min = 90
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    """
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary == 1)] = 255
    cv.imshow('s_binary', combined_binary)
    """
    # 결합
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(sxbinary == 1) | (s_binary == 1)] = 255
    return combined_binary

def line_check(frame, src, dst, LT):
    orig = frame.copy()

    # 흰색 범위 (value가 높고 saturation이 낮음)
    """white_lower = np.array([0, 0, 200])
    white_upper = np.array([255, 100, 255])

    # 노란색 범위
    yellow_lower = np.array([15, 100, 200])
    yellow_upper = np.array([35, 255, 255])


    hsv = cv.cvtColor(orig, cv.COLOR_BGR2HSV)

    white_mask = cv.inRange(hsv, white_lower, white_upper)
    yellow_mask = cv.inRange(hsv, yellow_lower, yellow_upper)

    lane_mask = cv.bitwise_or(white_mask, yellow_mask)

    h, s, v = cv.split(hsv)
    #v = cv.add(v, -100)  # 밝기 증가 (0~255 클램핑됨)
    s = cv.add(s, -50)  # 밝기 증가 (0~255 클램핑됨)
    final_hsv = cv.merge((h, s, v))

    masked = cv.bitwise_and(img, img, mask=lane_mask)
    """

    img_clahe = hls_clahe(orig)

    blurred = cv.GaussianBlur(img_clahe, (5, 5), 0)
    color = color_space_hls(blurred)

    sobel_test = combined_threshold(blurred)

    binary_result = open_img(sobel_test, 1)
    """
    cv.imshow('sobel_test', sobel_test)
    cv.imshow('binary_result', binary_result)
    """
    #return cv.bitwise_and(binary_result, binary_result, mask=shadow_mask)
    color = warp(binary_result, src, dst)



    # Step 3: Sliding windows to get curve points    
    #midpoint, lefts, rights = sliding_windows(color)

    
    #result = central_sliding_windows_based_on_existing(color, nwindows= 5, minimum =100, draw=True)
    result = LT.update(color)
    #return result["image"]
    # Step 4: Calculate curve and radius
    #left_fit, right_fit, radius = calc_curve(lefts, rights, orig)

    lower_red = np.array([0, 0, 200])
    upper_red = np.array([50, 50, 255])
    mask_red = cv.inRange(result["image"], lower_red, upper_red)

    # 파란색 마스크
    lower_blue = np.array([200, 0, 0])
    upper_blue = np.array([255, 50, 50])
    mask_blue = cv.inRange(result["image"], lower_blue, upper_blue)

    # 두 마스크 합치기
    mask_combined = cv.bitwise_or(mask_red, mask_blue)

    # 마스크 결과를 흑백 형태로 출력 (255: 흰색, 나머지: 검정)
    output = cv.cvtColor(mask_combined, cv.COLOR_GRAY2BGR)  # 필요 시 RGB 이미지로 변환


    ploty = np.linspace(0, result["image"].shape[0] - 1, num=result["image"].shape[0])
    
    if result["left"]["fit"] is not None:
        left_line_type = detect_dash_line_along_curve(mask_combined, result["left"]["fit"], ploty)
        left_lane_img = draw_lane_curve(mask_combined, result["left"]["fit"], ploty, left_line_type)
    else:
        left_line_type = "unknown"

    
    if result["right"]["fit"] is not None:
        right_line_type = detect_dash_line_along_curve(mask_combined, result["right"]["fit"], ploty)
        right_lane_img = draw_lane_curve(mask_combined, result["right"]["fit"], ploty, right_line_type)
    else:
        right_line_type = "unknown"

    if result["left"]["fit"] is not None:
        overlay = cv.addWeighted(output, 1.0, left_lane_img, 0.8, 0)
        #gap_draw_left = draw_gap_markers(mask_combined, result["left"]["fit"], ploty)
    if result["right"]["fit"] is not None:
        overlay = cv.addWeighted(output, 1.0, right_lane_img, 0.8, 0)

        #gap_draw_right = draw_gap_markers(mask_combined, result["right"]["fit"], ploty)

        if result["left"]["fit"] is not None:
            pass
            #gap_draw = cv.bitwise_or(gap_draw_left, gap_draw_right)
            #return gap_draw

    # When everything done, release the video capture object

    Minv = Re_warp(src, dst)
    result = draw_lane_area_with_labels(
        #original_img=cv.cvtColor(orig, cv.COLOR_BGR2RGB),
        original_img=orig,
        left_fit=result["left"]["fit"],
        right_fit=result["right"]["fit"],
        warped_shape=color.shape,
        Minv=Minv,
        left_color=(0, 255, 0),     # 초록
        right_color=(255, 0, 0),    # 파랑
        fill_color=(0, 255, 255),    # 차선 사이 채우기 (노랑)
        left_type=left_line_type,
        right_type=right_line_type
    )
    return result



# Open video file


#여기서부터 주요 예시
#video_file = 'project'

def example():
    cap = cv.VideoCapture('challenge_video.mp4')
    #cap = cv.VideoCapture('police.mp4')

    LT = LaneTracker(margin=50)
    #img = cv.imread('lane.jpg')

    # Manually turn car detection on or off
    CAR_DETECTION = False


    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    ret, frame = cap.read()

    start_timer = time.time() - 0.01
    frame_count = -1
    # If challenge video is played -> Define different points for transformation 
    height, width = frame.shape[:2]
    src = np.float32([
        [width * 0.45, height * 0.65],
        [width * 0.55, height * 0.65],
        [width * 0.9, height],
        [width * 0.1, height]
    ])
    dst = np.float32([
        [width * 0.3, 0],
        [width * 0.7, 0],
        [width * 0.7, height],
        [width * 0.3, height]
    ])

    while(cap.isOpened()):
        ret, frame = cap.read()
        if (frame is None):
            break
        frame = line_check(frame, src, dst, LT)

        cv.imshow('Frame', frame)

        if ret == True:   
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    cv.destroyAllWindows()
if __name__ == "__main__":
    example()