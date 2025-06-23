import cv2 as cv
import numpy as np
from ultralytics import YOLO

# === 사용자 설정 ===
video_path = r"C:\Users\USER\Downloads\3team3-best\sample2.mp4"  # 영상 경로
model_path = r"C:\Users\USER\Downloads\3team3-best\best.pt"      # YOLO 모델 경로

# === Kalman Filter (차선 중심 x좌표 예측용) ===
lane_kalman = cv.KalmanFilter(2, 1)
lane_kalman.measurementMatrix = np.array([[1, 0]], np.float32)
lane_kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
lane_kalman.processNoiseCov = np.eye(2, dtype=np.float32) * 0.03
lane_kalman.measurementNoiseCov = np.array([[1]], np.float32) * 0.5

# === 차선 색상 필터 ===
def color_space(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    mask_yellow = cv.inRange(img_hsv, (15, 90, 140), (40, 255, 255))
    mask_white = cv.inRange(img_hsv, (0, 0, 210), (180, 40, 255))
    mask_red1 = cv.inRange(img_hsv, (0, 100, 100), (10, 255, 255))
    mask_red2 = cv.inRange(img_hsv, (170, 100, 100), (180, 255, 255))
    mask_red = cv.bitwise_or(mask_red1, mask_red2)

    mask = cv.bitwise_or(mask_white, mask_yellow)
    mask = cv.bitwise_or(mask, mask_red)

    result = cv.bitwise_and(img, img, mask=mask)
    return cv.cvtColor(result, cv.COLOR_BGR2GRAY)

# === ROI 설정 (하단 사각형 영역) ===
def region_of_interest(img):
    height, width = img.shape
    polygon = np.array([[
        (int(0.2 * width), height),
        (int(0.8 * width), height),
        (int(0.6 * width), int(0.65 * height)),
        (int(0.4 * width), int(0.65 * height))
    ]], np.int32)
    mask = np.zeros_like(img)
    cv.fillPoly(mask, polygon, 255)
    return cv.bitwise_and(img, mask)

# === 차선 검출 및 Kalman 필터 적용 ===
def detect_lines(img, original_img):
    lines = cv.HoughLinesP(img, 1, np.pi / 180, threshold=100,
                           minLineLength=100, maxLineGap=20)
    line_img = np.zeros_like(original_img)
    lane_center_xs = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cx = int((x1 + x2) / 2)
            lane_center_xs.append(cx)
            cv.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    if lane_center_xs:
        avg_x = int(np.mean(lane_center_xs))
        measured = np.array([[np.float32(avg_x)]])
        lane_kalman.correct(measured)
        predicted = lane_kalman.predict()
        predicted_x = int(predicted[0])
        height = original_img.shape[0]
        cv.line(line_img, (predicted_x, height), (predicted_x, int(height * 0.6)), (255, 0, 0), 2)

    return cv.addWeighted(original_img, 0.8, line_img, 1, 1)

# === 메인 실행 ===
def main():
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 영상 열기 실패: {video_path}")
        return

    print("✅ 영상 열기 성공. YOLO 모델 로딩 중...")
    model = YOLO(model_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("🔚 영상 종료")
            break

        masked = color_space(frame)
        edges = cv.Canny(masked, 80, 200)
        roi = region_of_interest(edges)
        lane_frame = detect_lines(roi, frame.copy())

        results = model.predict(frame, verbose=False)[0]
        yolo_frame = results.plot()

        combined = cv.addWeighted(lane_frame, 0.5, yolo_frame, 0.5, 0)

        # === 프레임 크기 축소 표시 ===
        scale = 0.6
        resized = cv.resize(combined, None, fx=scale, fy=scale)
        cv.imshow("YOLO + Lane Detection with Kalman Filter", resized)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

