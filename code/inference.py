# import cv2
# import argparse
# import pandas as pd
# import os

# from utils import resize_frame, draw_trajectory
# from track import SimpleTracker

# def detect_ball(gray):
#     """
#     Detect ball using HoughCircles
#     Returns list of (x, y)
#     """
#     circles = cv2.HoughCircles(
#         gray,
#         cv2.HOUGH_GRADIENT,
#         dp=1.2,
#         minDist=20,
#         param1=50,
#         param2=30,
#         minRadius=3,
#         maxRadius=40
#     )

#     detections = []
#     if circles is not None:
#         circles = circles[0]
#         for x, y, r in circles:
#             detections.append((int(x), int(y)))

#     return detections

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_video", type=str, required=True)
#     parser.add_argument("--output_csv", type=str, default="annotations/output.csv")
#     parser.add_argument("--output_video", type=str, default="results/output.mp4")
#     args = parser.parse_args()

#     os.makedirs("annotations", exist_ok=True)
#     os.makedirs("results", exist_ok=True)

#     cap = cv2.VideoCapture(args.input_video)
#     if not cap.isOpened():
#         print("Error opening video")
#         return

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     ret, frame = cap.read()
#     frame = resize_frame(frame)
#     h, w = frame.shape[:2]

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(args.output_video, fourcc, fps, (w, h))

#     tracker = SimpleTracker()
#     trajectory = []
#     rows = []

#     frame_idx = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = resize_frame(frame)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray, (5, 5), 0)

#         detections = detect_ball(gray)
#         center = tracker.update(detections)

#         if center is not None:
#             x, y = center
#             rows.append([frame_idx, x, y, 1])
#             trajectory.append((x, y))
#             cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
#         else:
#             rows.append([frame_idx, -1, -1, 0])
#             trajectory.append(None)

#         draw_trajectory(frame, trajectory)
#         out.write(frame)

#         frame_idx += 1

#     cap.release()
#     out.release()

#     df = pd.DataFrame(rows, columns=["frame", "x", "y", "visible"])
#     df.to_csv(args.output_csv, index=False)

#     print("Done!")
#     print("CSV saved to:", args.output_csv)
#     print("Video saved to:", args.output_video)

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os

class KalmanTracker:
    """
    Acts as the 'Temporal Reasoning' module. 
    Maintains ball state even when detections are missing due to blur.
    """
    def __init__(self):
        # 4 state variables (x, y, dx, dy), 2 measurement variables (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        self.last_pos = None

    def predict_and_update(self, measurement):
        prediction = self.kf.predict()
        if measurement is not None:
            self.kf.correct(np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]]))
            self.last_pos = measurement
            return measurement
        return (int(prediction[0]), int(prediction[1]))

def run_test_inference(video_name):
    # 1. SETUP PATHS
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Use your custom trained model weights
    model_path = os.path.join(BASE_DIR, 'cricket_ball_tracker', 'v1_motion_blur_fix', 'weights', 'best.pt')
    video_path = os.path.join(BASE_DIR, 'data', '25_nov_2025', video_name)
    
    results_dir = os.path.join(BASE_DIR, 'results2')
    annotations_dir = os.path.join(BASE_DIR, 'annotations')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    out_vid_path = os.path.join(results_dir, f'{video_name}_out.mp4')
    out_csv_path = os.path.join(annotations_dir, f'{video_name}_out.csv')

    # 2. INITIALIZE MODULES
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    tracker = KalmanTracker()
    
    results_list = []
    frame_idx = 0
    
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    out = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"Processing: {video_name}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 3. DETECTION
        # Runs on every frame to find the ball
        detect_results = model.predict(frame, conf=0.25, verbose=False)
        
        ball_coords = None
        for box in detect_results[0].boxes:
            if int(box.cls[0]) == 0: # Assumes class 0 is 'cricket_ball'
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                ball_coords = (int((x1+x2)/2), int((y1+y2)/2))
                break 

        # 4. TRACKING (Temporal Reasoning)
        # Bridges frames where the detector misses the ball
        final_x, final_y = tracker.predict_and_update(ball_coords)
        visible = 1 if ball_coords else 0

        # 5. VISUAL OVERLAY (Single Dot Only)
        # Draw only the current ball position as a solid green dot
        cv2.circle(frame, (final_x, final_y), 6, (0, 255, 0), -1)
        
        # 6. DATA RECORDING
        results_list.append([frame_idx, round(float(final_x), 1), round(float(final_y), 1), visible])
        
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    
    # 7. SAVE CSV
    df = pd.DataFrame(results_list, columns=['frame', 'x', 'y', 'visible'])
    df.to_csv(out_csv_path, index=False)
    print(f"Completed! Output saved to results2/ and annotations/")


if __name__ == "__main__":
    run_test_inference("1.mp4")
    for i in range(2, 16):  # from 2 to 15 inclusive
        filename = f"{i}.mov"
        print(f"\nRunning inference on: {filename}")
        run_test_inference(filename)
