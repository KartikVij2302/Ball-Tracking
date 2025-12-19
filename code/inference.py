

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
        cv2.circle(frame, (final_x, final_y), 9, (0, 255, 0), -1)
        
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
