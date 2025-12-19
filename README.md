


# Cricket Ball Tracking System – Report

## 🎯 Objective

To build a robust, reproducible computer vision system that detects and tracks a **cricket ball** in match videos recorded from a **static camera**, producing:

- Per-frame ball centroid annotations (`frame, x, y, visible`)
- Processed videos without the trajectory overlay
- Code, model, and results for evaluation and replication

This system TRIES to addresses challenges like:
- Motion blur due to fast-moving balls
- Variable ball color (white or red)
- Missed detections due to occlusion or low contrast

---

## 📦 Repository Structure



Ball-Tracking/
│
├── code/
│   ├── train_yolo.py              # YOLOv8 training script
│   ├── inference_tracker.py       # Inference + Kalman tracking
│
├── data/
│   └── 25_nov_2025/               # Testing videos (2.mov to 15.mov)
│
├── cricket_ball_tracker/
│   └── v1_motion_blur_fix/        # Trained YOLOv8 model + weights
│
├── annotations/                   # CSV files (frame,x,y,visible)
├── results2/                      # Annotated output videos
├── requirements.txt               # Python dependencies
└── report.md                      # Project report (modeling, tracking, limitations)




---

## 🧠 Model: YOLOv8 with Blur-Resilient Augmentation

We use the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) Nano (`yolov8n.pt`) model, fine-tuned to detect cricket balls using custom-labeled training data.

### ⚙️ Key Training Details

- **Base model**: `yolov8n.pt` (fast and lightweight, optimized for GTX 1650 GPU)
- **Training duration**: 200 epochs, early stopping after 30 epochs of no improvement
- **Image size**: 1280 × 1280 (large resolution helps detect tiny, fast-moving balls)
- **Batch size**: 4 (chosen to fit GTX 1650 memory)
- **Loss objective**: objectness + bounding box regression for single-class ball detection

### 🧪 Data Augmentations (for motion blur & occlusion)

| Augmentation | Purpose |
|-------------|---------|
| `degrees=10.0` | Slight rotation for realism |
| `shear=2.5` | Mimics blur streaks from fast balls |
| `fliplr=0.5` | Horizontal flip to double training diversity |
| `mosaic=1.0` | Blends 4 images to simulate small-object crowding |
| `close_mosaic=10` | Forces close-up mosaic for small balls |
| `mixup=0.1` | Blends ball + occluder (e.g., batsman/umpire) |

### 📁 Training Data Format

The training data is organized in YOLO format, with a `data.yaml` pointing to image and label folders, and one class: `"cricket_ball"`.

---

## 🕵️ Inference + Tracking Pipeline

During evaluation, we process `.mp4` and `.mov` videos and output:

- Annotated video with ball overlay (`results2/`)
- Per-frame CSV with centroid and visibility flag (`annotations/`)

### 🧩 Detection + Temporal Reasoning

We combine:
1. **YOLOv8 Inference** on each frame
2. **Kalman Filter Tracker** to bridge missed detections

```python
# If ball is detected → update tracker
# If no detection → predict ball location using motion model
````

This approach stabilizes predictions and handles motion blur or short occlusion gaps.

### 🔁 Loop Summary

For each video frame:

* Run YOLO model: `model.predict(frame)`
* If class 0 (`cricket_ball`) is detected:

  * Compute bounding box center `(x, y)`
  * Pass to Kalman filter (`correct()`)
* Else:

  * Use Kalman prediction
* Save `frame_idx, x, y, visible` and overlay circle on frame

---

## 📄 Output Format

### CSV (annotations/)

```csv
frame,x,y,visible
0,512.3,298.1,1
1,518.7,305.4,1
2,-1,-1,0
```

* `x,y` = centroid (or -1 if missing)
* `visible = 1` if detection made, `0` if predicted only

### MP4 (results2/)

* Output video with green dot on ball and smooth trajectory
* Same resolution, frame rate as original

---

## 🔁 Batch Inference

```python
for i in range(2, 16):
    run_test_inference(f"{i}.mov")
```

* Loops through all videos `2.mov` to `15.mov` in the `data/25_nov_2025/` folder
* Saves annotated results to `results2/`, CSVs to `annotations/`

---

## 🛠️ Assumptions & Design Choices

| Choice                   | Rationale                                                             |
| ------------------------ | --------------------------------------------------------------------- |
| **YOLOv8n**              | Lightweight, fits on GTX 1650, sufficient for single-object detection |
| **Kalman Filter**        | Temporal smoothing without training a sequence model                  |
| **Single-class model**   | Simplifies learning; no need to distinguish object types              |
| **High-res images**      | Better accuracy for small cricket ball targets                        |
| **Per-frame prediction** | Matches task spec (spotting ball per frame, not per segment)          |

---

## ✅ Performance Highlights

* **Handles white & red balls** (model trained on both)
* Robust to **motion blur**, **occlusion**, and **lighting variation**
* Accurate tracking even during partial ball visibility
* End-to-end reproducible with GPU acceleration

---

## 📌 Possible Improvements

| Feature                | Next Steps                                                 |
| ---------------------- | ---------------------------------------------------------- |
| Multi-frame modeling   | Use a CNN + GRU to model temporal features (like E2E-Spot) |
| Confidence smoothing   | Weight detection confidence into Kalman update             |
| Auto-tuning thresholds | Adjust based on video resolution or brightness             |
| Real-time mode         | Optimize pipeline to run live with low-latency inference   |

---

## 🖥️ Hardware & Environment

* **GPU**: NVIDIA GeForce GTX 1650 (4 GB VRAM)
* **Training Duration**: ~60–90 mins for 200 epochs (depending on dataset size)
* **Python Libraries**:

  * `ultralytics`
  * `opencv-python`
  * `numpy`, `pandas`

---

## 🔁 Reproducibility Instructions

1. Clone the repo:

   ```bash
   git clone https://github.com/KartikVij2302/Ball-Tracking.git
   cd Ball-Tracking
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train model (optional):

   ```bash
   python code/train.py
   ```

4. Run inference:

   ```bash
   python code/inference.py
   ```

---

## 📎 Appendix

* `annotations/`: CSVs per video
* `results2/`: processed MP4s
* `cricket_ball_tracker/.../best.pt`: trained YOLOv8 model
* `data.yaml`: defines image path, class label
* `train.py`: model training script
* `inference.py`: test video evaluation

---



### ❌ Known Limitations and Failure Cases

Despite strong results on many clips, the system has several known weaknesses due to architectural and data constraints:

---

#### 1. **Missed Detections Under Severe Motion Blur**

* **Why it happens**: YOLOv8 relies on visual contrast and spatial patterns. If the ball moves extremely fast, it appears as a faint streak or blends into the pitch, making it hard to detect.
* **Effect**: YOLO may return no box, and the Kalman filter can drift over multiple missed frames.

---

#### 2. **False Positives on Static White/Red Objects**

* **Why it happens**: If logos, shoes, or pitch markings resemble the ball in color/shape, YOLO may falsely detect them — especially when the actual ball is temporarily occluded.
* **Effect**: Sudden jumps in trajectory, incorrect centroid output.

---

#### 3. **No Temporal Feature Learning**

* **Why it happens**: The model runs frame-by-frame without learning motion cues. It doesn't know that the object "must follow a physical trajectory" or have momentum.
* **Effect**: It fails to distinguish a fast-moving ball from other static ball-like artifacts.

> ⚠️ **In contrast**, approaches like [E2E-Spot](#) use CNN+GRU to model temporal dependencies, enabling smoother and more accurate predictions across time.

---

#### 4. **Single Detection Per Frame**

* **Why it happens**: The script exits after the **first valid box** per frame (`break`), assuming there's only one ball.
* **Effect**: In rare cases (e.g. multiple balls in warm-up), the detector may pick the wrong one.

---

#### 5. **YOLOv8 Struggles on Low-Contrast Frames**

* **Why it happens**: On dusty pitches or when the ball shadow overlaps with player shadows, the contrast between ball and background is poor.
* **Effect**: Detection confidence drops; YOLO fails to detect.

---

#### 6. **No Ball Class Differentiation**

* **Why it happens**: The model is trained on a single class `"cricket_ball"` without explicitly distinguishing red vs white balls.
* **Effect**: May overfit to the more frequent color or behave inconsistently across lighting conditions.

---

#### 7. **No Real-Time Optimization**

* **Why it happens**: The system assumes offline inference (e.g., Kalman filter depends on full video availability and OpenCV's slow prediction loop).
* **Effect**: Latency would be too high for live broadcast or umpire assistance.

---

### 🔁 Summary of Root Causes

| Root Issue                         | Impact on Model             | Suggested Remedy                                       |
| ---------------------------------- | --------------------------- | ------------------------------------------------------ |
| No temporal CNN features           | Misunderstands ball streaks | Use 3D CNN or CNN+GRU (E2E-Spot style)                 |
| Kalman-only smoothing              | Inaccurate long gaps        | Use learned motion priors or optical flow              |
| Small training set                 | Poor generalization         | Augment with synthetic blurred balls                   |
| Single object per frame assumption | Can skip the real ball      | Apply detection ranking or proximity to previous point |
| No multi-scale training            | Inconsistent ball sizing    | Include multi-resolution training & detection          |


