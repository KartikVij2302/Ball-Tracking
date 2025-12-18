from ultralytics import YOLO
import os

def train_cricket_ball_detector():
    # 1. Load the base model
    # 'yolov8n.pt' is the Nano version (fastest). 
    # Use 'yolov8m.pt' (Medium) if you have a powerful GPU (RTX 3060 or better).
    model = YOLO('yolov8n.pt') 
    
    # 2. Get the absolute path to your data.yaml
    # This ensures the model finds the file regardless of where you run the script
    yaml_path = os.path.abspath('data.yaml')

    print(f"Starting training using config: {yaml_path}")

    # 3. Train the model
    results = model.train(
        data=yaml_path,
        
        # Training Duration
        epochs=50,              # 50-100 is usually good for a single object
        patience=10,            # Stop if no improvement for 10 epochs
        
        # Image Settings
        imgsz=640,              # Higher resolution helps spot tiny balls
        batch=16,               # Reduce to 8 or 4 if you run out of GPU memory
        
        # OPTIMIZATIONS FOR CRICKET BALLS (The "Blur" Fix)
        # These augmentations teach the model to recognize distorted/blurred balls
        degrees=10.0,           # Slight rotation
        shear=2.5,              # Shearing mimics the "streak" of a fast ball
        fliplr=0.5,             # Flip left-right to double dataset variety
        mosaic=1.0,             # Stitches 4 images together (crucial for small objects)
        mixup=0.1,              # Blends images to help with occlusion (batsman blocking ball)
        
        # System Settings
        project='cricket_ball_tracker',
        name='v1_motion_blur_fix',
        exist_ok=True,          # Overwrite previous run if it exists
    )
    
    print("\nTraining Complete!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train_cricket_ball_detector()