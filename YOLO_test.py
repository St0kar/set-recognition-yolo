import os
from ultralytics import YOLO

# ========================== Variables ==========================
# Path to your trained model (best.pt or last.pt)
MODEL_PATH   = "runs/train/exp1/weights/best.pt"
# Path to the test image (or folder of images)
IMAGE_PATH   = "test/img_test3.jpg"
# Optional: directory to save inference outputs
OUTPUT_DIR   = "runs/detect"
# Inference settings
IMG_SIZE     = 640    # pixels (resize input)
CONF_THRESH  = 0.25   # confidence threshold
# ======================================================

def run_inference():
    """
    Loads the trained model and runs inference on the specified image.
    Saves results (images with boxes) under OUTPUT_DIR/test.
    """
    # Create output folder
    save_dir = os.path.join(OUTPUT_DIR, "test")
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model = YOLO(MODEL_PATH)

    # Run prediction
    results = model.predict(
        source=IMAGE_PATH,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        save=True,
        project=OUTPUT_DIR,
        name="test",
        exist_ok=True
    )

    # Print summary
    print(f"Inference completed. Results saved to {save_dir}")
    for r in results:
        print(r)  # prints detection summary

if __name__ == "__main__":
    run_inference()