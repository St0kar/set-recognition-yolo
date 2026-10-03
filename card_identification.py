import os
from ultralytics import YOLO

# ========================== Variables ==========================
# Path to your trained model (best.pt or last.pt)
MODEL_PATH   = "runs/train/exp1/weights/best.pt"
# Inference settings
IMG_SIZE     = 640    # pixels (resize input)
CONF_THRESH  = 0.8   # confidence threshold
# ======================================================

def run_inference(img_path):
    # Load model
    model = YOLO(MODEL_PATH)

    # Run prediction
    results = model.predict(
        source=img_path,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        save=False,
        exist_ok=True
    )

    return results

def main():
    img_path = "test/img_test2.jpg"
    cards = run_inference(img_path)
    for c in cards:
        print(c.boxes)


if __name__ == "__main__":
    main()