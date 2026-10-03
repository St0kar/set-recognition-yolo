import os
import yaml
import random
import shutil
from glob import glob
from ultralytics import YOLO

# ========================== Variables ==========================
# Paths
DATASET_DIR      = "AugCards"  # Root dir containing `images/` and `labels/` folders with all data
ORIG_IMAGES_DIR  = os.path.join(DATASET_DIR, "images")
ORIG_LABELS_DIR  = os.path.join(DATASET_DIR, "labels")

TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
VAL_IMAGES_DIR   = os.path.join(DATASET_DIR, "images", "val")
TRAIN_LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")
VAL_LABELS_DIR   = os.path.join(DATASET_DIR, "labels", "val")

# Split ratio (e.g., 0.2 for 20% validation)
VAL_SPLIT        = 0.2

# Output dataset YAML
DATA_YAML        = "data.yaml"

# Model and training hyperparameters
PRETRAINED_MODEL = "yolov8n.pt"
EPOCHS           = 50
BATCH_SIZE       = -1
IMG_SIZE         = 640  # pixels
LR0              = 0.0001
DEVICE           = 0   # GPU ID or 'cpu'
WORKERS          = 4
PATIENCE         = 5
DROPOUT          = 0.4

# Project settings
PROJECT          = "runs/train"
EXPERIMENT_NAME  = "exp1"
SAVE_PERIOD      = 1   # Save checkpoint every n epochs

# Number of classes and class names
NUM_CLASSES      = 1
CLASS_NAMES      = ["card"]
# ==============================================================

def split_dataset(val_split=VAL_SPLIT, seed=42):
    """
    Splits the original dataset into train and validation sets.
    """
    # Create target directories
    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
    os.makedirs(VAL_LABELS_DIR, exist_ok=True)

    # Gather all images
    image_files = glob(os.path.join(ORIG_IMAGES_DIR, "*.*"))
    random.seed(seed)
    random.shuffle(image_files)

    # Split
    num_val = int(len(image_files) * val_split)
    val_files = image_files[:num_val]
    train_files = image_files[num_val:]

    # Copy files
    for filepath in train_files:
        fname = os.path.basename(filepath)
        label_file = os.path.join(ORIG_LABELS_DIR, os.path.splitext(fname)[0] + ".txt")
        shutil.copy(filepath, os.path.join(TRAIN_IMAGES_DIR, fname))
        if os.path.exists(label_file):
            shutil.copy(label_file, os.path.join(TRAIN_LABELS_DIR, os.path.basename(label_file)))

    for filepath in val_files:
        fname = os.path.basename(filepath)
        label_file = os.path.join(ORIG_LABELS_DIR, os.path.splitext(fname)[0] + ".txt")
        shutil.copy(filepath, os.path.join(VAL_IMAGES_DIR, fname))
        if os.path.exists(label_file):
            shutil.copy(label_file, os.path.join(VAL_LABELS_DIR, os.path.basename(label_file)))

    print(f"Split {len(train_files)} train and {len(val_files)} val samples.")


def create_data_yaml():
    """
    Generates a data.yaml for YOLOv8 training.
    """
    data = {
        "train": TRAIN_IMAGES_DIR,
        "val":   VAL_IMAGES_DIR,
        "nc":    NUM_CLASSES,
        "names": CLASS_NAMES
    }
    with open(DATA_YAML, "w") as f:
        yaml.dump(data, f)
    print(f"Created {DATA_YAML} with content:\n{data}")


def train():
    """
    Launches YOLOv8 training.
    """
    model = YOLO(PRETRAINED_MODEL)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        lr0=LR0,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
        name=EXPERIMENT_NAME,
        save_period=SAVE_PERIOD,
        exist_ok = True,
        patience=PATIENCE,
        single_cls = True,
        dropout=DROPOUT,
        optimizer='Adam'
    )


if __name__ == "__main__":
    split_dataset()
    create_data_yaml()
    train()