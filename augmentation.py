# Configuration - set these paths and parameters before running

CARDS_DIR       = "Cards"
BG_DIR          = "Backgrounds"
OUT_DIR         = "AugCards"
NUM_AUG         = 10000      # Total augmented images to generate


# Single class for YOLOv8 detection
CLASS_ID   = 0
CLASS_NAME = "card"

import os
import cv2
import random
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox
from glob import glob


def img_transform(img, target_size):
    # card is bigger than back in this amount
    to_scale = target_size[0] / img.shape[0]
    bb = BoundingBox(x1=0,y1=0, x2=img.shape[1], y2=img.shape[0])
    # Augmentation pipeline:
    # 1) Crop random amount of pixels from each side
    # 2) Random scale and rotate the image
    # 3) Apply perspective transformation in 50% cases
    aug_pipe = iaa.Sequential([
        iaa.Crop(percent=((0,0.05),(0,0.05),(0,0.05),(0,0.05)), keep_size=False),
        iaa.SomeOf((1,5),
                   [iaa.Add((-40,40)),
                    iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)),
                    iaa.Multiply((0.5, 1.5)),
                    iaa.GaussianBlur(sigma=(0.0, 3.0)),
                    iaa.GammaContrast((0.5, 2.0)),
                    iaa.GammaContrast((0.5, 2.0)),
                    iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.5))
                    ], random_order=True),
        iaa.Affine(scale=(to_scale*0.75, to_scale*0.1), rotate=(-90, 90)),
        iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0, 0.1), fit_output=True)),
        iaa.CropToFixedSize(width=target_size[1], height=target_size[0], position='center'),
        iaa.PadToFixedSize(width=target_size[1], height=target_size[0]),
        iaa.TranslateX(percent=(-0.3, 0.3)),
        iaa.TranslateY(percent=(-0.3, 0.3))
    ])
    return aug_pipe(images=[img], bounding_boxes=[bb])

def main():
    card_paths = glob(os.path.join(CARDS_DIR, "*"))
    back_paths = glob(os.path.join(BG_DIR, "*"))
    img_out = os.path.join(OUT_DIR, "images")
    lbl_out = os.path.join(OUT_DIR, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for i in range(NUM_AUG):
        # Choose random image
        card_path = random.choice(card_paths)
        img = cv2.imread(card_path, cv2.IMREAD_UNCHANGED)
        # Choose random background
        back_path = random.choice(back_paths)
        back = cv2.imread(back_path)

        # Get transformated image
        aug_res = img_transform(img, back.shape)
        aug_img, aug_bb = aug_res[0][0], aug_res[1][0]

        # Add background to image
        aug_img[aug_img[:,:,3] == 0] = 0
        aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGRA2BGR)
        aug_img[aug_img.sum(axis=2) < 75] = back[aug_img.sum(axis=2) < 75]

        # Compute bbox
        x1 = min(max(aug_bb[0][0], 0), back.shape[1])
        x2 = min(max(aug_bb[1][0], 0), back.shape[1])
        y1 = min(max(aug_bb[0][1], 0), back.shape[0])
        y2 = min(max(aug_bb[1][1], 0), back.shape[0])
        xc = ((x1+x2) / 2.0) / back.shape[1]
        yc = ((y1+y2) / 2.0) / back.shape[0]
        wn = abs(x1-x2) / back.shape[1]
        hn = abs(y1-y2) / back.shape[0]

        # Save image
        img_fn = f"{card_path[6:10]}_{i:06d}.jpg"
        cv2.imwrite(os.path.join(img_out, img_fn), aug_img)

        # Save label
        lbl_fn = f"{card_path[6:10]}_{i:06d}.txt"
        with open(os.path.join(lbl_out, lbl_fn), "w") as f:
            f.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        if (i+1) % 100 == 0:
            print(f'Generated {i+1}/{NUM_AUG}')



if __name__ == "__main__":
    main()
