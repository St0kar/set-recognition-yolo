import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

cardH, cardW = (715,466)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.hypot(br[0] - bl[0], br[1] - bl[1])
    widthB = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    maxWidth = int(max(widthA, widthB))

    heightA = np.hypot(tr[0] - br[0], tr[1] - br[1])
    heightB = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# ---- Paths ----
input_folder = 'CardsRaw'     # Folder with your 81 images
output_folder = 'Cards'    # Folder where the extracted cards will be saved

os.makedirs(output_folder, exist_ok=True)
show = True
# ---- Process each image ----
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f'{filename.split('.')[0] + ".png"}')

        # Load image
        img = cv2.imread(input_path)

        if img is None:
            print(f"Could not load {filename}")
            continue

        try:
            # Preprocessing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blur, 50, 150)

            # Find contours
            cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(cnts) == 0:
                print(f"No contours found in {filename}")
                continue

            # Find largest contour (assume it is the card)
            card = max(cnts, key=cv2.contourArea)

            # Get rectangle box
            rect = cv2.minAreaRect(card)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="float32")

            # Apply perspective transform
            straight = four_point_transform(img, box)

            # Rotate to make them oriented the same way
            straight = cv2.rotate(straight, cv2.ROTATE_90_CLOCKWISE) if straight.shape[1] > straight.shape[0] else straight

            # Resize to make uniform size
            resized = cv2.resize(straight, (cardW, cardH))
            # Add alpha layer
            imgwarp = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)

            # Create an alpha mask with a shape of the card
            radius = 40
            margin = 4
            alphamask = np.zeros((cardH, cardW), dtype=np.uint8)
            cv2.rectangle(alphamask, (margin, radius), (cardW - margin, cardH - radius), 255, -1)
            cv2.rectangle(alphamask, (radius, margin), (cardW - radius, cardH - margin), 255, -1)
            cv2.circle(alphamask, (radius + margin, radius + margin), radius, 255, -1)
            cv2.circle(alphamask, (cardW - radius - margin, radius + margin), radius, 255, -1)
            cv2.circle(alphamask, (radius + margin, cardH - radius - margin), radius, 255, -1)
            cv2.circle(alphamask, (cardW - radius - margin, cardH - radius - margin), radius, 255, -1)

            # Add the alphachannel to the warped image
            imgwarp[:, :, 3] = alphamask

            # Save output
            cv2.imwrite(output_path, imgwarp)
            print(f"Saved {output_path}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")
