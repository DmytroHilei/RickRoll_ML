import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

image_dir = Path("../dataset/images/train")
label_dir = Path("../dataset/labels/train")

label_dir.mkdir(parents=True, exist_ok=True)

class_id = 0

# Mediapipe
model_path = r"C:\Users\giley\PycharmProjects\YOLO training\face_landmarker.task"   # Download from MediaPipe repo

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# MediaPipe eye landmark indices
LEFT_EYE = [
    33, 133, 160, 159, 158, 157, 173,
    246, 161, 163, 144, 145, 153, 154, 155
]

RIGHT_EYE = [
    362, 263, 387, 386, 385, 384, 398,
    466, 388, 390, 373, 374, 380, 381, 382
]


def bbox_from_landmarks(landmarks, indices, w, h):
    xs = [int(landmarks[i].x*w) for i in indices]
    ys = [int(landmarks[i].y*h) for i in indices]

    x_min = max(0, min(xs))
    x_max = min(w, max(xs))
    y_min = max(0, min(ys))
    y_max = min(h, max(ys))

    pad_x = int(0.2 * (x_max - x_min))
    pad_y = int(0.3 * (y_max - y_min))

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x)
    y_max = min(h, y_max + pad_y)

    return x_min, y_min, x_max, y_max

for img_path in image_dir.glob("*.png"):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue
    h, w = img_bgr.shape[:2]

    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    result = detector.detect(mp_image)
    if not result.face_landmarks:
        continue
    landmarks = result.face_landmarks[0]

    boxes = []

    boxes.append(bbox_from_landmarks(landmarks, LEFT_EYE, w, h))
    boxes.append(bbox_from_landmarks(landmarks, RIGHT_EYE, w, h))

    label_lines = []

    for (x_min, y_min, x_max, y_max) in boxes:
        box_w = x_max - x_min
        box_h = y_max - y_min

        if box_w <= 0 or box_h <= 0:
            continue

        x_center = ((x_min + x_max) / 2)/w
        y_center = ((y_min + y_max) / 2)/h

        bw = box_w/w
        bh = box_h/h

        label_lines.append(
            f"{class_id} {x_center} {y_center} {bw} {bh}"
        )
        if not label_lines:
            continue

        label_path = label_dir / (img_path.stem + ".txt")

        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

print("Done\n")

