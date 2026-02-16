import numpy as np
from ultralytics import YOLO
import cv2
import rick_roll

model = YOLO("runs/detect/train3/weights/best.pt")

cap = cv2.VideoCapture(0)

missed = 0
MISSED_LIMIT = 60

rickroll = False
rickroll_can_be_done = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    detected = False
    all_boxes = []
    all_confs = []

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        confs = r.boxes.conf.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()

        all_boxes.extend(boxes)
        all_confs.extend(confs)

    if len(all_confs) > 0:
        avg_conf = np.mean(all_confs)
        if avg_conf >= 0.75:
            detected = True

    if detected:
        missed = 0
        rickroll_can_be_done = True
        rickroll = False
    else:
        missed += 1
        if missed >= MISSED_LIMIT and rickroll_can_be_done:
            rickroll = True
            rickroll_can_be_done = False

    if rickroll:
        rick_roll.rickroll()
        missed = 0
        rickroll = False

    h, w = frame.shape[:2]

    i = 0
    counter = 1

    for (x1, y1, x2, y2), conf in zip(all_boxes, all_confs):
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Conf {counter}: {conf:.2f}",
            (10, h - 40 - i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        i += 30
        counter += 1

    cv2.imshow("YOLO eyes", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

