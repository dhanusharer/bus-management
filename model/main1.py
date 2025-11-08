import cv2
import cvzone
import pandas as pd
import datetime
import time
from ultralytics import YOLO
import numpy as np

# ----------------------------
# Tracker with disappearance handling
# ----------------------------
class Tracker:
    def __init__(self, max_distance=50, disappear_frames=30):
        self.tracked_objects = {}  # obj_id -> (cx, cy, disappear_count)
        self.max_distance = float(max_distance)
        self.next_object_id = 1
        self.disappear_frames = int(disappear_frames)

    def update(self, rects):
        updated_objects = {}
        centers = []
        for rect in rects:
            x1, y1, x2, y2 = rect
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centers.append((cx, cy, rect))

        used_idxs = set()
        for obj_id, (obj_cx, obj_cy, disappear_count) in list(self.tracked_objects.items()):
            min_dist = self.max_distance + 1
            match_idx = None
            for i, (cx, cy, _) in enumerate(centers):
                if i in used_idxs:
                    continue
                dist = float(np.hypot(cx - obj_cx, cy - obj_cy))
                if dist < min_dist:
                    min_dist = dist
                    match_idx = i

            if min_dist <= self.max_distance and match_idx is not None:
                cx, cy, rect = centers[match_idx]
                updated_objects[obj_id] = (cx, cy, 0, rect)
                used_idxs.add(match_idx)
            else:
                if disappear_count + 1 <= self.disappear_frames:
                    updated_objects[obj_id] = (obj_cx, obj_cy, disappear_count + 1, None)

        # Register new objects for unmatched detections
        for i, (cx, cy, rect) in enumerate(centers):
            if i in used_idxs:
                continue
            updated_objects[self.next_object_id] = (cx, cy, 0, rect)
            self.next_object_id += 1

        # Commit and output only visible objects
        self.tracked_objects = {}
        final_objects = {}
        for obj_id, (cx, cy, disappear_count, rect) in updated_objects.items():
            if disappear_count <= self.disappear_frames and rect is not None:
                self.tracked_objects[obj_id] = (cx, cy, disappear_count)
                final_objects[obj_id] = rect
            elif disappear_count < self.disappear_frames and rect is None:
                self.tracked_objects[obj_id] = (cx, cy, disappear_count)

        return final_objects

# ----------------------------
# Model and capture
# ----------------------------
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("http://10.220.241.62:8080/video")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

class_list = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet",
    "tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave",
    "oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]

tracker = Tracker(max_distance=50, disappear_frames=30)

# ROI in 960x540 space
roi = (60, 20, 900, 520)

# CSV setup
csv_file = "busstop_counts.csv"
if not pd.io.common.file_exists(csv_file):
    pd.DataFrame(columns=["Timestamp", "WaitingCount"]).to_csv(csv_file, index=False)

log_interval = 10  # seconds
last_log_time = time.time()

# Fullscreen window
cv2.namedWindow("Bus Stop People Counter", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Bus Stop People Counter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("⚠ Cannot read frame (stream ended or unreachable).")
        break

    frame = cv2.resize(frame, (960, 540))

    # ROI rectangle (yellow)
    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 255), 2)

    detections = []
    for r in model(frame, stream=True, verbose=False):
        for box in r.boxes:
            cls = int(box.cls[0])
            if 0 <= cls < len(class_list) and class_list[cls] == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]:
                    detections.append([x1, y1, x2, y2])

    tracked = tracker.update(detections)

    current_ids = set()
    for obj_id, rect in tracked.items():
        x1, y1, x2, y2 = rect
        # Optionally, change this color to green too: (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cvzone.putTextRect(frame, f'ID {obj_id}', (x1, max(0, y1 - 10)), 1, 1, colorR=(0, 255, 0))  # Green label background
        current_ids.add(obj_id)

    waiting_count = len(current_ids)

    # Periodic CSV logging
    now_ts = time.time()
    if now_ts - last_log_time >= log_interval:
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pd.DataFrame([[now_str, waiting_count]], columns=["Timestamp", "WaitingCount"]).to_csv(
            csv_file, mode="a", header=False, index=False)
        last_log_time = now_ts

    cvzone.putTextRect(frame, f'Waiting People: {waiting_count}', (50, 50), 2, 2, colorR=(0, 0, 0))
    cv2.imshow("Bus Stop People Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
