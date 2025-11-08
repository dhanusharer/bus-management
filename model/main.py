import cv2
from ultralytics import YOLO
import numpy as np
import cvzone
from ObjectMonitor import ObjectMonitor
import csv
from datetime import datetime
import os

# COCO object categories
categories = [
    "person", "bicycle", "car", "motorbike", "aeroplane","bus","train","truck",
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

# Load YOLO model
model = YOLO("yolov8s.pt")

roi = (60, 20, 900, 520)  # ROI box

video_sources = [
    "../videos/busstop_1.MOV",
    "../videos/busstop_2.MOV",
    "../videos/busstop_3.MOV",
    "../videos/busstop_4.mp4",
    "../videos/busstop_5.mp4",
    "../videos/busstop_6.mp4",
    "../videos/busstop_7.mp4",
    "../videos/busstop_8.mp4",
    "../videos/busstop_9.mp4"
]

N = len(video_sources)
captures = [cv2.VideoCapture(src) for src in video_sources]
trackers = [ObjectMonitor(max_dist=50, max_lost_frames=30) for _ in video_sources]

frame_width, frame_height = 480, 270
log_interval_sec = 10
last_logged_time = datetime.now()

csv_file = "../backend_database/bmtc_cctv_counts.csv"

# Create CSV header if file does not exist
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["Timestamp"]
        for i in range(N):
            header.extend([f"Video{i+1}", f"Count{i+1}"])
        header.append("Total")
        writer.writerow(header)

final_counts = [0] * N
total_final = 0

while True:
    frames = []
    all_ended = True
    current_counts = [0] * N

    for i, cap in enumerate(captures):
        ret, frame = cap.read()
        if not ret or frame is None:
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        else:
            all_ended = False
            frame = cv2.resize(frame, (frame_width, frame_height))

            cv2.rectangle(frame,
                          (roi[0]*frame_width//960, roi[1]*frame_height//540),
                          (roi[2]*frame_width//960, roi[3]*frame_height//540),
                          (0, 255, 255), 2)

            results = model(frame, stream=True, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if 0 <= cls_id < len(categories) and categories[cls_id] == "person":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        if (roi[0]*frame_width//960) <= cx <= (roi[2]*frame_width//960) and \
                           (roi[1]*frame_height//540) <= cy <= (roi[3]*frame_height//540):
                            detections.append([x1, y1, x2, y2])

            tracked_objects = trackers[i].track(detections)
            current_ids = set()
            for obj_id, box in tracked_objects.items():
                x1, y1, x2, y2 = box
                # Bounding box in green
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Label box background in green
                cvzone.putTextRect(frame, f"ID {obj_id}", (x1, y1 - 10), scale=1, thickness=1, colorR=(0, 255, 0))
                current_ids.add(obj_id)

            current_counts[i] = len(current_ids)
            cvzone.putTextRect(frame, f"Waiting: {current_counts[i]}", (20, 30),
                              scale=1.1, thickness=2, colorR=(0, 0, 0))

        frames.append(frame)

    total_now = sum(current_counts)
    now = datetime.now()

    if (now - last_logged_time).total_seconds() >= log_interval_sec:
        last_logged_time = now
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [now.strftime("%Y-%m-%d %H:%M:%S")]
            for i in range(N):
                filename_only = os.path.splitext(os.path.basename(video_sources[i]))[0]
                row.extend([filename_only, current_counts[i]])
            row.append(total_now)
            writer.writerow(row)

    final_counts = current_counts
    total_final = total_now

    if all_ended:
        print("All videos ended.")
        break

    grid_cols, grid_rows = 3, 3
    while len(frames) < grid_cols * grid_rows:
        frames.append(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))

    rows_frames = []
    for r in range(grid_rows):
        row_frames = frames[r*grid_cols:(r+1)*grid_cols]
        rows_frames.append(np.hstack(row_frames))
    combined = np.vstack(rows_frames)

    cv2.imshow("Bus Stop 3x3 CCTV Grid", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

for cap in captures:
    cap.release()
cv2.destroyAllWindows()

print("Final waiting counts:")
for i, src in enumerate(video_sources):
    filename_only = os.path.splitext(os.path.basename(src))[0]
    print(f"{filename_only}: {final_counts[i]}")
print(f"Total waiting passengers (last count): {total_final}")
