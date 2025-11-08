import numpy as np

class ObjectMonitor:
    def __init__(self, max_dist=50, max_lost_frames=30):
        self.max_dist = max_dist
        self.max_lost_frames = max_lost_frames
        self.next_object_id = 0
        self.objects = {}  # object_id -> (centroid, bbox)
        self.lost_frames = {}  # object_id -> lost count

    def _distance(self, c1, c2):
        return np.linalg.norm(np.array(c1) - np.array(c2))

    def track(self, detections):
        """
        detections: list of bounding boxes [x1, y1, x2, y2]
        Returns dict of object_id: bbox
        """
        input_centroids = []
        for det in detections:
            x1, y1, x2, y2 = det
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.objects[self.next_object_id] = (centroid, detections[i])
                self.lost_frames[self.next_object_id] = 0
                self.next_object_id += 1
            return {obj_id: bbox for obj_id, (centroid, bbox) in self.objects.items()}

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[obj_id][0] for obj_id in object_ids]

        D = np.zeros((len(object_centroids), len(input_centroids)))
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = self._distance(oc, ic)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows = set()
        assigned_cols = set()
        new_objects = {}

        for row, col in zip(rows, cols):
            if row in assigned_rows or col in assigned_cols:
                continue
            if D[row, col] > self.max_dist:
                continue

            obj_id = object_ids[row]
            centroid = input_centroids[col]
            bbox = detections[col]
            new_objects[obj_id] = (centroid, bbox)
            assigned_rows.add(row)
            assigned_cols.add(col)

        for i, obj_id in enumerate(object_ids):
            if i not in assigned_rows:
                self.lost_frames[obj_id] += 1
            else:
                self.lost_frames[obj_id] = 0

        for obj_id in list(self.lost_frames.keys()):
            if self.lost_frames[obj_id] > self.max_lost_frames:
                self.lost_frames.pop(obj_id)
                self.objects.pop(obj_id, None)

        for i, centroid in enumerate(input_centroids):
            if i not in assigned_cols:
                self.objects[self.next_object_id] = (centroid, detections[i])
                self.lost_frames[self.next_object_id] = 0
                self.next_object_id += 1
            else:
                obj_id = [object_ids[r] for r, c in zip(rows, cols) if c == i and r in assigned_rows][0]
                self.objects[obj_id] = new_objects[obj_id]

        for obj_id in new_objects:
            self.objects[obj_id] = new_objects[obj_id]

        return {obj_id: bbox for obj_id, (centroid, bbox) in self.objects.items()}
