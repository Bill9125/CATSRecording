import json, os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--out', type=str)
args = parser.parse_args()
dir = args.dir
out = args.out
data_path = os.path.join(dir, "mediapipe_landmarks.txt")
outpu_path = os.path.join(out, 'Benchpress_data', 'Armpit_Angle.json')

# ✅ 計算關節角度
def calculate_joint_angle(p1, p2, p3):
    """計算關節角度，p1-p2-p3 為三個點"""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    # 計算夾角
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# 初始化數據存儲
frames = []
values = []

# 解析 YOLO txt 檔案
landmarks = {}  # { frame: { landmark_id: (x, y) } }

with open(data_path, "r") as file:
    for line in file:
        parts = line.strip().split(",")
        if len(parts) < 4:
            continue
        
        frame = int(parts[0])
        landmark_id = int(parts[1])
        x, y = float(parts[2]), float(parts[3])

        if frame not in landmarks:
            landmarks[frame] = {}
        landmarks[frame][landmark_id] = (x, y)

# ✅ 計算每一幀的角度
for frame, points in landmarks.items():
    if all(k in points for k in [11, 12, 13, 14, 23, 24]):  # 確保數據完整
        left_angle = calculate_joint_angle(points[13], points[11], points[23])  # 13-11-23
        right_angle = calculate_joint_angle(points[14], points[12], points[24])  # 14-12-24

        frames.append(frame)
        values.append((left_angle, right_angle))

# ✅ 計算 y_min 和 y_max
if values:
    y_values = [angle for pair in values for angle in pair]  # 提取所有角度
    y_min = min(y_values) * 0.9  # 讓範圍多 10%
    y_max = max(y_values) * 1.1
else:
    y_min = y_max = 0

# ✅ 轉換成 JSON 格式
data = {
    "title": "Armpit Angles",
    "y_label": "Angle (degrees)",
    "y_min": y_min,
    "y_max": y_max,
    "frames": frames,
    "values": values  # (左肩角度, 右肩角度)
}

# ✅ 存成 JSON 檔案
with open(outpu_path, "w") as json_file:
    json.dump(data, json_file, indent=4)

print(f"✅ JSON 檔案已儲存: {outpu_path}")
