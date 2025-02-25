import json, os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--out', type=str)
args = parser.parse_args()
dir = args.dir
out = args.out
# 讀取 yolo 檔案
yolo_txt_path = os.path.join(dir, "yolo_coordinates.txt")  # 你的 txt 檔案路徑
output_json_path = os.path.join(out, "Bar_Position.json")  # 輸出的 JSON 檔案

# 初始化數據存儲
frames = []
values = []

# 讀取 YOLO 偵測數據
with open(yolo_txt_path, "r") as file:
    for line in file:
        parts = line.strip().split(",")
        if len(parts) < 3:  # 確保資料完整
            continue
        
        frame_count = int(parts[0])  # 幀數
        x_center = float(parts[1])  # X 中心
        y_center = float(parts[2])  # Y 中心
        
        frames.append(frame_count)
        values.append((x_center, y_center))  # 存成 (x, y) 格式

# 計算 x_min, x_max, y_min, y_max
if values:
    x_values = [val[0] for val in values]  # 取所有 x_center
    y_values = [val[1] for val in values]  # 取所有 y_center

    x_min = min(x_values) * 0.9  # X 軸最小值，留 10% 緩衝
    x_max = max(x_values) * 1.1  # X 軸最大值，留 10% 緩衝
    y_min = min(y_values) * 0.9  # Y 軸最小值，留 10% 緩衝
    y_max = max(y_values) * 1.1  # Y 軸最大值，留 10% 緩衝
else:
    x_min = x_max = y_min = y_max = 0

# 轉換成 JSON 格式
data = {
    "title": "Barbell Center Positions",
    "y_label": "Position (pixels)",
    "x_min": x_min,
    "x_max": x_max,
    "y_min": y_min,
    "y_max": y_max,
    "frames": frames,
    "values": values  # (x_center, y_center)
}


# 轉換成 JSON 格式
data = {
    "title": "Barbell Center Positions",
    "y_label": "Position (pixels)",
    "y_min": y_min,
    "y_max": y_max,
    "frames": frames,
    "values": values  # (x_center, y_center)
}

# 存成 JSON 檔案
with open(output_json_path, "w") as json_file:
    json.dump(data, json_file, indent=4)

print(f"✅ JSON 檔案已儲存: {output_json_path}")
