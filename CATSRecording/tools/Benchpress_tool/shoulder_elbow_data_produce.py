import json
import os
import re
import numpy as np
import argparse

def read_skeleton_data(file_path):
    frames = []
    angles = {"right_shoulder": [], "left_shoulder": [], "right_elbow": [], "left_elbow": []}

    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r'Frame (\d+):\s*\[\[(.*?)\]\]', line)
            if not match:
                print(f"❌ Format mismatch: {line.strip()}")
                continue

            frame = int(match.group(1))
            points_str = match.group(2)

            try:
                # 轉換 `()` 為 `[]`，以便 `eval()` 解析
                points_str = points_str.replace("(", "[").replace(")", "]")
                points = eval(f"[{points_str}]")  # 直接解析為 list

                if not isinstance(points, list) or len(points) < 6:
                    print(f"⚠️ Frame {frame} has {len(points)} points, proceeding with calculation.")
                
                frames.append(frame)
                points = np.array(points)

                # 取出骨架關鍵點
                s0, s1, e0, e1, w0, w1 = points[:6]


                ###計算每個向量之間的小角度
                def calculate_angle(v1, v2):
                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)
                    
                    if norm_v1 == 0 or norm_v2 == 0:
                        return None  # 避免除零錯誤
                    
                    # 確保夾角範圍在 0°~180°
                    angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)))
                    
                    return angle  # 確保角度只在 0°~180°


                # 計算角度
                right_shoulder_angle = 180 - calculate_angle(s1 - s0, e1 - s1)
                print("right_shoulder_angle" ,right_shoulder_angle)
                left_shoulder_angle = 180 - calculate_angle(s0 - s1, e0 - s0)
                print("left_shoulder_angle" ,left_shoulder_angle)
                right_elbow_angle = 180 - calculate_angle(e1 - s1, w1 - e1)
                print("right_elbow_angle" ,right_elbow_angle)
                left_elbow_angle = 180 - calculate_angle(e0 - s0, w0 - e0)
                print("left_elbow_angle" ,left_elbow_angle)

                # **確保每一個 Frame 都存入值，即使是 None**
                angles["right_shoulder"].append(right_shoulder_angle if right_shoulder_angle is not None else 0)
                angles["left_shoulder"].append(left_shoulder_angle if left_shoulder_angle is not None else 0)
                angles["right_elbow"].append(right_elbow_angle if right_elbow_angle is not None else 0)
                angles["left_elbow"].append(left_elbow_angle if left_elbow_angle is not None else 0)

            except Exception as e:
                print(f"❌ Skipping frame {frame} due to parsing error: {e}")
                # **如果有解析錯誤，確保也存入 0 來保持長度一致**
                angles["right_shoulder"].append(None)
                angles["left_shoulder"].append(None)
                angles["right_elbow"].append(None)
                angles["left_elbow"].append(None)

    print(f"✅ Total frames processed: {len(frames)}")
    return frames, angles


def save_json(title, y_label, frames, values, output_path):
    """將計算結果儲存成 JSON"""
    if len(values) > 200:
        trimmed_data = values[100:-100]
    else:
        trimmed_data = values

    # 過濾掉 None 並確保是 float
    valid_data = [float(v) for v in trimmed_data if v is not None and isinstance(v, (int, float))]

    # 確保 valid_data 不是空的，否則設置 y_min=0, y_max=180
    y_min = min(valid_data) * 0.9 if valid_data else 0
    y_max = max(valid_data) * 1.1 if valid_data else 180

    data = {
        "title": title,
        "y_label": y_label,
        "y_min": round(y_min, 2),
        "y_max": round(y_max, 2),
        "frames": frames,
        "values": values
    }

    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"✅ {title} 已儲存到 {output_path}")

def process_skeleton_data(file_path, output_dir):
    """計算關節角度，並存成 JSON 檔案"""
    frames, angles = read_skeleton_data(file_path)

    # ✅ 儲存 Armpit_Angle.json（腋下角度）
    shoulder_values = list(zip(angles["right_shoulder"], angles["left_shoulder"]))
    shoulder_output_path = os.path.join(output_dir, 'Shoulder_Angle.json')
    save_json("Shoulder Joint Angles", "Angle (degrees)", frames, shoulder_values, shoulder_output_path)

    # ✅ 儲存 Shoulder_Angle.json（手肘角度）
    elbow_values = list(zip(angles["right_elbow"], angles["left_elbow"]))
    elbow_output_path = os.path.join(output_dir, 'Elbow_Angle.json')
    save_json("Elbow Joint Angles", "Angle (degrees)", frames, elbow_values, elbow_output_path)

# ✅ CLI 參數處理
parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--out', type=str)
args = parser.parse_args()
dir = args.dir
out = args.out
data_path = os.path.join(dir, "yolo_skeleton_interpolated.txt")
output_dir = os.path.join(out, 'Benchpress_data')

os.makedirs(output_dir, exist_ok=True)
process_skeleton_data(data_path, output_dir)
