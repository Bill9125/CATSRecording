import json
import os
import re
import numpy as np
import argparse

def calculate_joint_angle(A, B, C):
    """計算三個關節形成的角度"""
    A, B, C = np.array(A), np.array(B), np.array(C)

    # 計算向量
    BA = A - B  # 中點 → 第一點 向量
    BC = C - B  # 中點 → 第三點 向量

    # 計算內積與長度
    dot_product = np.dot(BA, BC)
    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)

    if norm_BA == 0 or norm_BC == 0:
        return np.nan  # 避免除以 0

    cosine_angle = dot_product / (norm_BA * norm_BC)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # 避免浮點誤差
    angle = np.degrees(np.arccos(cosine_angle))  # 轉換為角度

    return angle

def read_skeleton_data(file_path):
    """讀取骨架數據，解析 Frame 和關鍵點"""
    frames = []
    angles = {"right_armpit": [], "left_armpit": [], "right_elbow": [], "left_elbow": []}

    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r'Frame (\d+):\s*\[\[(.*?)\]\]', line)
            if not match:
                continue  # 跳過格式錯誤的行

            frame = int(match.group(1))
            points_str = match.group(2)

            try:
                # 解析座標資料
                points_str = points_str.replace("(", "[").replace(")", "]")
                points = eval(f"[{points_str}]")  # ⚠️ 建議改用 json.loads()
                
                if len(points) < 6:
                    continue  # 跳過無效數據

                frames.append(frame)
                points = np.array(points)

                # **確保骨架關鍵點正確**
                right_shoulder, left_shoulder = points[0], points[1]
                right_elbow, left_elbow = points[2], points[3]
                right_wrist, left_wrist = points[4], points[5]

                # ✅ 計算腋下角度（肩 → 肘 → 手腕）
                right_armpit_angle = 180 - calculate_joint_angle(right_shoulder, right_elbow, right_wrist)
                left_armpit_angle = 180 - calculate_joint_angle(left_shoulder, left_elbow, left_wrist)

                # ✅ 計算手肘角度（手腕 → 肘 → 肩膀）
                right_elbow_angle = 180 - calculate_joint_angle(right_wrist, right_elbow, right_shoulder)
                left_elbow_angle = 180 - calculate_joint_angle(left_wrist, left_elbow, left_shoulder)

                # ✅ 確保數據完整，避免 NaN
                angles["right_armpit"].append(right_armpit_angle if not np.isnan(right_armpit_angle) else 0)
                angles["left_armpit"].append(left_armpit_angle if not np.isnan(left_armpit_angle) else 0)
                angles["right_elbow"].append(right_elbow_angle if not np.isnan(right_elbow_angle) else 0)
                angles["left_elbow"].append(left_elbow_angle if not np.isnan(left_elbow_angle) else 0)

            except Exception as e:
                print(f"❌ Skipping frame {frame} due to parsing error: {e}")
                continue  # 繼續處理下一個 Frame

    return frames, angles

def save_json(title, y_label, frames, values, output_path):
    """將計算結果儲存成 JSON"""
    if len(values) > 200:
        trimmed_data = values[100:-100]
    else:
        trimmed_data = values

    y_min = min(trimmed_data) * 0.9 if trimmed_data else 0
    y_max = max(trimmed_data) * 1.1 if trimmed_data else 180

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
    armpit_values = list(zip(angles["right_armpit"], angles["left_armpit"]))
    armpit_output_path = os.path.join(output_dir, 'Armpit_Angle.json')
    save_json("Armpit Joint Angles", "Angle (degrees)", frames, armpit_values, armpit_output_path)

    # ✅ 儲存 Shoulder_Angle.json（手肘角度）
    shoulder_values = list(zip(angles["right_elbow"], angles["left_elbow"]))
    shoulder_output_path = os.path.join(output_dir, 'Shoulder_Angle.json')
    save_json("Elbow Joint Angles", "Angle (degrees)", frames, shoulder_values, shoulder_output_path)

# ✅ CLI 參數處理
parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--out', type=str)
args = parser.parse_args()
dir = args.dir
out = args.out
data_path = os.path.join(dir, "yolo_skeleton.txt")
output_dir = os.path.join(out, 'Benchpress_data')

os.makedirs(output_dir, exist_ok=True)
process_skeleton_data(data_path, output_dir)
