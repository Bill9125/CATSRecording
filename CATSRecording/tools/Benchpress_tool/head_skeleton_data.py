import json
import os
import re
import numpy as np
import argparse

def calculate_joint_angle(A, B, C):
    """
    計算關節夾角（使用餘弦定理）
    A: (x, y) - 第一點（肩膀）
    B: (x, y) - 中間點（手肘）
    C: (x, y) - 第三點（手腕）
    """
    A, B, C = np.array(A), np.array(B), np.array(C)

    # 計算向量
    BA = A - B  # 手肘 -> 肩膀 向量
    BC = C - B  # 手肘 -> 手腕 向量

    # 計算內積與長度
    dot_product = np.dot(BA, BC)
    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)

    # 避免 0 長度向量
    if norm_BA == 0 or norm_BC == 0:
        return np.nan  # 回傳 NaN，讓後續使用上一個有效值

    # 計算夾角
    cosine_angle = dot_product / (norm_BA * norm_BC)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # 避免浮點誤差
    angle = np.degrees(np.arccos(cosine_angle))  # 轉換為角度

    return angle

def read_skeleton_data(txt_file_path):
    """
    讀取 txt 檔案並解析骨架點
    :param txt_file_path: txt 檔案路徑
    :return: 解析後的骨架數據 list
    """
    skeleton_data = []
    frame_pattern = re.compile(r"Frame (\d+): \[\[(.*?)\]\]")  # 提取 frame 和骨架點
    point_pattern = re.compile(r"\(([\d.]+), ([\d.]+)\)")  # 提取 (x, y) 座標

    with open(txt_file_path, "r") as file:
        for line in file:
            match = frame_pattern.match(line.strip())
            if match:
                frame_idx = int(match.group(1))
                points_str = match.group(2)

                keypoints = []
                for point_match in point_pattern.findall(points_str):
                    x, y = map(float, point_match)
                    keypoints.append((x, y))

                skeleton_data.append((frame_idx, keypoints))

    return skeleton_data

def process_skeleton_data(txt_file_path, output_json_path):
    """
    讀取骨架數據，計算腋下角度，並儲存 JSON
    :param txt_file_path: txt 檔案路徑
    :param output_json_path: 輸出 JSON 檔案路徑
    """
    skeleton_data = read_skeleton_data(txt_file_path)
    frames = []
    left_angles = []
    right_angles = []

    # 初始設定，讓第一幀的 NaN 值變成 0
    prev_left_angle = 0
    prev_right_angle = 0

    for frame_idx, keypoints in skeleton_data:
        if len(keypoints) >= 6:
            right_shoulder = keypoints[0]  # 右肩膀
            right_elbow = keypoints[1]  # 右手肘
            right_wrist = keypoints[3]  # 右手腕
            left_shoulder = keypoints[1]  # 左肩膀
            left_elbow = keypoints[0]  # 左手肘
            left_wrist = keypoints[2]  # 左手腕

            left_angle = calculate_joint_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_joint_angle(right_shoulder, right_elbow, right_wrist)

            # ✅ 如果 NaN，使用上一幀的值
            if np.isnan(left_angle):
                left_angle = prev_left_angle
            else:
                prev_left_angle = left_angle  # 記錄當前值

            if np.isnan(right_angle):
                right_angle = prev_right_angle
            else:
                prev_right_angle = right_angle  # 記錄當前值

            frames.append(frame_idx)
            left_angles.append(left_angle)
            right_angles.append(right_angle)

            print(f"Frame {frame_idx} - 左腋下角度: {left_angle:.2f}°, 右腋下角度: {right_angle:.2f}°")

    # ✅ 計算 `y_min` 和 `y_max`（排除前 100 幀）
    if len(left_angles) > 200:
        trimmed_data = left_angles[100:-100] + right_angles[100:-100]
    else:
        trimmed_data = left_angles + right_angles

    y_min = min(trimmed_data) * 0.9 if trimmed_data else 0
    y_max = max(trimmed_data) * 1.1 if trimmed_data else 180

    # ✅ 生成 JSON 格式
    data = {
        "title": "Underarm Joint Angle",
        "y_label": "Angle (degrees)",
        "y_min": round(y_min, 2),
        "y_max": round(y_max, 2),
        "frames": frames,
        "values": list(zip(left_angles, right_angles))  # (左腋下角度, 右腋下角度)
    }

    # ✅ 儲存 JSON 檔案
    with open(output_json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"✅ 腋下角度數據已儲存到 {output_json_path}")

# ✅ CLI 參數處理
parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--out', type=str)
args = parser.parse_args()
dir = args.dir
out = args.out
data_path = os.path.join(dir, "yolo_skeleton.txt")
process_skeleton_data(data_path, out)
