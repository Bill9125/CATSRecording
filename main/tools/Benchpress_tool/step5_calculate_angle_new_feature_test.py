import os
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd

def parse_yolo_file(file_path):
    frame_pattern = re.compile(r"Frame (\d+): \[\[(.*?)\]\]")
    frames = []
    coords = []
    with open(file_path, 'r') as f:
        for line in f:
            match = frame_pattern.match(line.strip())
            if match:
                frame = int(match.group(1))
                point_str = match.group(2).replace("(", "[").replace(")", "]")
                try:
                    points = np.array(eval(f"[{point_str}]"))
                    if points.shape[0] >= 6:
                        frames.append(frame)
                        coords.append(points)
                except Exception:
                    continue
    return frames, coords

def calc_angle(a, b, c):
    v1 = a - b
    v2 = c - b
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return None
    cosine = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def perpendicular_distance(point, line_start, line_end):
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return None
    proj_len = np.dot(point_vec, line_vec) / line_len
    proj_point = line_start + (proj_len / line_len) * line_vec
    return np.linalg.norm(point - proj_point)

def process_wrist_to_shoulder_line(coords_list):
    right_distances = []
    left_distances = []
    for pts in coords_list:
        if len(pts) >= 6:
            right_wrist = pts[4]
            left_wrist = pts[5]
            right_shoulder = pts[0]
            left_shoulder = pts[1]
            line_start = np.array(right_shoulder)
            line_end = np.array(left_shoulder)
            right_distances.append(perpendicular_distance(np.array(right_wrist), line_start, line_end))
            left_distances.append(perpendicular_distance(np.array(left_wrist), line_start, line_end))
        else:
            right_distances.append(None)
            left_distances.append(None)
    return right_distances, left_distances

def save_feature(file_path, frames, values):
    with open(file_path, 'w') as f:
        for frame, val in zip(frames, values):
            if val is None or np.isnan(val):
                continue
            f.write(f"{frame},{val}\n")

def plot_dual_series(frames, right, left, title, ylabel, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(frames, right, label='Right')
    plt.plot(frames, left, label='Left')
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_bar_series(frames, bar_x, bar_y, bar_yx, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(frames, bar_x, label='Bar X')
    plt.plot(frames, bar_y, label='Bar Y')
    plt.plot(frames, bar_yx, label='Bar Y/X')
    plt.title("Barbell Features")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def read_bar_coordinates_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    frames = df.iloc[:, 0].tolist()
    x_centers = df.iloc[:, 1].tolist()
    y_centers = df.iloc[:, 2].tolist()
    return frames, x_centers, y_centers

def process_subject_folder(subject_path):
    print(f"\U0001F4C2 處理: {subject_path}")

    file_top_11m = os.path.join(subject_path, "yolo_skeleton_top_11m_interpolated_hampel.txt")
    file_rear = os.path.join(subject_path, "yolo_skeleton_interpolated_hampel.txt")
    file_bar = os.path.join(subject_path, "yolo_coordinates_interpolated_hampel.txt")

    # 11m view
    if os.path.exists(file_top_11m):
        frames_top_11m, coords_top_11m = parse_yolo_file(file_top_11m)
        r_arm_angle_11m = [calc_angle(p[2], p[0], p[4]) for p in coords_top_11m]
        l_arm_angle_11m = [calc_angle(p[3], p[1], p[5]) for p in coords_top_11m]
        r_dist_top_11m, l_dist_top_11m = process_wrist_to_shoulder_line(coords_top_11m)
        save_feature(os.path.join(subject_path, "right_torso_arm_angle_11m.txt"), frames_top_11m, r_arm_angle_11m)
        save_feature(os.path.join(subject_path, "left_torso_arm_angle_11m.txt"), frames_top_11m, l_arm_angle_11m)
        save_feature(os.path.join(subject_path, "right_wrist_to_shoulderline_11m.txt"), frames_top_11m, r_dist_top_11m)
        save_feature(os.path.join(subject_path, "left_wrist_to_shoulderline_11m.txt"), frames_top_11m, l_dist_top_11m)
        plot_dual_series(frames_top_11m, r_arm_angle_11m, l_arm_angle_11m, "Torso-Arm Angle (YOLO 11m)", "Angle (degrees)", os.path.join(subject_path, "plot_torso_arm_angle_yolo_11m.png"))
        plot_dual_series(frames_top_11m, r_dist_top_11m, l_dist_top_11m, "Wrist to Shoulder Line Distance (YOLO 11m)", "Distance (pixels)", os.path.join(subject_path, "plot_wrist_to_shoulder_yolo_11m.png"))

    # Barbell position
    if os.path.exists(file_bar):
        frames_bar, bar_x, bar_y = read_bar_coordinates_csv(file_bar)
        bar_yx = [y / x if x != 0 else None for x, y in zip(bar_x, bar_y)]
        save_feature(os.path.join(subject_path, "bar_x.txt"), frames_bar, bar_x)
        save_feature(os.path.join(subject_path, "bar_y.txt"), frames_bar, bar_y)
        save_feature(os.path.join(subject_path, "bar_y_div_x.txt"), frames_bar, bar_yx)
        plot_bar_series(frames_bar, bar_x, bar_y, bar_yx, os.path.join(subject_path, "plot_bar_features.png"))

    # Rear view
    if os.path.exists(file_rear):
        frames_rear, coords_rear = parse_yolo_file(file_rear)
        r_shoulder = [calc_angle(p[0], p[1], p[3]) if len(p) > 3 else None for p in coords_rear]
        l_shoulder = [calc_angle(p[1], p[0], p[2]) if len(p) > 2 else None for p in coords_rear]
        r_elbow = [180 - calc_angle(p[3], p[1], p[5]) if len(p) > 5 and calc_angle(p[3], p[1], p[5]) is not None else None for p in coords_rear]
        l_elbow = [180 - calc_angle(p[2], p[0], p[4]) if len(p) > 4 and calc_angle(p[2], p[0], p[4]) is not None else None for p in coords_rear]
        save_feature(os.path.join(subject_path, "right_shoulder_angle.txt"), frames_rear, r_shoulder)
        save_feature(os.path.join(subject_path, "left_shoulder_angle.txt"), frames_rear, l_shoulder)
        save_feature(os.path.join(subject_path, "right_elbow_angle.txt"), frames_rear, r_elbow)
        save_feature(os.path.join(subject_path, "left_elbow_angle.txt"), frames_rear, l_elbow)
        plot_dual_series(frames_rear, r_shoulder, l_shoulder, "Shoulder Angle (YOLO)", "Angle (degrees)", os.path.join(subject_path, "plot_shoulder_angle_yolo.png"))
        plot_dual_series(frames_rear, r_elbow, l_elbow, "Elbow Angle (YOLO)", "Angle (degrees)", os.path.join(subject_path, "plot_elbow_angle_yolo.png"))
        r_shoulder_y = [p[0][1] if len(p) > 0 else None for p in coords_rear]
        l_shoulder_y = [p[1][1] if len(p) > 1 else None for p in coords_rear]
        save_feature(os.path.join(subject_path, "right_shoulder_y.txt"), frames_rear, r_shoulder_y)
        save_feature(os.path.join(subject_path, "left_shoulder_y.txt"), frames_rear, l_shoulder_y)
        plot_dual_series(frames_rear, r_shoulder_y, l_shoulder_y, "Shoulder Y-Coordinate (YOLO Rear View)", "Y Position (pixels)", os.path.join(subject_path, "plot_shoulder_y.png"))

def process_all_subjects(root_path):
    for category in os.listdir(root_path):
        category_path = os.path.join(root_path, category)
        if not os.path.isdir(category_path):
            continue
        for subject in os.listdir(category_path):
            subject_path = os.path.join(category_path, subject)
            if not os.path.isdir(subject_path):
                continue
            process_subject_folder(subject_path)

if __name__ == "__main__":
    USE_LATEST = True  # ❗️切換手動指定資料夾還是自動判定最新的

    if USE_LATEST:
        recordings_dir = r"C:/Users/92A27/benchpress/recordings"
        all_folders = [os.path.join(recordings_dir, d) for d in os.listdir(recordings_dir) if os.path.isdir(os.path.join(recordings_dir, d))]
        base_path = os.path.join(max(all_folders, key=os.path.getmtime), '')
    else:
        base_path = r"E:/DATASET/abc"  # 手動指定

    # 🔍 取得 recordings 下所有子資料夾，並找出最新的
    all_folders = [os.path.join(recordings_dir, d) for d in os.listdir(recordings_dir) if os.path.isdir(os.path.join(recordings_dir, d))]
    if not all_folders:
        raise FileNotFoundError("❌ recordings 資料夾下沒有任何子資料夾")

    latest_folder = max(all_folders, key=os.path.getmtime)  # 依建立時間找最新資料夾
    base_path = os.path.join(latest_folder, '')  # base_path 最後補上斜線

    process_subject_folder(base_path)
    print("✅ Done")