import numpy as np
import re
import os
from tqdm import tqdm
from scipy.interpolate import interp1d

def read_bar_frames(file_path):
    frames = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0].isdigit():
                frames.append(int(parts[0]))
    return sorted(frames)

def read_skeleton_data_with_nan(file_path):
    skeleton_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r'Frame (\d+):\s*\[\[(.*?)\]\]', line)
            if match:
                frame = int(match.group(1))
                points_str = match.group(2).replace("(", "[").replace(")", "]")
                try:
                    points = eval(f"[{points_str.replace('NaN', 'np.nan')}]")
                    skeleton_data[frame] = np.array(points)
                except Exception as e:
                    print(f"❌ Skipping frame {frame}: {e}")
    return skeleton_data

def interpolate_skeleton_with_nan(bar_frames, skeleton_data, num_keypoints):
    interpolated_data = {}
    keypoint_x = {i: [] for i in range(num_keypoints)}
    keypoint_y = {i: [] for i in range(num_keypoints)}
    valid_frames = sorted(skeleton_data.keys())

    for frame in valid_frames:
        points = skeleton_data[frame]
        for i in range(num_keypoints):
            x, y = points[i]
            keypoint_x[i].append((frame, x))
            keypoint_y[i].append((frame, y))

    interp_x_func = {}
    interp_y_func = {}

    for i in range(num_keypoints):
        fx = np.array([f for f, v in keypoint_x[i] if not np.isnan(v)])
        vx = np.array([v for f, v in keypoint_x[i] if not np.isnan(v)])
        fy = np.array([f for f, v in keypoint_y[i] if not np.isnan(v)])
        vy = np.array([v for f, v in keypoint_y[i] if not np.isnan(v)])

        interp_x_func[i] = interp1d(fx, vx, kind='linear', fill_value='extrapolate') if len(fx) > 1 else None
        interp_y_func[i] = interp1d(fy, vy, kind='linear', fill_value='extrapolate') if len(fy) > 1 else None

    for frame in bar_frames:
        points = []
        for i in range(num_keypoints):
            x = float(interp_x_func[i](frame)) if interp_x_func[i] is not None else np.nan
            y = float(interp_y_func[i](frame)) if interp_y_func[i] is not None else np.nan
            points.append((x, y))
        interpolated_data[frame] = points

    return interpolated_data

def write_interpolated_skeleton(output_path, interpolated_data):
    with open(output_path, 'w') as f:
        for frame, points in sorted(interpolated_data.items()):
            points_str = ", ".join([f"({p[0]}, {p[1]})" for p in points])
            f.write(f"Frame {frame}: [[{points_str}]]\n")

def process_subject_folder(folder):
    bar_file = os.path.join(folder, 'yolo_coordinates_interpolated_hampel.txt')
    if not os.path.exists(bar_file):
        return

    tasks = [
        ('yolo_skeleton_top_11m_hampel.txt', 'yolo_skeleton_top_11m_interpolated_hampel.txt', 8),
        ('yolo_skeleton_hampel.txt', 'yolo_skeleton_interpolated_hampel.txt', 6)
    ]

    for input_file, output_file, num_kpts in tasks:
        input_path = os.path.join(folder, input_file)
        output_path = os.path.join(folder, output_file)

        if not os.path.exists(input_path):
            continue
        if os.path.exists(output_path):
            continue

        try:
            bar_frames = read_bar_frames(bar_file)
            skeleton_data = read_skeleton_data_with_nan(input_path)
            if not skeleton_data:
                continue
            print(f"✅ [{input_file}] 讀入 {len(skeleton_data)} 幀")
            interpolated_data = interpolate_skeleton_with_nan(bar_frames, skeleton_data, num_kpts)
            write_interpolated_skeleton(output_path, interpolated_data)
        except Exception as e:
            print(f"⚠️ 錯誤處理 {folder} 中的 {input_file}: {e}")

def process_all_folders(base_root):
    for category in os.listdir(base_root):
        category_path = os.path.join(base_root, category)
        if not os.path.isdir(category_path):
            continue
        for subject in tqdm(os.listdir(category_path), desc=f"處理類別：{category}"):
            subject_path = os.path.join(category_path, subject)
            if os.path.isdir(subject_path):
                process_subject_folder(subject_path)

# if __name__ == "__main__":
#     base_dir = r"F:\DATASET"
#     process_all_folders(base_dir)
#     print("✅ 所有 yolo_skeleton 檔案已補齊內插（如尚未存在）")

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
    print("✅ 所有 yolo_skeleton 檔案已補齊內插（如尚未存在）")
