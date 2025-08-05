import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

def interpolate_yolo(yolo_data):
    if len(yolo_data) == 0:
        return yolo_data

    frames = yolo_data[:, 0]
    interpolated_data = [frames]

    for col in range(1, 5):
        values = yolo_data[:, col]
        valid = ~np.isnan(values)
        if valid.sum() >= 2:
            interp_func = interp1d(frames[valid], values[valid], kind='linear', fill_value='extrapolate')
            interpolated_values = interp_func(frames)
        else:
            interpolated_values = np.nan_to_num(values)
        interpolated_data.append(interpolated_values)

    return np.stack(interpolated_data, axis=1)

def load_yolo_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            values = line.strip().split(',')
            if len(values) == 5:
                try:
                    row = [float(v) if v.strip() != '' else np.nan for v in values]
                    data.append(row)
                except:
                    continue
    return np.array(data)

def save_yolo_output(folder, yolo_interp_data):
    np.savetxt(
        os.path.join(folder, 'yolo_coordinates_interpolated_hampel.txt'),
        yolo_interp_data,
        delimiter=',',
        fmt='%d,%.8f,%.8f,%.8f,%.8f'
    )

def process_subject_folder(folder):
    yolo_file = os.path.join(folder, 'yolo_coordinates_hampel.txt')
    if not os.path.exists(yolo_file):
        return

    try:
        yolo_raw = load_yolo_data(yolo_file)
        if yolo_raw.shape[0] == 0:
            return

        yolo_interp = interpolate_yolo(yolo_raw)
        save_yolo_output(folder, yolo_interp)
    except Exception as e:
        print(f"⚠️ 錯誤處理 {folder}: {e}")

def process_all_folders(base_root):
    if not os.path.exists(base_root):
        print(f"❌ 找不到資料夾: {base_root}")
        return

    for category in os.listdir(base_root):
        category_path = os.path.join(base_root, category)
        if not os.path.isdir(category_path):
            continue
        for subject in tqdm(os.listdir(category_path), desc=f"處理 {category}"):
            subject_path = os.path.join(category_path, subject)
            if os.path.isdir(subject_path):
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