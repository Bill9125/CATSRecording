import tkinter as tk
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
from scipy.signal import find_peaks, savgol_filter
import cv2
import math
import os
import numpy as np
import pandas as pd



def read_skeleton_data(filename):
    data = {}
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            if len(values) < 4:
                continue
            try:
                frame, joint, x, y = map(int, values[:4])
                if frame not in data:
                    data[frame] = {}
                data[frame][joint] = (x, y)
            except ValueError:
                continue
    return data

def calculate_angles(data):
    frames = sorted(data.keys())
    angles = []

    for frame in frames:
        joints = data[frame]
        if all(k in joints for k in [12, 14, 16]):
            angles.append(calculate_angle(joints[12], joints[14], joints[16]))
        else:
            angles.append(None)

    return frames, angles

def calculate_angle(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba, bc = a - b, c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def find_valleys(smoothed_angles, peaks, search_range=10, min_valley_value=170, min_depth=10):
    valleys = []
    valleys1 = []

    for peak in peaks:
        # 在峰值前後各 `search_range` 幀內找最小值
        left_bound = max(0, peak - search_range)
        right_bound = min(len(smoothed_angles) - 1, peak + search_range)

        # 找到左邊和右邊的波谷
        left_min_index = left_bound + np.argmin(smoothed_angles[left_bound:peak])
        right_min_index = peak + np.argmin(smoothed_angles[peak:right_bound + 1])

        left_min_value = smoothed_angles[left_min_index]
        right_min_value = smoothed_angles[right_min_index]
        peak_value = smoothed_angles[peak]

        # 檢查條件：波谷值小於 170，且谷底夠深 (峰值 - 谷底 >= min_depth)
        if left_min_value < min_valley_value and (peak_value - left_min_value) >= min_depth:
            valleys.append(left_min_index)

        if right_min_value < min_valley_value and (peak_value - right_min_value) >= min_depth:
            valleys1.append(right_min_index)

    return valleys, valleys1

def read_yolo_data(yolo_file):
    """ 讀取 yolo_coordinates_interpolated.txt，回傳 {frame: x} 字典 """
    yolo_data = {}
    with open(yolo_file, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            if len(values) < 3:
                continue
            try:
                frame = int(values[0])
                x_value = float(values[2])  # 第三個數值是 X 座標
                yolo_data[frame] = x_value
            except ValueError:
                continue
    return yolo_data

def split_video(video_path, output_folder, valleys, valleys1, yolo_file, fps=30):
    """
    根據 valleys (黃色谷底) 和 valleys1 (綠色谷底) 分割影片，
    **只存 X 座標變化超過 20 的影片**，如果變化 <= 20，則不存成影片。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 讀取影片
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 格式

    # 讀取 yolo 數據
    yolo_data = read_yolo_data(yolo_file)

    for i, (start_frame, end_frame) in enumerate(zip(valleys, valleys1)):
        if start_frame >= end_frame or start_frame >= total_frames or end_frame >= total_frames:
            print(f"跳過無效範圍: 開始 {start_frame}, 結束 {end_frame}")
            continue

        # 取得該範圍內的 yolo X 值變化
        x_values = [yolo_data[f] for f in range(start_frame, end_frame + 1) if f in yolo_data]
        x_range = max(x_values) - min(x_values) if x_values else 0

        # **如果變化小於等於 20，則跳過該片段**
        if x_range <= 50 or np.isnan(x_range):
            print(f"跳過片段 {i+1}（X 變化範圍: {x_range:.2f}，過小）")
            continue

        # 存影片
        output_path = os.path.join(output_folder, f"clip_{i+1}.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        print(f"影片儲存: {output_path}（X 變化範圍: {x_range:.2f}）")

    cap.release()
    print("所有片段處理完成！")


def split_skeleton_data(original_skeleton_path, output_folder, valleys, valleys1, yolo_file):
    """
    根據 valleys (黃色谷底) 和 valleys1 (綠色谷底) 分割骨架數據，
    **只存 X 座標變化超過 20 的片段**，如果變化 <= 20，則不存。
    """
    # 讀取骨架數據檔案
    with open(original_skeleton_path, 'r') as file:
        lines = file.readlines()

    # 讀取 yolo 數據
    yolo_data = read_yolo_data(yolo_file)

    for i, (start_frame, end_frame) in enumerate(zip(valleys, valleys1)):
        if start_frame >= end_frame:
            print(f"跳過無效範圍: 開始 {start_frame}, 結束 {end_frame}")
            continue

        # 計算該片段內的 X 座標變化範圍
        x_values = [yolo_data[f] for f in range(start_frame, end_frame + 1) if f in yolo_data]
        x_range = max(x_values) - min(x_values) if x_values else 0

        # **如果變化小於等於 20，則跳過該片段**
        if x_range <= 50 or np.isnan(x_range):
            print(f"跳過骨架數據 {i+1}（X 變化範圍: {x_range:.2f}，過小）")
            continue

        # 存骨架數據
        clip_folder = os.path.join(output_folder)
        os.makedirs(clip_folder, exist_ok=True)

        output_skeleton_path = os.path.join(clip_folder, f"skeleton_{i+1}.txt")
        with open(output_skeleton_path, 'w') as out_file:
            for line in lines:
                values = line.strip().split(',')
                if len(values) < 4:
                    continue
                try:
                    frame = int(values[0])  # 解析 frame 數
                    if start_frame <= frame <= end_frame:
                        out_file.write(line)  # 保留符合範圍的行
                except ValueError:
                    continue

        print(f"骨架數據儲存: {output_skeleton_path}（Y 變化範圍: {x_range:.2f}）")

    print("所有骨架數據處理完成！")


def split_bar_data(original_bar_path, output_folder, valleys, valleys1, yolo_file):
    """
    根據 valleys (黃色谷底) 和 valleys1 (綠色谷底) 分割槓鈴數據，
    **只存 X 座標變化超過 20 的片段**，如果變化 ≤ 20，則不存。
    """
    # 讀取槓鈴數據檔案
    with open(original_bar_path, 'r') as file:
        lines = file.readlines()

    # 讀取 yolo 數據
    yolo_data = read_yolo_data(yolo_file)

    for i, (start_frame, end_frame) in enumerate(zip(valleys, valleys1)):
        if start_frame >= end_frame:
            print(f"跳過無效範圍: 開始 {start_frame}, 結束 {end_frame}")
            continue

        # 計算該片段內的 X 座標變化範圍
        x_values = [yolo_data[f] for f in range(start_frame, end_frame + 1) if f in yolo_data]
        x_range = max(x_values) - min(x_values) if x_values else 0

        # **如果變化小於等於 20，則跳過該片段**
        if x_range <= 50 or np.isnan(x_range):
            print(f"跳過槓鈴數據 {i+1}（X 變化範圍: {x_range:.2f}，過小）")
            continue

        # 存槓鈴數據
        clip_folder = os.path.join(output_folder)
        os.makedirs(clip_folder, exist_ok=True)

        output_bar_path = os.path.join(clip_folder, f"bar_{i+1}.txt")
        with open(output_bar_path, 'w') as out_file:
            for line in lines:
                values = line.strip().split(',')
                if len(values) < 4:
                    continue
                try:
                    frame = int(values[0])  # 解析 frame 數
                    if start_frame <= frame <= end_frame:
                        out_file.write(line)  # 保留符合範圍的行
                except ValueError:
                    continue

        print(f"槓鈴數據儲存: {output_bar_path}（X 變化範圍: {x_range:.2f}）")

    print("所有槓鈴數據處理完成！")

        
def plot_metrics_in_tkinter():
    global root, canvas, valleys, valleys1
    root = tk.Tk()
    root.title("Left Knee Angle Analysis")
    root.geometry("1000x600")

    fig, axes = Figure(figsize=(10, 6), dpi=100), [None, None]
    axes[0] = fig.add_subplot(2, 1, 1)  # 原始數據圖
    axes[1] = fig.add_subplot(2, 1, 2)  # 平滑後數據圖

    # 過濾掉 None 值
    valid_frames = [frames[i] for i in range(len(left_knee_angles)) if left_knee_angles[i] is not None]
    valid_angles = np.array([angle for angle in left_knee_angles if angle is not None])

    # **平滑化曲線**
    smoothed_angles = savgol_filter(valid_angles, window_length=11, polyorder=3)

    # 找出 170 度以上的峰值，間隔 10 幀以上
    peaks, _ = find_peaks(smoothed_angles, height=160, distance=55, prominence=5)

    # 找出峰值前後的谷底
    all_valleys, all_valleys1 = find_valleys(smoothed_angles, peaks, search_range=90, min_valley_value=170, min_depth=10)

    # 新的波谷列表
    valleys, valleys1 = [], []

    # 確保每個峰值左右都有波谷
    for peak in peaks:
        # 檢查左側的波谷
        left_valley = None
        for valley in all_valleys:
            if valley < peak:
                left_valley = valley
            else:
                break
        
        # 檢查右側的波谷
        right_valley = None
        for valley1 in all_valleys1:
            if valley1 > peak:
                right_valley = valley1
                break

        # 如果找到的左側和右側波谷都存在，則將其添加到各自的列表中
        if left_valley is not None and right_valley is not None:
            valleys.append(left_valley)
            valleys1.append(right_valley)

#     ======= 繪製原始數據 =======
#     axes[0].plot(valid_frames, valid_angles, color='gray', alpha=0.7)
#     axes[0].plot(np.array(valid_frames)[peaks], valid_angles[peaks], 'ro')  # 紅色標記峰值
#     axes[0].plot(np.array(valid_frames)[valleys], valid_angles[valleys], 'yo')  # 黃色標記谷底
#     axes[0].plot(np.array(valid_frames)[valleys1], valid_angles[valleys1], 'go')  # 綠色標記谷底
#     axes[0].set_title("Raw Left Knee Angle with Peaks and Valleys")
#     axes[0].set_xlabel("Frame")
#     axes[0].set_ylabel("Angle (degrees)")
#     axes[0].legend()
#     axes[0].grid()

#     # ======= 繪製平滑後數據 =======
#     axes[1].plot(valid_frames, smoothed_angles, color='blue')
#     axes[1].plot(np.array(valid_frames)[peaks], smoothed_angles[peaks], 'ro')
#     axes[1].plot(np.array(valid_frames)[valleys], smoothed_angles[valleys], 'yo')
#     axes[1].plot(np.array(valid_frames)[valleys1], smoothed_angles[valleys1], 'go')
#     axes[1].set_title("Smoothed Left Knee Angle with Peaks and Valleys")
#     axes[1].set_xlabel("Frame")
#     axes[1].set_ylabel("Angle (degrees)")
#     axes[1].legend()
#     axes[1].grid()

    print(valleys, valleys1)

#     canvas = FigureCanvasTkAgg(fig, root)
#     canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
#     root.mainloop()

def calculate_distance(x1, y1, x2, y2):
    """計算兩點之間的歐氏距離"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_angle1(x1, y1, x2, y2, x3, y3):
    """計算由三點構成的夾角，x2, y2 是頂點"""
    c = calculate_distance(x2, y2, x3, y3)
    a = calculate_distance(x1, y1, x3, y3)
    b = calculate_distance(x1, y1, x2, y2)

    if b == 0 or c == 0:
        return 0.0

    try:
        cos_theta = (b**2 + c**2 - a**2) / (2 * b * c)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle = math.acos(cos_theta)
        return math.degrees(angle)
    except ValueError:
        return 0.0

def process_file(input_file, output_file):
    joint_data = {}

    try:
        # 讀取檔案並整理資料
        with open(input_file, 'r') as f:
            for line in f:
                frame, index, x, y = map(float, line.strip().split(','))
                frame = int(frame)
                index = int(index)
                if frame not in joint_data:
                    joint_data[frame] = {}
                joint_data[frame][index] = (x, y)

        # 計算角度和長度，並保存結果
        with open(output_file, 'w') as f_out:
#             f_out.write("frame,hip_angle,knee_angle,body_length,knee_hip_ratio\n")
            for frame, joints in sorted(joint_data.items()):
                if all(idx in joints for idx in [6, 12, 14, 16]):
                    x6, y6 = joints[6]
                    x12, y12 = joints[12]
                    x14, y14 = joints[14]
                    x16, y16 = joints[16]

                    hip_angle = calculate_angle1(x6, y6, x12, y12, x14, y14)
                    knee_angle = calculate_angle1(x12, y12, x14, y14, x16, y16)
                    body_length = calculate_distance(x6, y6, x12, y12)
                    knee_hip_ratio = knee_angle / hip_angle if hip_angle != 0 else 0

                    f_out.write(f"{frame},{hip_angle:.2f},{knee_angle:.2f},{body_length:.2f},{knee_hip_ratio:.2f}\n")
    except Exception as e:
        print(f"處理檔案時發生錯誤：{input_file}，錯誤訊息：{e}")

def process_all_folders(base_path):
    # 遞迴遍歷所有子資料夾
    for root, dirs, files in os.walk(base_path):
        for filename in files:
            if filename.startswith("skeleton_") and filename.endswith(".txt"):
                input_file = os.path.join(root, filename)
                
                # 在上層資料夾建立 processed 資料夾
                parent_folder = os.path.dirname(root)
                output_folder = os.path.join(parent_folder, "angle")
                os.makedirs(output_folder, exist_ok=True)

                # 輸出檔案路徑
                output_file = os.path.join(output_folder, f"angle_{filename}")
                process_file(input_file, output_file)
                print(f"已處理檔案：{input_file}，結果保存至：{output_file}")

def linear_interpolate_fixed_length(df, frame_col=0, target_length=110):
    """對 DataFrame 進行線性內插，使其長度固定為 target_length。"""
    df.set_index(df.columns[frame_col], inplace=True)  # 設定時間軸為索引
    new_index = np.linspace(df.index.min(), df.index.max(), target_length)
    df = df.reindex(df.index.union(new_index)).interpolate(method='linear').loc[new_index]
    df.reset_index(drop=True, inplace=True)
    return df

def merge_and_interpolate(angle_path, bar_path, output_path, target_length=110):
    """合併 angle 和 bar 資料夾內的檔案，並進行線性內插。"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    angle_files = sorted([f for f in os.listdir(angle_path) if f.startswith("angle_skeleton_")])
    bar_files = sorted([f for f in os.listdir(bar_path) if f.startswith("bar_")])
    
    # 確保檔案名稱匹配
    angle_dict = {f.split("_")[-1]: f for f in angle_files}
    bar_dict = {f.split("_")[-1]: f for f in bar_files}
    
    common_keys = set(angle_dict.keys()) & set(bar_dict.keys())  # 找到對應的檔案

    for key in sorted(common_keys):
        angle_file = angle_dict[key]
        bar_file = bar_dict[key]

        try:
            angle_df = pd.read_csv(os.path.join(angle_path, angle_file), header=None)
            bar_df = pd.read_csv(os.path.join(bar_path, bar_file), header=None)
            
            # 選擇需要的欄位
            angle_df = angle_df.iloc[:, [0, 1, 2, 3]]  # 包含 frame_col (假設是第 0 欄)
            bar_df = bar_df.iloc[:, [1, 2]]

            merged_df = pd.concat([angle_df, bar_df], axis=1)

            # 進行內插，確保 frame_col 為索引
            merged_df = linear_interpolate_fixed_length(merged_df, frame_col=0, target_length=target_length)

            output_file = os.path.join(output_path, f"merged_{key}")
            merged_df.to_csv(output_file, index=False, header=False, float_format="%.2f")

            print(f"Processed: {angle_file} + {bar_file} -> {output_file}")

        except Exception as e:
            print(f"Error processing {angle_file} and {bar_file}: {e}")
def process_all_folders1(root_path, target_length=110):
    """遍歷 root_path 內的所有子資料夾，並處理其中的 angle 和 bar。"""
    for category in os.listdir(root_path):
        category_path = os.path.join(root_path, category)
        if not os.path.isdir(category_path):
            continue  # 跳過非資料夾的檔案

        for recording in os.listdir(category_path):
            recording_path = os.path.join(category_path, recording)
            if not os.path.isdir(recording_path):
                continue
            
            angle_path = os.path.join(recording_path, "angle")
            bar_path = os.path.join(recording_path, "bar")
            output_path = os.path.join(recording_path, "out")

            if os.path.exists(angle_path) and os.path.exists(bar_path):
                print(f"Processing {recording_path} ...")
                merge_and_interpolate(angle_path, bar_path, output_path, target_length)

def remove_outliers(df):
    """移除超過3個標準差的極端值，並用該列均值取代。"""
    mean = df.mean()
    std = df.std()
    
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    # 過濾掉極端值，超過範圍的值用均值取代
    df_filtered = df.mask((df < lower_bound) | (df > upper_bound), mean, axis=1)
    return df_filtered

def process_filtered_files(root_path):
    """遍歷所有 merged 資料夾內的檔案，去除極端值後輸出到 filtered 資料夾。"""
    for category in os.listdir(root_path):
        category_path = os.path.join(root_path, category)
        if not os.path.isdir(category_path):
            continue  # 跳過非資料夾的檔案

        for recording in os.listdir(category_path):
            recording_path = os.path.join(category_path, recording)
            if not os.path.isdir(recording_path):
                continue
            
            merged_path = os.path.join(recording_path, "out")  # 原始數據
            filtered_path = os.path.join(recording_path, "data/filtered")  # 存放處理後數據的資料夾

            if not os.path.exists(merged_path):
                continue  # 若 merged 資料夾不存在則跳過
            if not os.path.exists(filtered_path):
                os.makedirs(filtered_path)  # 建立 filtered 資料夾

            for file in sorted(os.listdir(merged_path)):
                if file.startswith("merged_"):  # 只處理 merged_ 開頭的檔案
                    input_file = os.path.join(merged_path, file)
                    output_file = os.path.join(filtered_path, file)

                    try:
                        df = pd.read_csv(input_file, header=None)
                        df_filtered = remove_outliers(df)  # 移除極端值

                        df_filtered.to_csv(output_file, index=False, header=False, float_format="%.4f")
                        print(f"Processed Filtered: {input_file} -> {output_file}")

                    except Exception as e:
                        print(f"Error processing {input_file}: {e}")

def compute_differences(df):
    """計算 DataFrame 每行與前一行的變化量，第一行補 0。"""
    delta_df = df.diff().fillna(0)  # 第一行 NaN 填補為 0
    return delta_df

def process_delta_files(root_path):
    """遍歷所有 out 資料夾內的 merged_ 檔案，計算變化量並輸出到 delta 資料夾。"""
    for category in os.listdir(root_path):
        category_path = os.path.join(root_path, category)
        if not os.path.isdir(category_path):
            continue  # 跳過非資料夾的檔案

        for recording in os.listdir(category_path):
            recording_path = os.path.join(category_path, recording)
            if not os.path.isdir(recording_path):
                continue
            
            out_path = os.path.join(recording_path, "data\\filtered")
            delta_path = os.path.join(recording_path, "data\\filtered_delta")

            if not os.path.exists(out_path):
                continue  # 若 out 資料夾不存在則跳過
            if not os.path.exists(delta_path):
                os.makedirs(delta_path)  # 建立 delta 資料夾

            for file in sorted(os.listdir(out_path)):
                if file.startswith("merged_"):  # 只處理 merged_ 開頭的檔案
                    input_file = os.path.join(out_path, file)
                    output_file = os.path.join(delta_path, file)

                    try:
                        df = pd.read_csv(input_file, header=None)
                        df_diff = compute_differences(df)  # 計算變化量

                        df_diff.to_csv(output_file, index=False, header=False, float_format="%.4f")
                        print(f"Processed Δ: {input_file} -> {output_file}")

                    except Exception as e:
                        print(f"Error processing {input_file}: {e}")

def compute_second_differences(df):
    """計算 DataFrame 每行與前一行的變化量的變化量，第一行補 0。"""
    delta2_df = df.diff().fillna(0)  # 計算變化量的變化量
    return delta2_df

def process_second_differences(root_path):
    """遍歷所有 delta 資料夾內的檔案，計算變化量的變化量並輸出到 delta2 資料夾。"""
    for category in os.listdir(root_path):
        category_path = os.path.join(root_path, category)
        if not os.path.isdir(category_path):
            continue  # 跳過非資料夾的檔案

        for recording in os.listdir(category_path):
            recording_path = os.path.join(category_path, recording)
            if not os.path.isdir(recording_path):
                continue
            
            delta_path = os.path.join(recording_path, "data/filtered_delta")
            delta2_path = os.path.join(recording_path, "data/filtered_delta_square")

            if not os.path.exists(delta_path):
                continue  # 若 delta 資料夾不存在則跳過
            if not os.path.exists(delta2_path):
                os.makedirs(delta2_path)  # 建立 delta2 資料夾

            for file in sorted(os.listdir(delta_path)):
                if file.startswith("merged_"):  # 只處理 merged_ 開頭的檔案
                    input_file = os.path.join(delta_path, file)
                    output_file = os.path.join(delta2_path, file)

                    try:
                        df = pd.read_csv(input_file, header=None)
                        df_delta2 = compute_second_differences(df)  # 計算變化量的變化量

                        df_delta2.to_csv(output_file, index=False, header=False, float_format="%.4f")
                        print(f"Processed Δ²: {input_file} -> {output_file}")

                    except Exception as e:
                        print(f"Error processing {input_file}: {e}")

def compute_delta_ratio(data):
    """計算變化量與原始值的比 (B - A) / A"""
    epsilon = 1e-6  # 避免 A 為 0
    delta_ratio = (data[1:] - data[:-1]) / (data[:-1] + epsilon)
    delta_ratio = np.vstack([np.zeros((1, data.shape[1])) , delta_ratio])  # 第一行補 0
    return delta_ratio

def process_delta_ratio(root_path, input_folder="data\\filtered", output_folder="data\\filtered_delta2"):
    """讀取 `out/`，計算 (B - A) / A 並存入 `out_delta2/`"""
    for category in os.listdir(root_path):
        category_path = os.path.join(root_path, category)
        if not os.path.isdir(category_path):
            continue

        for recording in os.listdir(category_path):
            recording_path = os.path.join(category_path, recording)
            if not os.path.isdir(recording_path):
                continue

            input_path = os.path.join(recording_path, input_folder)
            output_path = os.path.join(recording_path, output_folder)

            if not os.path.exists(input_path):
                continue  # 沒有這個資料夾就跳過
            
            os.makedirs(output_path, exist_ok=True)  # 確保輸出資料夾存在

            for file in sorted(os.listdir(input_path)):
                if file.startswith("merged_"):  # 只處理 merged_*.txt
                    file_path = os.path.join(input_path, file)
                    try:
                        df = pd.read_csv(file_path, header=None)
                        delta_ratio_data = compute_delta_ratio(df.to_numpy(dtype=np.float32))
                        pd.DataFrame(delta_ratio_data).to_csv(os.path.join(output_path, file), 
                                                              index=False, header=False, float_format="%.6f")
                        print(f"Processed delta ratio: {file_path} -> {output_path}/{file}")

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

def z_score_normalization(df):
    """對 DataFrame 進行 Z-score 標準化，每一列 (特徵) 依據自身均值與標準差計算。"""
    mean = df.mean()
    std = df.std()
    
    std.replace(0, 1, inplace=True)  # 避免標準差為 0 導致除以 0
    zscore_df = (df - mean) / std
    return zscore_df

def process_zscore_from_merged(root_path):
    """遍歷所有 merged 資料夾內的檔案，對數據進行 Z-score 標準化並輸出到 zscore 資料夾。"""
    for category in os.listdir(root_path):
        category_path = os.path.join(root_path, category)
        if not os.path.isdir(category_path):
            continue  # 跳過非資料夾的檔案

        for recording in os.listdir(category_path):
            recording_path = os.path.join(category_path, recording)
            if not os.path.isdir(recording_path):
                continue
            
            merged_path = os.path.join(recording_path, "data\\filtered")  # 原始數據
            zscore_path = os.path.join(recording_path, "data\\filtered_zscore")  # 存放 Z-score 的資料夾

            if not os.path.exists(merged_path):
                continue  # 若 merged 資料夾不存在則跳過
            if not os.path.exists(zscore_path):
                os.makedirs(zscore_path)  # 建立 zscore 資料夾

            for file in sorted(os.listdir(merged_path)):
                if file.startswith("merged_"):  # 只處理 merged_ 開頭的檔案
                    input_file = os.path.join(merged_path, file)
                    output_file = os.path.join(zscore_path, file)

                    try:
                        df = pd.read_csv(input_file, header=None)
                        df_zscore = z_score_normalization(df)  # 計算 Z-score 標準化

                        df_zscore.to_csv(output_file, index=False, header=False, float_format="%.4f")
                        print(f"Processed Z-score: {input_file} -> {output_file}")

                    except Exception as e:
                        print(f"Error processing {input_file}: {e}")

def normalize_data(data):
    """將數據正規化到 [-1, 1]"""
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    
    # 防止除以零
    range_val = max_val - min_val
    range_val[range_val == 0] = 1  # 避免 NaN，若 min==max，則保持為 0

    norm_data = 2 * (data - min_val) / range_val - 1
    return norm_data

def process_normalization(root_path, input_folder, output_folder):
    """讀取 out_delta/ 或 out_delta2/，正規化後存入新資料夾"""
    for category in os.listdir(root_path):
        category_path = os.path.join(root_path, category)
        if not os.path.isdir(category_path):
            continue  # 跳過非資料夾

        for recording in os.listdir(category_path):
            recording_path = os.path.join(category_path, recording)
            if not os.path.isdir(recording_path):
                continue

            input_path = os.path.join(recording_path, input_folder)
            output_path = os.path.join(recording_path, output_folder)

            if not os.path.exists(input_path):
                continue  # 沒有這個資料夾就跳過
            
            os.makedirs(output_path, exist_ok=True)  # 確保輸出資料夾存在

            for file in sorted(os.listdir(input_path)):
                if file.startswith("merged_"):  # 處理所有 merged_*.txt
                    file_path = os.path.join(input_path, file)
                    try:
                        df = pd.read_csv(file_path, header=None)
                        norm_data = normalize_data(df.to_numpy(dtype=np.float32))
                        pd.DataFrame(norm_data).to_csv(os.path.join(output_path, file), 
                                                       index=False, header=False, float_format="%.6f")
                        print(f"Normalized: {file_path} -> {output_path}/{file}")

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

# 指定最上層資料夾
base_path = "D:/labdata/MOCAP/test/Category_1/"
root_path = 'D:\\labdata\\MOCAP\\test\\'
# 取得 base_path 下的所有子資料夾
for sub_folder in os.listdir(base_path):
    path = os.path.join(base_path, sub_folder)

    # 確保 path 是資料夾
    if not os.path.isdir(path):
        continue


    # 設定檔案路徑
    skeleton_file_path = os.path.join(path, 'yolo_skeleton_coordinates_1st_interp.txt')
    bar_file_path = os.path.join(path, 'yolo_coordinates_interpolated.txt')
    video_path = os.path.join(path, 'vision1.avi')
    output_folder = os.path.join(path, 'clips')
    output_folder1 = os.path.join(path, 'skeleton')

    output_folder2 = os.path.join(path, 'bar')

    # 檢查檔案是否存在，避免錯誤
    if not os.path.exists(skeleton_file_path):
        print(f"檔案 {skeleton_file_path} 不存在，跳過該資料夾")
        continue

#         讀取數據
    skeleton_data = read_skeleton_data(skeleton_file_path)
    frames, left_knee_angles = calculate_angles(skeleton_data)

    # 顯示波形圖
    plot_metrics_in_tkinter()
    split_video(video_path, output_folder, valleys, valleys1,bar_file_path, fps=30)
    # 執行切割
    split_skeleton_data(skeleton_file_path, output_folder1, valleys, valleys1,bar_file_path)
    split_bar_data(bar_file_path, output_folder2, valleys, valleys1,bar_file_path)
    

process_all_folders(base_path)



# Example usage
process_all_folders1(root_path, target_length=110)
process_filtered_files(root_path)
process_delta_files(root_path)
process_second_differences(root_path)
process_delta_ratio(root_path)
process_zscore_from_merged(root_path)
# 處理 delta 和 delta2
process_normalization(root_path, "data\\filtered", "data1\\filtered_norm")
process_normalization(root_path, "data\\filtered_delta", "data1\\filtered_delta_norm")
process_normalization(root_path, "data\\filtered_delta2", "data1\\filtered_delta2_norm")
process_normalization(root_path, "data\\filtered_zscore", "data1\\filtered_zscore_norm")
process_normalization(root_path, "data\\filtered_delta_square", "data1\\filtered_delta_square_norm")
