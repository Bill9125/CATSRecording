import os
import numpy as np
import re

def hampel_filter(series, window_size=7, n_sigmas=3):
    n = len(series)
    k = 1.4826
    half_window = window_size // 2
    outlier_mask = np.zeros(n, dtype=bool)

    for i in range(half_window, n - half_window):
        window = series[i - half_window:i + half_window + 1]
        median = np.nanmedian(window)
        mad = k * np.nanmedian(np.abs(window - median))
        if mad == 0:
            continue
        if np.abs(series[i] - median) > n_sigmas * mad:
            outlier_mask[i] = True

    return outlier_mask

def parse_skeleton_line(line):
    match = re.match(r"Frame (\d+): \[\[(.+)\]\]", line.strip())
    if not match:
        return None, None
    frame_idx = int(match.group(1))
    coord_str = match.group(2)

    # 抓所有像 (x, y) 的對
    coord_parts = re.findall(r"\(([^)]+)\)", coord_str)
    coords = []
    for part in coord_parts:
        try:
            x_str, y_str = part.strip().split(",")
            x, y = float(x_str), float(y_str)
            coords.append((x, y))
        except Exception as e:
            print(f"⚠️ 解析失敗：'{part}'，錯誤訊息：{e}")
            coords.append((np.nan, np.nan))  # 無法解析的點標為 NaN
    return frame_idx, coords

def process_skeleton_file(input_path, output_path):
    all_frames = []
    all_points = []

    with open(input_path, "r") as f:
        for line in f:
            frame_idx, coords = parse_skeleton_line(line)
            if coords is not None and len(coords) == 8:
                all_frames.append(frame_idx)
                all_points.append(coords)

    if len(all_points) == 0:
        print(f"⚠️ 無有效資料：{input_path}")
        return False

    all_points = np.array(all_points)  # (num_frames, 8, 2)
    mask = np.zeros_like(all_points[:, :, 0], dtype=bool)

    for joint_idx in range(6):
        x_series = all_points[:, joint_idx, 0]
        y_series = all_points[:, joint_idx, 1]
        x_outliers = hampel_filter(x_series)
        y_outliers = hampel_filter(y_series)
        mask[:, joint_idx] = x_outliers | y_outliers

    cleaned_points = all_points.copy()
    cleaned_points[mask] = np.nan

    with open(output_path, "w") as f:
        for i, frame in enumerate(all_frames):
            coords_str = ", ".join(
                f"(NaN, NaN)" if np.isnan(x) or np.isnan(y) else f"({x}, {y})"
                for x, y in cleaned_points[i]
            )
            f.write(f"Frame {frame}: [[{coords_str}]]\n")

    return True

# def process_all_skeletons(root_dir):
#     processed_files = []
#     for folder, _, files in os.walk(root_dir):
#         for file in files:
#             if file == "yolo_skeleton_top_11m.txt":
#                 input_path = os.path.join(folder, file)
#                 output_path = os.path.join(folder, "yolo_skeleton_top_11m_hampel.txt")
#                 success = process_skeleton_file(input_path, output_path)
#                 if success:
#                     print(f"✅ 已處理：{output_path}")
#                     processed_files.append(output_path)


    # # === 處理報告 ===
    # print("\n📋 處理完成列表：")
    # for path in processed_files:
    #     print(f"  ➤ {path}")
    # print(f"\n✅ 共處理 {len(processed_files)} 個檔案")

def process_all_skeletons(root_dir):
    processed_files = []
    files = os.listdir(root_dir)  # 不用 walk，只取該資料夾
    for file in files:
        if file == "yolo_skeleton_top_11m.txt":
            input_path = os.path.join(root_dir, file)
            output_path = os.path.join(root_dir, "yolo_skeleton_top_11m_hampel.txt")
            success = process_skeleton_file(input_path, output_path)
            if success:
                print(f"✅ 已處理：{output_path}")
                processed_files.append(output_path)

    # === 處理報告 ===
    print("\n📋 處理完成列表：")
    for path in processed_files:
        print(f"  ➤ {path}")
    print(f"\n✅ 共處理 {len(processed_files)} 個檔案")




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

    process_all_skeletons(base_path)