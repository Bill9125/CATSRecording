import os
import pandas as pd
import numpy as np

def hampel_filter_for_outliers(series, window_size=7, n_sigmas=3):
    n = len(series)
    k = 1.4826
    half_window = window_size // 2
    outlier_mask = np.zeros(n, dtype=bool)

    for i in range(half_window, n - half_window):
        window = series[i - half_window:i + half_window + 1]

        # 濾除 NaN 後檢查是否為空
        window_valid = window[~np.isnan(window)]
        if len(window_valid) < 3:
            continue  # 跳過無法運算中位數的情況

        median = np.median(window_valid)
        mad = k * np.median(np.abs(window_valid - median))
        if mad == 0:
            continue
        if np.abs(series[i] - median) > n_sigmas * mad:
            outlier_mask[i] = True

    return outlier_mask

import os
import numpy as np
import pandas as pd

# ✅ Hampel 過濾主處理函數
def process_yolo_file(folder_path):  # 傳入資料夾路徑，而不是檔案路徑
    input_filename = "yolo_coordinates.txt"  # 指定目標檔名
    input_path = os.path.join(folder_path, input_filename)  # 組合完整路徑

    print(f"📂 檢查資料夾：{folder_path}")
    print(f"📄 組合後檔案路徑：{input_path}")
    
    if not os.path.exists(input_path):
        print(f"⛔ 找不到檔案：{input_path}")
        return


    try:
        # 讀取資料並轉換為數字型態，非數字轉為 NaN
        df = pd.read_csv(input_path, header=None, names=["frame", "x_center", "y_center", "width", "height"], dtype=str)
        df[["frame", "x_center", "y_center", "width", "height"]] = df[["frame", "x_center", "y_center", "width", "height"]].apply(pd.to_numeric, errors='coerce')

        # Hampel 過濾離群值
        x_outliers = hampel_filter_for_outliers(df["x_center"].values)
        y_outliers = hampel_filter_for_outliers(df["y_center"].values)

        outlier_frames = x_outliers | y_outliers
        df_filtered = df.copy()
        df_filtered.loc[outlier_frames, ["x_center", "y_center", "width", "height"]] = np.nan

        # 儲存新檔
        output_file = os.path.join(folder_path, "yolo_coordinates_hampel.txt")
        df_filtered.to_csv(output_file, header=False, index=False, float_format="%.8f")
        print(f"✅ 已儲存：{output_file}")
    except Exception as e:
        print(f"❌ 處理失敗：{input_path}\n錯誤原因：{e}")

# ✅ 遞迴搜尋所有資料夾
def walk_through_subjects_and_process(base_dir):
    for root, _, _ in os.walk(base_dir):
        process_yolo_file(root)  # 直接將每個資料夾丟進主處理函數



# def process_yolo_file(file_path):
#     try:
#         # 強制轉換成 float，非數字一律轉成 NaN
#         df = pd.read_csv(file_path, header=None, names=["frame", "x_center", "y_center", "width", "height"], dtype=str)
#         df[["frame", "x_center", "y_center", "width", "height"]] = df[["frame", "x_center", "y_center", "width", "height"]].apply(pd.to_numeric, errors='coerce')

#         x_outliers = hampel_filter_for_outliers(df["x_center"].values)
#         y_outliers = hampel_filter_for_outliers(df["y_center"].values)

#         outlier_frames = x_outliers | y_outliers
#         df_filtered = df.copy()
#         df_filtered.loc[outlier_frames, ["x_center", "y_center", "width", "height"]] = np.nan

#         # 儲存為新檔案
#         output_file = os.path.join(os.path.dirname(file_path), "yolo_coordinates_hampel.txt")
#         df_filtered.to_csv(output_file, header=False, index=False, float_format="%.8f")
#         print(f"✅ 已儲存：{output_file}")
#     except Exception as e:
#         print(f"❌ 處理失敗：{file_path}\n錯誤原因：{e}")

# def walk_through_subjects_and_process(base_dir):
#     for root, dirs, files in os.walk(base_dir):
#         for file in files:
#             if file == "yolo_coordinates.txt":
#                 file_path = os.path.join(root, file)
#                 process_yolo_file(file_path)

# if __name__ == "__main__":
#     base_dir = r"D:\benchpress data\舊資料_新分類"
#     walk_through_subjects_and_process(base_dir)


if __name__ == "__main__":
    USE_LATEST = True  # ❗️切換手動指定資料夾還是自動判定最新的

    if USE_LATEST:
        recordings_dir = r"C:/Users/92A27/benchpress/recordings"
        all_folders = [os.path.join(recordings_dir, d) for d in os.listdir(recordings_dir) if os.path.isdir(os.path.join(recordings_dir, d))]
        base_path = max(all_folders, key=os.path.getmtime)
    else:
        base_path = r"E:/DATASET/abc"  # 手動指定

    # 🔍 取得 recordings 下所有子資料夾，並找出最新的
    all_folders = [os.path.join(recordings_dir, d) for d in os.listdir(recordings_dir) if os.path.isdir(os.path.join(recordings_dir, d))]
    if not all_folders:
        raise FileNotFoundError("❌ recordings 資料夾下沒有任何子資料夾")

    latest_folder = max(all_folders, key=os.path.getmtime)  # 依建立時間找最新資料夾
    base_path = os.path.join(latest_folder, '')  # base_path 最後補上斜線

    process_yolo_file(base_path)