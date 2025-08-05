import os
import pandas as pd
import cv2

# 特徵檔案名稱清單
target_files = [
    "bar_x.txt",
    "bar_y.txt",
    "bar_y_div_x.txt",
    "left_elbow_angle.txt",
    "left_shoulder_angle.txt",
    "left_torso_arm_angle_11m.txt",
    "left_wrist_to_shoulderline_11m.txt",
    "right_elbow_angle.txt",
    "right_shoulder_angle.txt",
    "right_torso_arm_angle_11m.txt", 
    "right_wrist_to_shoulderline_11m.txt",
    "left_shoulder_y.txt",
    "right_shoulder_y.txt"

]



# 切割區間讀取函式
def read_cut_file(path):
    intervals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().replace(" ", "").split("-")
            if len(parts) == 2:
                try:
                    intervals.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    print(f"⚠️ 無法轉數字: {line}")
    return intervals

# 主程式：處理單一資料夾
def process_single_folder(folder_path, category_name, subject_id, output_root, cut_prefix="cut4_"):
    print(f"📁 處理: {category_name}/{subject_id}")

    cut_files = [f for f in os.listdir(folder_path) if f.startswith(cut_prefix) and f.endswith(".txt")]
    if not cut_files:
        print(f"⚠️ 找不到任何 cut 檔案於 {folder_path}")
        return

    for cut_file in cut_files:
        intervals = read_cut_file(os.path.join(folder_path, cut_file))
        cut_name = os.path.splitext(cut_file)[0]

        for file in target_files:
            file_path = os.path.join(folder_path, file)
            if not os.path.exists(file_path):
                print(f"❌ 檔案不存在: {file_path}")
                continue

            data = pd.read_csv(file_path, header=None)
            for idx, (start, end) in enumerate(intervals):
                if end >= len(data):
                    print(f"⚠️ 區間超出範圍: {file} ({start}-{end})")
                    continue

                rep_dir = os.path.join(output_root, category_name, subject_id, cut_name, f"rep_{idx+1}")
                os.makedirs(rep_dir, exist_ok=True)

                out_path = os.path.join(rep_dir, file)
                if os.path.exists(out_path):
                    continue  # ✅ 已存在就跳過

                sliced = data.iloc[start-1:end]
                sliced.to_csv(out_path, index=False, header=False)

    print("✅ 處理完成")
# 影片裁切
def cut_video_to_intervals(video_path, intervals, output_dir, video_prefix):
    if not os.path.exists(video_path):
        print(f"❌ 找不到影片: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_dict = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_dict[frame_idx] = frame
        frame_idx += 1
    cap.release()

    for idx, (start, end) in enumerate(intervals):
        out_path = os.path.join(output_dir, f"{video_prefix}_{idx+1}.avi")
        if os.path.exists(out_path):
            continue  # ✅ 已存在影片就跳過

        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
        for i in range(start - 1, end):
            if i in frame_dict:
                out.write(frame_dict[i])
        out.release()
    print(f"🎬 {video_prefix} 裁切完成，共 {len(intervals)} 段")

# 加入影片處理的總整合
# def process_all_cut4_with_video(folder_path, category_name, subject_id, output_root):
#     cut_files = [f for f in os.listdir(folder_path) if f.startswith("cut4_") and f.endswith(".txt")]
#     if not cut_files:
#         print(f"⚠️ 找不到任何 cut4 檔案於 {folder_path}")
#         return

#     for cut_file in cut_files:
#         print(f"\n📁 處理: {category_name}/{subject_id} ({cut_file})")
#         cut_path = os.path.join(folder_path, cut_file)
#         intervals = read_cut_file(cut_path)
#         cut_name = os.path.splitext(cut_file)[0]
#         output_dir = os.path.join(output_root, category_name, subject_id, cut_name)
#         os.makedirs(output_dir, exist_ok=True)

#         # ✅ 特徵裁切（每個檔案只在所有 rep_* 檔案都缺少時才會處理）
#         for file in target_files:
#             file_path = os.path.join(folder_path, file)
#             if not os.path.exists(file_path):
#                 print(f"❌ 檔案不存在: {file_path}")
#                 continue

#             # 檢查是否所有 rep 檔案都已存在
#             all_exist = True
#             for idx in range(len(intervals)):
#                 rep_out_path = os.path.join(output_dir, f"{file[:-4]}_{idx+1}.txt")
#                 if not os.path.exists(rep_out_path):
#                     all_exist = False
#                     break
#             if all_exist:
#                 continue  # ✅ 全部 rep 存在就跳過這個特徵檔案

#             # 開始處理這份特徵檔案
#             data = pd.read_csv(file_path, header=None)
#             for idx, (start, end) in enumerate(intervals):
#                 if end >= len(data):
#                     print(f"⚠️ 區間超出範圍: {file} ({start}-{end})")
#                     continue

#                 sliced = data.iloc[start-1:end]
#                 out_path = os.path.join(output_dir, f"{file[:-4]}_{idx+1}.txt")
#                 sliced.to_csv(out_path, index=False, header=False)

#         # ✅ 裁切影片（每個視角只在所有切片都缺少時才會處理）
#         for vision_idx in [1, 2, 3]:
#             video_path = os.path.join(folder_path, f"original_vision{vision_idx}.avi")

#             # 檢查是否所有影片片段都存在
#             all_exist = all(
#                 os.path.exists(os.path.join(output_dir, f"vision{vision_idx}_{i+1}.avi"))
#                 for i in range(len(intervals))
#             )
#             if all_exist:
#                 print(f"⏭️ 所有 vision{vision_idx} 的影片已存在，跳過")
#                 continue

#             # 若有缺少，就裁切
#             cut_video_to_intervals(video_path, intervals, output_dir, f"vision{vision_idx}")

#     print(f"\n✅ 全部 cut4 檔案處理完成: {category_name}/{subject_id}")


# def process_entire_dataset(dataset_root, output_root):
#     for category in os.listdir(dataset_root):
#         category_path = os.path.join(dataset_root, category)
#         if not os.path.isdir(category_path):
#             continue

#         for subject_id in os.listdir(category_path):
#             subject_path = os.path.join(category_path, subject_id)
#             if not os.path.isdir(subject_path):
#                 continue

#             process_all_cut4_with_video(subject_path, category, subject_id, output_root)


def process_all_cut4_with_video(folder_path):
    output_root = os.path.join(folder_path, "feature")  # 自動建立 feature 子資料夾
    os.makedirs(output_root, exist_ok=True)  # 若 feature 資料夾不存在則建立

    cut_files = [f for f in os.listdir(folder_path) if f.startswith("cut4_") and f.endswith(".txt")]
    if not cut_files:
        print(f"⚠️ 找不到任何 cut4 檔案於 {folder_path}")
        return

    for cut_file in cut_files:
        print(f"\n📁 處理: {cut_file}")
        cut_path = os.path.join(folder_path, cut_file)
        intervals = read_cut_file(cut_path)
        cut_name = os.path.splitext(cut_file)[0]
        output_dir = os.path.join(output_root, cut_name)
        os.makedirs(output_dir, exist_ok=True)

        # ✅ 特徵裁切
        for file in target_files:
            file_path = os.path.join(folder_path, file)
            if not os.path.exists(file_path):
                print(f"❌ 檔案不存在: {file_path}")
                continue

            all_exist = all(
                os.path.exists(os.path.join(output_dir, f"{file[:-4]}_{idx+1}.txt"))
                for idx in range(len(intervals))
            )
            if all_exist:
                continue  # ✅ 所有檔案都存在就跳過

            data = pd.read_csv(file_path, header=None)
            for idx, (start, end) in enumerate(intervals):
                if end >= len(data):
                    print(f"⚠️ 區間超出範圍: {file} ({start}-{end})")
                    continue

                sliced = data.iloc[start-1:end]
                out_path = os.path.join(output_dir, f"{file[:-4]}_{idx+1}.txt")
                sliced.to_csv(out_path, index=False, header=False)

        # ✅ 裁切影片
        for vision_idx in [1, 2, 3]:
            video_path = os.path.join(folder_path, f"original_vision{vision_idx}.avi")
            all_exist = all(
                os.path.exists(os.path.join(output_dir, f"vision{vision_idx}_{i+1}.avi"))
                for i in range(len(intervals))
            )
            if all_exist:
                print(f"⏭️ 所有 vision{vision_idx} 的影片已存在，跳過")
                continue

            cut_video_to_intervals(video_path, intervals, output_dir, f"vision{vision_idx}")

    print(f"\n✅ 全部 cut4 檔案處理完成: {folder_path}")



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

    process_all_cut4_with_video(base_path)
