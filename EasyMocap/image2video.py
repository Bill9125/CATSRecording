import cv2
import os
dir = './cam_group_3/0_output/'
for target in ['detec', 'repro', 'repro_smpl', 'smpl']: 
    # 設定輸入文件夾和輸出影片檔
    input_folder = dir + target  # 包含影像的文件夾
    output_file = dir + target + '.mp4'  # 輸出的影片檔案名

    # 取得圖片檔名列表並排序
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")])

    # 檢查文件夾內是否有圖片
    if not image_files:
        print("文件夾中沒有找到任何 JPG 文件。")
        exit()

    # 讀取第一幀圖片以獲取影片的寬和高
    first_frame = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, _ = first_frame.shape

    # 定義影片的編碼方式和每秒幀數
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 格式
    fps = 30  # 每秒幀數，可以根據需要調整
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 遍歷所有圖片並將它們寫入影片
    for image_file in image_files:
        img_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"無法讀取圖片: {img_path}")
            continue

        out.write(frame)

    # 釋放資源
    out.release()
    print(f"影片已成功儲存至 {output_file}")
