import os
import cv2

# 定義執行文件夾
input_folder = "./cam_group_2/3_input/videos"

def avi_2_mp4(folder):
    # 確保輸出文件夾存在
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 遍歷輸入文件夾中的所有 .avi 文件
    for filename in os.listdir(folder):
        if filename.endswith(".avi"):
            avi_file = os.path.join(folder, filename)
            
            # 獲取輸出文件的文件名，並將擴展名更改為 .mp4
            mp4_filename = filename.replace(".avi", ".mp4")
            output_file = os.path.join(folder, mp4_filename)

            print(f"正在將 {avi_file} 轉換為 {output_file}")

            # 打開 .avi 文件
            cap = cv2.VideoCapture(avi_file)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 定義 MP4 編碼器並創建輸出文件
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

            # 逐幀讀取和寫入
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            # 釋放資源
            cap.release()
            out.release()

            print(f"完成轉換 {output_file}\n")

    for filename in os.listdir(folder):
        if filename.endswith(".avi"):
            avi_file = os.path.join(folder, filename)
            os.remove(avi_file)
    print('avi file Removed')

avi_2_mp4(input_folder)
