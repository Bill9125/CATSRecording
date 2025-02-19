###處理所有影片
import os
import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 pose model
model_path = "D:\\skeleton_train_data\\runs\\pose\\train29\\weights\\best.pt"  # 你的訓練好的模型檔案
model = YOLO(model_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Define the skeleton connections as pairs of keypoint indices
skeleton_connections = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5)]  # 例: 肩膀到手肘，手肘到手腕

def process_video(video_path, output_video_path, output_txt_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    with open(output_txt_path, 'w') as f:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(source=rgb_frame, conf=0.5)

            annotated_frame = frame.copy()
            frame_data = []

            for result in results[0].keypoints:
                keypoints = result.xy.tolist()

                if not keypoints or not keypoints[0]:  
                    continue

                keypoint_list = []
                for keypoint in keypoints[0]:  
                    if len(keypoint) == 2:  
                        x, y = keypoint
                        keypoint_list.append((x, y))
                        cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                for (start_idx, end_idx) in skeleton_connections:
                    if start_idx < len(keypoint_list) and end_idx < len(keypoint_list):
                        start_point = keypoint_list[start_idx]
                        end_point = keypoint_list[end_idx]
                        
                        if start_point != (0, 0) and end_point != (0, 0):
                            cv2.line(annotated_frame, (int(start_point[0]), int(start_point[1])),
                                     (int(end_point[0]), int(end_point[1])), (255, 0, 0), 2)

                frame_data.append(keypoint_list)

            out.write(annotated_frame)
            f.write(f"Frame {frame_idx}: {frame_data}\n")
            frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def process_all_videos_in_directory(root_directory):
    for subdir, _, files in os.walk(root_directory):
        video_path = os.path.join(subdir, "original_vision3.avi")
        if os.path.exists(video_path):
            output_video_path = os.path.join(subdir, "vision3_new_skeleton.avi")
            output_txt_path = os.path.join(subdir, "vision3_new_skeleton.txt")
            print(f"Processing: {video_path}")
            process_video(video_path, output_video_path, output_txt_path)

if __name__ == "__main__":
    root_directory = "D:\\yolo_mediapipe\\所有受試者資料\\資料整理\\肩膀"  # 設定要遍歷的根資料夾
    process_all_videos_in_directory(root_directory)
