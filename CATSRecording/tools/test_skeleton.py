##處理單個影片
# import cv2
# import torch
# from ultralytics import YOLO

# # Load YOLOv8 pose model
# model_path = "D:\\skeleton_train_data\\runs\pose\\train29\\weights\\best.pt"  # 你的訓練好的模型檔案
# model = YOLO(model_path)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)

# # Define the skeleton connections as pairs of keypoint indices
# # Adjust these indices to match your keypoint connections
# skeleton_connections = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5)]  # Example: shoulder to elbow, elbow to wrist

# def main():
#     # Video file paths
#     video_path = 'D:\\skeleton_train_data\\original_vision3.avi'
#     output_path = 'D:\\skeleton_train_data\\output_video_窄握.mp4'

#     # Open the video file
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     # Get video properties
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Initialize VideoWriter for output
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         if not ret:
#             print("End of video or error reading frame.")
#             break

#         # Convert the frame to RGB for YOLO model
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Perform pose detection
#         results = model.predict(source=rgb_frame, conf=0.5)

#         # Process each detected person
#         # Process each detected person
#         annotated_frame = frame.copy()
#         for result in results[0].keypoints:
#             keypoints = result.xy.tolist()  # Convert keypoints to list

#             # Debug print to understand the structure of keypoints
#             print("Keypoints structure:", keypoints)  # Debug line to check structure

#             # Skip if no keypoints detected for this person
#             if not keypoints or not keypoints[0]:  
#                 continue

#             # Draw each keypoint (assumes no visibility value)
#             for keypoint in keypoints[0]:  # Iterate through keypoints without unpacking
#                 if len(keypoint) == 2:  # Check if keypoint has only x and y
#                     x, y = keypoint
#                     cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

#             # Draw skeleton connections (assuming visibility isn't required)
#             for (start_idx, end_idx) in skeleton_connections:
#                 # Ensure that start_idx and end_idx are within the keypoints range
#                 if start_idx < len(keypoints[0]) and end_idx < len(keypoints[0]):
#                     start_point = keypoints[0][start_idx]
#                     end_point = keypoints[0][end_idx]
                    
#                     # Draw line only if both points are valid and not (0, 0)
#                     if start_point != [0, 0] and end_point != [0, 0]:
#                         cv2.line(annotated_frame, (int(start_point[0]), int(start_point[1])),
#                                 (int(end_point[0]), int(end_point[1])), (255, 0, 0), 2)


#         # Write the annotated frame to the output video
#         out.write(annotated_frame)

#         # Display the resulting frame
#         cv2.imshow('Pose Detection', annotated_frame)

#         # Exit the loop when 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release resources
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

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
