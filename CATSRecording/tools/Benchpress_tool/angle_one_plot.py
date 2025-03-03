import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json

def safe_arccos(value):
    return np.degrees(np.arccos(np.clip(value, -1.0, 1.0)))

def read_bar_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    frames, x, y = data[:, 0], data[:, 1], data[:, 2]
    return frames, x, y

def read_mediapipe_data(file_path, landmarks=[(24, 12, 14), (23, 11, 13)]):
    data = np.loadtxt(file_path, delimiter=',')
    frames = np.unique(data[:, 0])
    angles = {landmark: [] for landmark in landmarks}
    
    for frame in frames:
        frame_data = data[data[:, 0] == frame]
        points = {int(row[1]): row[2:5] for row in frame_data}
        for landmark in landmarks:
            if all(l in points for l in landmark):
                vec1 = np.array(points[landmark[0]]) - np.array(points[landmark[1]])
                vec2 = np.array(points[landmark[2]]) - np.array(points[landmark[1]])
                angle = safe_arccos(
                    np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                )
                angles[landmark].append(angle)
            else:
                angles[landmark].append(None)
    
    return frames, angles

def read_skeleton_data(file_path):
    frames = []
    angles = {"right_shoulder": [], "left_shoulder": [], "right_elbow": [], "left_elbow": []}

    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r'Frame (\d+):\s*\[\[(.*?)\]\]', line)
            if not match:
                print(f"❌ Format mismatch: {line.strip()}")
                continue

            frame = int(match.group(1))
            points_str = match.group(2)

            try:
                # 轉換 `()` 為 `[]`，以便 `eval()` 解析
                points_str = points_str.replace("(", "[").replace(")", "]")
                points = eval(f"[{points_str}]")  # 直接解析為 list

                if not isinstance(points, list) or len(points) < 6:
                    print(f"⚠️ Frame {frame} has {len(points)} points, proceeding with calculation.")
                
                frames.append(frame)
                points = np.array(points)

                print(f"✅ Frame {frame} parsed successfully: {points.shape}")

                # 取出骨架關鍵點
                s0, s1, e0, e1, w0, w1 = points[:6]


                ###計算每個向量之間的小角度
                def calculate_angle(v1, v2):
                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)
                    
                    if norm_v1 == 0 or norm_v2 == 0:
                        return None  # 避免除零錯誤
                    
                    # 確保夾角範圍在 0°~180°
                    angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)))
                    
                    return angle  # 確保角度只在 0°~180°


                # 計算角度
                right_shoulder_angle = 180 - calculate_angle(s1 - s0, e1 - s1)
                print("right_shoulder_angle" ,right_shoulder_angle)
                left_shoulder_angle = 180 - calculate_angle(s0 - s1, e0 - s0)
                print("left_shoulder_angle" ,left_shoulder_angle)
                right_elbow_angle = 180 - calculate_angle(e1 - s1, w1 - e1)
                print("right_elbow_angle" ,right_elbow_angle)
                left_elbow_angle = 180 - calculate_angle(e0 - s0, w0 - e0)
                print("left_elbow_angle" ,left_elbow_angle)

                # **確保每一個 Frame 都存入值，即使是 None**
                angles["right_shoulder"].append(right_shoulder_angle if right_shoulder_angle is not None else 0)
                angles["left_shoulder"].append(left_shoulder_angle if left_shoulder_angle is not None else 0)
                angles["right_elbow"].append(right_elbow_angle if right_elbow_angle is not None else 0)
                angles["left_elbow"].append(left_elbow_angle if left_elbow_angle is not None else 0)

            except Exception as e:
                print(f"❌ Skipping frame {frame} due to parsing error: {e}")
                # **如果有解析錯誤，確保也存入 0 來保持長度一致**
                angles["right_shoulder"].append(0)
                angles["left_shoulder"].append(0)
                angles["right_elbow"].append(0)
                angles["left_elbow"].append(0)

    print(f"✅ Total frames processed: {len(frames)}")
    return frames, angles


def plot_data():
    base_dir = 'C:/jinglun/CATSRecording/data/recording_Benchpress/1'  # 設定你的資料夾路徑
    
    bar_file = os.path.join(base_dir, 'new_bar_interpolate.txt')
    mediapipe_file = os.path.join(base_dir, 'interpolated_mediapipe_landmarks_1.txt')
    skeleton_file = os.path.join(base_dir, 'vision3_new_skeleton_interpolated.txt')
    
    frames_bar, x_bar, y_bar = read_bar_data(bar_file)
    frames_mediapipe, angles_mediapipe = read_mediapipe_data(mediapipe_file)
    frames_skeleton, angles_skeleton = read_skeleton_data(skeleton_file)
    
    plt.figure(figsize=(15, 10))
    
    # 槓端軌跡
    ax1 = plt.subplot(3, 1, 1)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('X-Y Position')
    ax1.plot(frames_bar, x_bar, 'b-', label='X Center')
    ax1.plot(frames_bar, y_bar, 'r-', label='Y Center')
    ax1.legend()
    ax1.set_title('Barbell X-Y Movement')
    ax1.invert_yaxis()  # 反轉 y 軸 
    
    # 骨架1夾角
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(frames_mediapipe, angles_mediapipe.get((24, 12, 14), []), label='Right Arm Angle')
    ax2.plot(frames_mediapipe, angles_mediapipe.get((23, 11, 13), []), label='Left Arm Angle')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Angle (Degrees)')
    ax2.set_title('Arm Angles from Mediapipe')
    ax2.legend()
    
    # 骨架2關節角度
    ax3 = plt.subplot(3, 1, 3)
    for key, values in angles_skeleton.items():
        if len(values) == len(frames_skeleton):  # 確保長度相符
            ax3.plot(frames_skeleton, values, label=key.replace('_', ' ').title())
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Angle (Degrees)')
    ax3.set_title('Joint Angles from YOLO Pose')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_data()
