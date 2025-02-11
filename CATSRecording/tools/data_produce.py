import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import argparse
import cv2

# 函式：讀取骨架數據的 txt 檔案
def read_skeleton_data(filename):
    data = {}
    with open(filename, 'r') as file:
        for line in file:
            frame, joint, x, y = map(int, line.strip().split(','))
            if frame not in data:
                data[frame] = {}
            data[frame][joint] = (x, y)
    return data

# 函式：計算角度與長度
def calculate_angles_and_length(data):
    frames = sorted(data.keys())
    left_knee_angles, left_hip_angles, body_lengths = [], [], []

    for frame in frames:
        joints = data[frame]
        if all(k in joints for k in [12, 14, 16, 6, 10]):
            left_knee_angles.append(calculate_angle(joints[12], joints[14], joints[16]))
            left_hip_angles.append(calculate_angle(joints[6], joints[12], joints[16]))
            body_lengths.append(calculate_distance(joints[6], joints[12]))
        else:
            left_knee_angles.append(0)
            left_hip_angles.append(0)
            body_lengths.append(0)

    return frames, left_knee_angles, left_hip_angles, body_lengths

# 函式：計算角度
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

# 函式：計算距離
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# 函式：讀取槓端數據
def read_barbell_positions(filename):
    frames, x_coords, y_coords = [], [], []
    with open(filename, 'r') as file:
        for line in file:
            frame, x, y = map(float, line.strip().split(',')[:3])
            frames.append(int(frame))
            x_coords.append(x)
            y_coords.append(y)
    return frames, x_coords, y_coords
    
# 函式：生成單個動畫並存成影片
def generate_individual_animation(title, y_label, y_data, y_limit, output_file):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(title)
    ax.set_xlabel('Frame')
    ax.set_ylabel(y_label)
    ax.set_ylim(y_limit)
    ax.grid()

    line, = ax.plot([], [], color='blue')

    def update(frame_index):
        # 確保不會出現 x 軸為 0 的情況
        if frame_index < 1:
            frame_index = 1
        start_frame = max(0, frame_index - 500)  # 只顯示最近 200 幀
        end_frame = frame_index
        line.set_data(skeleton_frames[start_frame:end_frame], y_data[start_frame:end_frame])
        ax.set_xlim(start_frame, end_frame)
        return line,

    ani = FuncAnimation(fig, update, frames=len(skeleton_frames), interval=10, blit=True)

    # 保存影片
    try:
        ani.save(output_file, fps=30, writer='ffmpeg')
        print(f"Saved animation to {output_file}")
    except FileNotFoundError:
        print(f"FFmpeg is not available on your system. Unable to save {output_file}.")
    except Exception as e:
        print(f"Error saving {output_file}: {e}")
    finally:
        plt.close(fig)

parser = argparse.ArgumentParser()
parser.add_argument('dir',type=str)
args = parser.parse_args()
dir = args.dir
skeleton_file_path = os.path.join(dir ,'interpolated_mediapipe_landmarks_1.txt')
barbell_file_path = os.path.join(dir , 'yolo_coordinates_interpolated.txt')

skeleton_data = read_skeleton_data(skeleton_file_path)
skeleton_frames, left_knee_angles, left_hip_angles, body_lengths = calculate_angles_and_length(skeleton_data)
knee_to_hip_ratios = [
    left_knee / left_hip if left_hip != 0 else 0
    for left_knee, left_hip in zip(left_knee_angles, left_hip_angles)
]

# 生成五個動畫並存成影片
generate_individual_animation(
    title='Left Knee Angle Over Time',
    y_label='Angle (degrees)',
    y_data=left_knee_angles,
    y_limit=(100, 200),
    output_file = os.path.join(dir, 'vision2_data.mp4')
)

generate_individual_animation(
    title='Left Hip Angle Over Time',
    y_label='Angle (degrees)',
    y_data=left_hip_angles,
    y_limit=(80, 200),
    output_file = os.path.join(dir, 'vision3_data.mp4')
)

generate_individual_animation(
    title='Knee-to-Hip Angle Ratio Over Time',
    y_label='Ratio (Knee / Hip)',
    y_data=knee_to_hip_ratios,
    y_limit=(0, 2),  # 假設比例範圍合理為 0-2
    output_file = os.path.join(dir, 'vision4_data.mp4')
)

generate_individual_animation(
    title='Body Length Over Time',
    y_label='Length',
    y_data=body_lengths,
    y_limit=(80, 160),
    output_file = os.path.join(dir, 'vision5_data.mp4')
)
