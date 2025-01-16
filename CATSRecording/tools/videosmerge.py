import cv2
import numpy as np
import os

def merge_videos_vertically(video1_path, video2_path, output_path):
    """
    使用 OpenCV 将两部视频上下合并为一个视频，并将第二个视频的宽度调整为与第一个视频一致。

    :param video1_path: 第一个视频的路径。
    :param video2_path: 第二个视频的路径。
    :param output_path: 输出视频的路径。
    """
    # 打开两个视频文件
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # 获取视频的帧率和大小
    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    fps = min(fps1, fps2)  # 选取两个视频的最低帧率

    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 确保第二个视频的宽度与第一个视频一致
    if width1 != width2:
        cap2_width = width1  # 将第二个视频的宽度调整为第一个视频的宽度
    else:
        cap2_width = width2

    # 合并后的视频大小
    output_width = width1
    output_height = height1 + height2  # 高度相加

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用 mp4 格式
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # 如果任一视频播放完毕，停止
        if not ret1 or not ret2:
            break

        # 如果第二个视频宽度不等于第一个视频宽度，进行调整
        if cap2_width != width2:
            frame2 = cv2.resize(frame2, (cap2_width, height2))

        # 垂直拼接帧
        combined_frame = np.vstack((frame1, frame2))

        # 写入合并后的帧
        out.write(combined_frame)

    # 释放资源
    cap1.release()
    cap2.release()
    out.release()

    print(f"影片已保存到: {output_path}")

dir = 'C:/jinglun/CATSRecording/data/recording_Deadlift/Subject65/recording_20241226_161743'

for target in ['vision2', 'vision3', 'vision4', 'vision5']:
    vision = os.path.join(dir, f'{target}.avi')
    data = os.path.join(dir, f'{target}_data.mp4')
    output = os.path.join(dir, f'{target}_merged.mp4')  
    merge_videos_vertically(vision, data, output)
