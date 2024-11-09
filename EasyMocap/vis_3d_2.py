
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# OpenPose Body25 鍵點的骨架連接（每一對是需要連接的關節）
connections = [
    (0, 1), (1, 2), (17,15), (15,0),(0,16),(16,18),(1,5),
    (1,8),(2,3),(3,4),(5,6),(6,7),(8,9),(8,12),(9,10),(10,11),
    (11,22),(11,24),(22,23),(12,13),(13,14),(14,21),(14,19),(19,20)
]
# 定義資料夾路徑
json_folder = 'C:\\labdata\\MOCAP\\EasyMocap-master\\5_output\\project\\keypoints3d'



# 讀取資料夾中的所有 JSON 檔案
json_files = sorted([f for f in os.listdir(json_folder) if f.endswith('.json')])

# 初始化 keypoints 資料
keypoints_data = []

# 從每個 JSON 檔案中提取 3D keypoints
for json_file in json_files:
    with open(os.path.join(json_folder, json_file), 'r') as f:
        data = json.load(f)
        keypoints = np.array([kp[:3] for kp in data[0]['keypoints3d']])  # 只取 x, y, z 座標
        keypoints_data.append(keypoints)

# 初始化 3D 圖形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 設定播放暫停狀態
is_paused = False

# 定義一個函數來更新圖形
def update_graph(num):
    ax.clear()

    # 提取當前時間點的 keypoints
    keypoints = keypoints_data[num]

    # 在 3D 圖中繪製 keypoints
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c='r', marker='o')

    # 繪製骨架連接
    for connection in connections:
        p1, p2 = connection
        
        # 檢查點 p1 和 p2 是否為 (0, 0, 0)，如果是，跳過連接
        if np.all(keypoints[p1] == [0, 0, 0]) or np.all(keypoints[p2] == [0, 0, 0]):
            continue
        
        # 如果兩個點都不是 (0, 0, 0)，連接它們
        x_vals = [keypoints[p1, 0], keypoints[p2, 0]]
        y_vals = [keypoints[p1, 1], keypoints[p2, 1]]
        z_vals = [keypoints[p1, 2], keypoints[p2, 2]]
        ax.plot(x_vals, y_vals, z_vals, c='b')

    # 設置坐標範圍
    ax.set_xlim([-1, 1])  # 根據你的數據範圍調整
    ax.set_ylim([-1, 1])
    ax.set_zlim([-0.5, 2])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# 定義動畫
ani = FuncAnimation(fig, update_graph, frames=len(keypoints_data), interval=50)

# 滑鼠控制旋轉功能
def on_click(event):
    global is_paused
    if is_paused:
        ani.event_source.start()
    else:
        ani.event_source.stop()
    is_paused = not is_paused

# 連接點擊事件來暫停和播放
fig.canvas.mpl_connect('button_press_event', on_click)

# 顯示圖形
plt.show()
