import time, os, cv2
from PyQt5 import QtCore, QtGui

def deadlift_bar_loop(i, frame, label, save_sig, recording_sig, folder,
                      start_time, frame_count, fps, out, model, txt_file, frame_count_for_detect, barrier):
    # fps 計算
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # 錄影開始
    if recording_sig:
        if out is None:  # 初始化 VideoWriter
            file = os.path.join(folder, f'vision{i + 1}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_size = (frame.shape[1], frame.shape[0])  # 幀大小 (width, height)
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)
            print(f"Initialized VideoWriter for camera {i + 1}")
        out.write(frame)

        if txt_file is None:
            txt_file_path = os.path.join(folder, 'yolo_coordinates.txt')
            txt_file = open(txt_file_path, "w")  # ✅ 錄影開始時開啟檔案
            frame_count_for_detect = 0  # ✅ 只在錄影開始時歸零
            print(f"Started writing data to {txt_file_path}")

    # frame 處理
    results = model(source=frame, imgsz=320, conf=0.5, verbose=False)
    boxes = results[0].boxes
    detected = False
    for result in results:
        frame = result.plot()
    
    # write result
    if recording_sig or txt_file is not None:
        for box in boxes.xywh:
            detected = True
            x_center, y_center, width, height = box
            frame_count_for_detect += 1
            txt_file.write(f"{frame_count_for_detect},{x_center},{y_center},{width},{height}\n")
            
        if not detected:
            frame_count_for_detect += 1
            txt_file.write(f"{frame_count_for_detect},no detection\n")

    barrier.wait()
    if not recording_sig:
        frame_count_for_detect = 0
        # 錄影結束
        if save_sig and out is not None:
            out.release()
            print(f"Released VideoWriter for camera {i + 1}")
            save_sig = False
        out = None
        if txt_file is not None:
            txt_file.close()
            txt_file = None  # ✅ 確保 `txt_file` 被正確關閉
            print(f"Closed txt_file for camera {i + 1}")

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out, frame_count_for_detect, save_sig, txt_file

def deadlift_bone_loop(i, frame, label, save_sig, recording_sig, folder,
                       start_time, frame_count, fps, out, model, txt_file, frame_count_for_detect, skeleton_connections, barrier):
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # ✅ 錄影開始
    if recording_sig:
        if out is None:
            file = os.path.join(folder, f'vision{i + 1}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_size = (frame.shape[1], frame.shape[0])
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)
            print(f"Initialized VideoWriter for camera {i + 1}")
        out.write(frame)

        if txt_file is None:
            txt_file_path = os.path.join(folder, 'mediapipe_landmarks.txt')
            txt_file = open(txt_file_path, "w")
            frame_count_for_detect = 0
            print(f"Started writing data to {txt_file_path}")

    # ✅ YOLO 偵測骨架
    results = list(model(source=frame, stream=True, verbose=False))
    frame_count_for_detect += 1

    if results and results[0].keypoints:  # ✅ 確保有偵測到人
        r2 = results[0]  # ✅ 只取第一個偵測結果
        keypoints = r2.keypoints
        kpts = keypoints[0]  # ✅ 只取第一個人的骨架點
        keypoints_xy = kpts.xy  # shape: (1, 17, 2) -> 17 個關鍵點

        # ✅ 過濾無效骨架點 (0,0)
        kp_coords = []
        frame_data = []  # 存放該幀的骨架點
        for idx, kp in enumerate(keypoints_xy[0]):
            x_kp, y_kp = int(kp[0].item()), int(kp[1].item())

            # ✅ 若骨架點為 (0,0)，則標記為 None（不畫）
            if x_kp == 0 and y_kp == 0:
                kp_coords.append(None)
            else:
                kp_coords.append((x_kp, y_kp))
                cv2.circle(frame, (x_kp, y_kp), 5, (0, 255, 0), cv2.FILLED)
            
            frame_data.append(f"{frame_count_for_detect},{idx},{x_kp},{y_kp}")

        # ✅ 繪製骨架連線，若其中一個點為 None，則不畫線
        for start_idx, end_idx in skeleton_connections:
            if start_idx < len(kp_coords) and end_idx < len(kp_coords):
                if kp_coords[start_idx] is None or kp_coords[end_idx] is None:
                    continue
                cv2.line(frame, kp_coords[start_idx], kp_coords[end_idx], (0, 255, 255), 2)

        # ✅ **將骨架點資訊寫入 `txt_file`**
        if txt_file is not None:
            txt_file.write("\n".join(frame_data) + "\n")

    else:
        # ❌ **沒有偵測到人，寫入 "no detection"**
        if txt_file is not None:
            txt_file.write(f"{frame_count_for_detect},no detection\n")

    # ✅ **錄影完全結束後才關閉 `txt_file`**
    barrier.wait()
    if not recording_sig:
        if txt_file is not None:
            txt_file.close()
            txt_file = None
            print(f"Closed txt_file for camera {i + 1}")

        if out is not None:
            out.release()
            out = None
            print(f"Released VideoWriter for camera {i + 1}")
        save_sig = False

    # ✅ 繪製 FPS
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out, frame_count_for_detect, save_sig, txt_file


def deadlift_general_loop(i, frame, label, save_sig, recording_sig, folder,
                          start_time, frame_count, fps, out, barrier):
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # 錄影開始
    if recording_sig:
        if out is None:  # 初始化 VideoWriter
            file = os.path.join(folder, f'vision{i + 1}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_size = (frame.shape[1], frame.shape[0])  # 幀大小 (width, height)
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)
            print(f"Initialized VideoWriter for camera {i + 1}")
        out.write(frame)
    
    # 錄影結束    
    barrier.wait()
    if not recording_sig:
        if out is not None:
            out.release()
            print(f"Released VideoWriter for camera {i + 1}")
        out = None
    
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out, save_sig

def squat_bar_loop(i, frame, label, save_sig, recording_sig, folder,
                      start_time, frame_count, fps, out, model, txt_file, frame_count_for_detect, barrier):
    # fps 計算
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # 錄影開始
    if recording_sig:
        if out is None:  # 初始化 VideoWriter
            file = os.path.join(folder, f'vision{i + 1}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_size = (frame.shape[1], frame.shape[0])  # 幀大小 (width, height)
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)
            print(f"Initialized VideoWriter for camera {i + 1}")
        out.write(frame)

        if txt_file is None:
            txt_file_path = os.path.join(folder, 'yolo_coordinates.txt')
            txt_file = open(txt_file_path, "w")  # ✅ 錄影開始時開啟檔案
            frame_count_for_detect = 0  # ✅ 只在錄影開始時歸零
            print(f"Started writing data to {txt_file_path}")

    # frame 處理
    results = model(source=frame, imgsz=320, conf=0.5, verbose=False)
    boxes = results[0].boxes
    detected = False
    for result in results:
        frame = result.plot()
    
    # write result
    if recording_sig or txt_file is not None:
        for box in boxes.xywh:
            detected = True
            x_center, y_center, width, height = box
            frame_count_for_detect += 1
            txt_file.write(f"{frame_count_for_detect},{x_center},{y_center},{width},{height}\n")
            
        if not detected:
            frame_count_for_detect += 1
            txt_file.write(f"{frame_count_for_detect},no detection\n")

    barrier.wait()
    if not recording_sig:
        frame_count_for_detect = 0
        # 錄影結束
        if save_sig and out is not None:
            out.release()
            print(f"Released VideoWriter for camera {i + 1}")
            save_sig = False
        out = None
        if txt_file is not None:
            txt_file.close()
            txt_file = None  # ✅ 確保 `txt_file` 被正確關閉
            print(f"Closed txt_file for camera {i + 1}")

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out, frame_count_for_detect, save_sig, txt_file

def squat_bone_loop(i, frame, label, save_sig, recording_sig, folder,
                       start_time, frame_count, fps, out, model, txt_file, frame_count_for_detect, skeleton_connections, barrier):
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # ✅ 錄影開始
    if recording_sig:
        if out is None:
            file = os.path.join(folder, f'vision{i + 1}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_size = (frame.shape[1], frame.shape[0])
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)
            print(f"Initialized VideoWriter for camera {i + 1}")
        out.write(frame)

        if txt_file is None:
            txt_file_path = os.path.join(folder, 'mediapipe_landmarks.txt')
            txt_file = open(txt_file_path, "w")
            frame_count_for_detect = 0
            print(f"Started writing data to {txt_file_path}")

    # ✅ YOLO 偵測骨架
    results = list(model(source=frame, stream=True, verbose=False))
    frame_count_for_detect += 1

    if results and results[0].keypoints:  # ✅ 確保有偵測到人
        r2 = results[0]  # ✅ 只取第一個偵測結果
        keypoints = r2.keypoints
        kpts = keypoints[0]  # ✅ 只取第一個人的骨架點
        keypoints_xy = kpts.xy  # shape: (1, 17, 2) -> 17 個關鍵點

        # ✅ 過濾無效骨架點 (0,0)
        kp_coords = []
        frame_data = []  # 存放該幀的骨架點
        for idx, kp in enumerate(keypoints_xy[0]):
            x_kp, y_kp = int(kp[0].item()), int(kp[1].item())

            # ✅ 若骨架點為 (0,0)，則標記為 None（不畫）
            if x_kp == 0 and y_kp == 0:
                kp_coords.append(None)
            else:
                kp_coords.append((x_kp, y_kp))
                cv2.circle(frame, (x_kp, y_kp), 5, (0, 255, 0), cv2.FILLED)
            
            frame_data.append(f"{frame_count_for_detect},{idx},{x_kp},{y_kp}")

        # ✅ 繪製骨架連線，若其中一個點為 None，則不畫線
        for start_idx, end_idx in skeleton_connections:
            if start_idx < len(kp_coords) and end_idx < len(kp_coords):
                if kp_coords[start_idx] is None or kp_coords[end_idx] is None:
                    continue
                cv2.line(frame, kp_coords[start_idx], kp_coords[end_idx], (0, 255, 255), 2)

        # ✅ **將骨架點資訊寫入 `txt_file`**
        if txt_file is not None:
            txt_file.write("\n".join(frame_data) + "\n")

    else:
        # ❌ **沒有偵測到人，寫入 "no detection"**
        if txt_file is not None:
            txt_file.write(f"{frame_count_for_detect},no detection\n")

    # ✅ **錄影完全結束後才關閉 `txt_file`**
    barrier.wait()
    if not recording_sig:
        if txt_file is not None:
            txt_file.close()
            txt_file = None
            print(f"Closed txt_file for camera {i + 1}")

        if out is not None:
            out.release()
            out = None
            print(f"Released VideoWriter for camera {i + 1}")
        save_sig = False

    # ✅ 繪製 FPS
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out, frame_count_for_detect, save_sig, txt_file


def squat_general_loop(i, frame, label, save_sig, recording_sig, folder,
                          start_time, frame_count, fps, out, barrier):
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # 錄影開始
    if recording_sig:
        if out is None:  # 初始化 VideoWriter
            file = os.path.join(folder, f'vision{i + 1}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_size = (frame.shape[1], frame.shape[0])  # 幀大小 (width, height)
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)
            print(f"Initialized VideoWriter for camera {i + 1}")
        out.write(frame)
    
    # 錄影結束    
    barrier.wait()
    if not recording_sig:
        if out is not None:
            out.release()
            print(f"Released VideoWriter for camera {i + 1}")
        out = None
    
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out, save_sig


# ====== 緩衝常數（可依需求調整）======
BODY_BUF_FRAMES = 20                     # 人體偵測命中/未命中緩衝幀數                         # 遲滯
BAR_HOLD_FRAMES = 20                     # 槓位移命中後維持 True 的幀數                       # 槓Gate保持
BAR_LOSS_TOL_FRAMES = 10                 # 槓暫時偵測不到時可容忍的連續幀數                   # 偵測遺失容忍
END_GRACE_FRAMES = 15                    # Gate 轉 False 後需連續幀數才真正結束分段           # 關檔緩衝

# ============================== 共用工具（utils for loops） ==============================

def _update_fps(start_time, frame_count, fps):                                                # 每秒刷新 FPS
    import time                                                                               # 時間模組
    frame_count += 1                                                                          # 幀+1
    elapsed = time.time() - start_time                                                        # 距離上次刷新秒數
    if elapsed >= 1:                                                                          # 每秒更新
        fps = frame_count / elapsed                                                           # 計算FPS
        frame_count = 0                                                                       # 幀歸零
        start_time = time.time()                                                              # 起點重設
    return start_time, frame_count, fps                                                       # 回傳更新值

def _qt_show(label, frame, fps):                                                              # 疊 FPS 並顯示到 Qt Label
    import cv2                                                                                # 影像處理
    from PyQt5 import QtGui, QtCore                                                           # Qt 顯示
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # 疊FPS
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                        # BGR→RGB
    h, w, ch = frame.shape                                                                    # 取尺寸
    qimg = QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)              # 建立 QImage
    pix = QtGui.QPixmap.fromImage(qimg)                                                       # 轉 QPixmap
    label.setPixmap(pix.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))  # 顯示
    return None                                                                               # 無回傳

def _shared_get(shared_state, shared_lock, key, default=None):                                # 讀單一 shared key
    with shared_lock:                                                                         # 進入臨界區
        return shared_state.get(key, default)                                                 # 取值

def _shared_set_many(shared_state, shared_lock, kv: dict):                                    # 批次回寫 shared_state
    with shared_lock:                                                                         # 進入臨界區
        for k, v in kv.items():                                                               # 逐項
            shared_state[k] = v                                                               # 回寫

def _start_segment_writers(folder, i, seg_no, frame, need_original, need_txt, txt_suffix):    # 開啟暫存 writer 與 txt
    import os, cv2                                                                            # 檔案/影像
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                                  # mp4v 編碼
    size = (frame.shape[1], frame.shape[0])                                                   # 取影像尺寸
    tmp = {}                                                                                  # 暫存路徑字典
    out = None                                                                                # 疊圖 writer
    original_out = None                                                                       # 原始 writer
    txt_file = None                                                                           # txt 物件
    # 視訊路徑
    if need_original:                                                                         # 是否需要原始輸出
        tmp['o'] = os.path.join(folder, f'_staging_cam{i}_seg{seg_no:03d}_original.mp4')      # 原始暫存檔名
        original_out = cv2.VideoWriter(tmp['o'], fourcc, 29, size)                            # 開原始 writer
    tmp['v'] = os.path.join(folder, f'_staging_cam{i}_seg{seg_no:03d}_vision.mp4')            # 疊圖暫存檔名
    out = cv2.VideoWriter(tmp['v'], fourcc, 29, size)                                         # 開疊圖 writer
    # 文字路徑
    if need_txt:                                                                              # 是否需要 txt
        tmp['t'] = os.path.join(folder, f'_staging_cam{i}_seg{seg_no:03d}_{txt_suffix}.txt')  # txt 暫存檔名
        txt_file = open(tmp['t'], 'w')                                                        # 開啟 txt
    return out, original_out, txt_file, tmp                                                   # 回傳 I/O 與路徑

def _close_io(out=None, original_out=None, txt_file=None):                                    # 關閉 I/O
    if txt_file is not None: txt_file.close()                                                 # 關 txt
    if out is not None: out.release()                                                         # 關疊圖
    if original_out is not None: original_out.release()                                       # 關原始
    return None                                                                               # 無回傳

def _end_and_move(folder, i, seg_no, tmp_paths, mapping):                                     # 結束段落、建資料夾並搬檔
    import os, shutil, time                                                                   # 檔案/時間
    end_ts = time.strftime("%Y%m%d_%H%M%S")                                                   # 以結束時間命名
    root_dir = os.path.dirname(folder)                                                        # recordings 根目錄
    rec_folder = os.path.join(root_dir, f"recording_{end_ts}")                                # 目標資料夾
    os.makedirs(rec_folder, exist_ok=True)                                                    # 建資料夾
    for k, new_name in mapping.items():                                                       # 依對應表搬移
        p = tmp_paths.get(k)                                                                  # 暫存路徑
        if p and os.path.exists(p):                                                           # 存在才搬
            shutil.move(p, os.path.join(rec_folder, new_name))                                # 搬並改名
    print(f"[SEG] End SEG {seg_no:03d} on cam{i+1} -> {rec_folder}")                          # 紀錄
    return rec_folder                                                                         # 回傳目的資料夾路徑

def _yolo_first_box_xywh(results):                                                            # 取第一個框 xywh
    try:                                                                                      # 防呆
        boxes = results[0].boxes if len(results) > 0 else None                                # 第1張結果
        if boxes is not None and boxes.xywh is not None and len(boxes.xywh) > 0:              # 有框
            x, y, w, h = boxes.xywh[0]                                                        # 取第一框
            return float(x), float(y), float(w), float(h)                                     # 轉 float
    except Exception:                                                                          # 例外
        pass                                                                                  # 略過
    return None                                                                               # 無框

def _latch_by_buffer(hit_cnt, miss_cnt, condition, buf_frames):                               # 命中/未中緩衝鎖存
    if condition:                                                                             # 條件成立
        hit_cnt += 1                                                                          # 命中+1
        miss_cnt = 0                                                                          # 未中歸零
        latched = hit_cnt >= buf_frames                                                       # 命中達閾 → True
    else:                                                                                     # 條件不成立
        miss_cnt += 1                                                                         # 未中+1
        hit_cnt = 0                                                                           # 命中歸零
        latched = not (miss_cnt >= buf_frames)                                                # 未中達閾 → False
    return hit_cnt, miss_cnt, latched                                                         # 回傳

# ============================== 改寫後的三個 loop ==============================

def benchpress_bar_loop(i, frame, label, save_sig, folder,                                    # 槓視角：更新 bar_y_changed + 分段錄影
                        start_time, frame_count, fps, out, original_out, model, txt_file,     # writer / 原始writer / 模型 / txt
                        frame_count_for_detect, barrier,                                      # 幀計數 / 柵欄
                        shared_state, shared_lock, BAR_MOVE_THRESH):                          # 共享狀態 / 鎖 / 槓位移門檻
    import cv2                                                                                # 影像處理

    # ---- FPS ----
    start_time, frame_count, fps = _update_fps(start_time, frame_count, fps)                  # 刷新 FPS

    # ---- 讀取 Gate 與錄影狀態 ----
    gate_ui = _shared_get(shared_state, shared_lock, "recording_sig", False)                  # UI Gate
    cam_rec_key = f"rec_cam{i}"                                                               # 是否在錄 key
    cam_seg_key = f"seg_cam{i}"                                                               # 段號 key
    end_false_key = f"rec_end_false_cam{i}"                                                   # Gate False 連續幀 key
    tmp_paths_key = f"tmp_paths_cam{i}"                                                       # 暫存路徑 key
    is_rec = _shared_get(shared_state, shared_lock, cam_rec_key, False)                       # 是否在錄
    seg_no = _shared_get(shared_state, shared_lock, cam_seg_key, 0)                           # 段號
    end_false_cnt = _shared_get(shared_state, shared_lock, end_false_key, 0)                  # False 緩衝
    tmp_paths = _shared_get(shared_state, shared_lock, tmp_paths_key, {})                     # 暫存路徑

    # ---- UI 未啟動：收乾淨並顯示 ----
    if not gate_ui:                                                                           # 未按錄影
        _close_io(out, original_out, txt_file)                                                # 關 I/O
        out, original_out, txt_file = None, None, None                                        # 清 I/O 變數
        save_sig = False                                                                      # 清保存
        frame_count_for_detect = 0                                                            # 幀歸零
        _shared_set_many(shared_state, shared_lock, {cam_rec_key: False})                     # 標記不在錄
        _qt_show(label, frame, fps)                                                           # 顯示
        barrier.wait()                                                                        # 同步
        return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file  # 回傳

    # ---- YOLO：取當幀槓 y + 疊圖 ----
    results = model.predict(source=frame, imgsz=320, conf=0.5, verbose=False)                 # YOLO 推論
    xywh = _yolo_first_box_xywh(results)                                                      # 取第一框
    for r in results:                                                                          # 疊圖
        frame = r.plot()                                                                      # 繪結果

    # ---- 槓 Gate（保持/遺失容忍）----
    prev_x = _shared_get(shared_state, shared_lock, "prev_bar_x", None)                       # 上一幀 x
    prev_y = _shared_get(shared_state, shared_lock, "prev_bar_y", None)                       # 上一幀 y
    bar_hold_key = f"bar_hold_cam{i}"                                                         # 保持 key
    bar_loss_key = f"bar_loss_cnt_cam{i}"                                                     # 遺失 key
    bar_hold = _shared_get(shared_state, shared_lock, bar_hold_key, 0)                        # 讀保持
    bar_loss = _shared_get(shared_state, shared_lock, bar_loss_key, 0)                        # 讀遺失

    if xywh is not None:                                                                      # 有偵測
        x_center, y_center, width, height = xywh                                              # 取中心
        bar_loss = 0                                                                          # 遺失歸零
        # ① 主要 Y 門檻
        if prev_y is not None and abs(y_center - prev_y) >= BAR_MOVE_THRESH:                  # Y 位移達門檻
            bar_hold = BAR_HOLD_FRAMES                                                        # 保持滿格
        # ② 保持續命（X 或 Y 移動 ≥1）
        elif bar_hold > 0 and ((prev_x is not None and abs(x_center - prev_x) >= 1) or abs(y_center - prev_y if prev_y is not None else 0) >= 1):  # 續命條件
            bar_hold = BAR_HOLD_FRAMES                                                        # 續命
        else:
            bar_hold = max(0, bar_hold - 1)                                                   # 遞減
        _shared_set_many(shared_state, shared_lock, {"prev_bar_x": x_center, "prev_bar_y": y_center})  # 更新 prev
    else:                                                                                     # 無偵測
        bar_loss += 1                                                                         # 遺失+1
        bar_hold = max(0, bar_hold - 1) if bar_loss <= BAR_LOSS_TOL_FRAMES else 0            # 容忍內遞減，超過清零

    _shared_set_many(shared_state, shared_lock, {bar_hold_key: bar_hold, bar_loss_key: bar_loss})  # 回寫保持/遺失
    bar_changed = bar_hold > 0                                                                # 槓 Gate 值
    _shared_set_many(shared_state, shared_lock, {"bar_y_changed": bar_changed})               # 回寫 Gate
    gate_body = _shared_get(shared_state, shared_lock, "body_detected", False)                # 人體 Gate
    should_record = gate_ui and gate_body and bar_changed                                     # 三 Gate 決定

    # ---- 開新段（只建暫存）----
    if should_record and not is_rec:                                                          # False→True
        seg_no += 1                                                                           # 段+1
        out, original_out, txt_file, tmp_paths = _start_segment_writers(                      # 開 writer
            folder, i, seg_no, frame, need_original=True, need_txt=True, txt_suffix="bar"     # 原始+txt
        )
        _shared_set_many(shared_state, shared_lock, {
            cam_seg_key: seg_no, cam_rec_key: True, end_false_key: 0, tmp_paths_key: tmp_paths
        })                                                                                    # 回寫狀態
        frame_count_for_detect = 0                                                            # 幀歸零
        print(f"[BAR] Start SEG {seg_no:03d} on cam{i+1}")                                    # log

    # ---- 寫入或結束 ----
    if should_record:                                                                         # 錄影中
        if original_out is not None: original_out.write(frame)                                # 寫原始
        if out is not None: out.write(frame)                                                  # 寫疊圖
        frame_count_for_detect += 1                                                           # 幀+1
        if txt_file is not None:                                                              # 寫 txt
            if xywh is not None:
                x_center, y_center, width, height = xywh                                      # 取框
                txt_file.write(f"{frame_count_for_detect},{x_center},{y_center},{width},{height}\n")  # 寫座標
            else:
                txt_file.write(f"{frame_count_for_detect},no detection\n")                    # 無偵測
        _shared_set_many(shared_state, shared_lock, {end_false_key: 0})                       # 重置關檔緩衝
    else:                                                                                     # Gate False
        end_false_cnt = min(END_GRACE_FRAMES, end_false_cnt + 1)                              # False 累+1
        _shared_set_many(shared_state, shared_lock, {end_false_key: end_false_cnt})           # 回寫
        if is_rec and end_false_cnt >= END_GRACE_FRAMES:                                      # 達緩衝→結束
            _close_io(out, original_out, txt_file)                                            # 關 I/O
            out, original_out, txt_file = None, None, None                                    # 清 I/O 變數
            paths = _shared_get(shared_state, shared_lock, tmp_paths_key, {})                 # 取暫存
            _shared_set_many(shared_state, shared_lock, {cam_rec_key: False, tmp_paths_key: {}})  # 清狀態
            _end_and_move(folder, i, seg_no, paths, mapping={
                "o": "original_vision1.mp4", "v": "vision1.mp4", "t": "yolo_coordinates.txt"
            })                                                                                # 搬檔改名
        if not is_rec:                                                                        # 原本沒錄
            frame_count_for_detect = 0                                                        # 幀歸零

    # ---- 顯示與同步 ----
    _qt_show(label, frame, fps)                                                               # 顯示
    barrier.wait()                                                                            # 同步
    return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file  # 回傳


def benchpress_body_loop(i, frame, label, save_sig, folder,                                   # 人體視角：更新 body_detected + 分段錄影
                         start_time, frame_count, fps, out, model, txt_file,                  # writer / 模型 / txt
                         frame_count_for_detect, skeleton_connections, barrier,               # 幀計數 / 連線 / 柵欄
                         shared_state, shared_lock):                                          # 共享狀態 / 鎖
    import cv2, numpy as np                                                                   # 影像處理/陣列

    # ---- FPS ----
    start_time, frame_count, fps = _update_fps(start_time, frame_count, fps)                  # 刷新 FPS

    # ---- 讀取 Gate 與狀態 ----
    if not skeleton_connections:                                                              # 預設連線
        skeleton_connections = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(4,6),(5,7)]              # 簡化預設
    gate_ui = _shared_get(shared_state, shared_lock, "recording_sig", False)                  # UI Gate
    cam_rec_key = f"rec_cam{i}"                                                               # 是否在錄 key
    cam_seg_key = f"seg_cam{i}"                                                               # 段號 key
    hit_key = f"body_hit_cnt_cam{i}"                                                          # 命中 key
    miss_key = f"body_miss_cnt_cam{i}"                                                        # 未中 key
    end_false_key = f"rec_end_false_cam{i}"                                                   # Gate False key
    tmp_paths_key = f"tmp_paths_cam{i}"                                                       # 暫存路徑 key
    is_rec = _shared_get(shared_state, shared_lock, cam_rec_key, False)                       # 是否在錄
    seg_no = _shared_get(shared_state, shared_lock, cam_seg_key, 0)                           # 段號
    hit_cnt = _shared_get(shared_state, shared_lock, hit_key, 0)                              # 命中幀
    miss_cnt = _shared_get(shared_state, shared_lock, miss_key, 0)                            # 未中幀
    latched_body = _shared_get(shared_state, shared_lock, "body_detected", False)             # 鎖存人體
    end_false_cnt = _shared_get(shared_state, shared_lock, end_false_key, 0)                  # 關檔緩衝
    tmp_paths = _shared_get(shared_state, shared_lock, tmp_paths_key, {})                     # 暫存路徑

    # ---- UI 未啟動：收乾淨並顯示 ----
    if not gate_ui:                                                                           # 未按錄影
        _close_io(out, None, txt_file)                                                        # 關 I/O（本視角無原始）
        out, txt_file = None, None                                                            # 清 I/O 變數
        save_sig = False                                                                      # 清保存
        frame_count_for_detect = 0                                                            # 幀歸零
        _shared_set_many(shared_state, shared_lock, {cam_rec_key: False})                     # 標記不在錄
        _qt_show(label, frame, fps)                                                           # 顯示
        barrier.wait()                                                                        # 同步
        return start_time, frame_count, fps, out, frame_count_for_detect, save_sig, txt_file  # 回傳

    # ---- YOLO 關鍵點偵測 + 繪圖 ----
    try:
        results = list(model(source=frame, stream=True, verbose=False))                       # 逐幀 keypoints
    except Exception as e:
        results = []                                                                          # 失敗視為無偵測
        print(f"[benchpress_body_loop] model error: {e}")                                     # log

    frame_count_for_detect += 1                                                               # 幀+1
    body_detected_now, frame_rows = False, []                                                 # 當幀旗標/txt 暫存

    if results and getattr(results[0], "keypoints", None) is not None:                        # 有 keypoints
        kpts = results[0].keypoints                                                           # 取關鍵點
        xy = kpts.xy                                                                          # 取 xy（tensor 或 ndarray）
        if hasattr(xy, "detach"): xy = xy.detach().cpu().numpy()                              # tensor→numpy
        first = xy[0] if len(xy) > 0 else None                                                # 取第一人
        if first is not None:                                                                 # 有人體
            body_detected_now = True                                                          # 標記偵測
            K = first.shape[0]                                                                # 點數
            pts = []                                                                          # 畫圖點集
            for idx in range(K):                                                              # 逐點
                xk, yk = int(first[idx,0]), int(first[idx,1])                                 # 轉 int
                if xk==0 and yk==0:                                                           # 無效點
                    pts.append(None)                                                          # 記 None
                else:
                    pts.append((xk, yk))                                                      # 記點
                    cv2.circle(frame, (xk,yk), 5, (0,255,0), cv2.FILLED)                      # 畫點
                frame_rows.append(f"{frame_count_for_detect},{idx},{xk},{yk}")                # 記錄 txt
            for a,b in skeleton_connections:                                                  # 逐線
                if a<len(pts) and b<len(pts) and pts[a] and pts[b]:                           # 檢查
                    cv2.line(frame, pts[a], pts[b], (0,255,255), 2)                           # 畫線

    # ---- 人體 Gate 20 幀遲滯 ----
    hit_cnt, miss_cnt, latched_from_now = _latch_by_buffer(hit_cnt, miss_cnt, body_detected_now, BODY_BUF_FRAMES)  # 緩衝鎖存
    latched_body = latched_from_now if body_detected_now or miss_cnt >= BODY_BUF_FRAMES else latched_body          # 更新鎖存值
    _shared_set_many(shared_state, shared_lock, {hit_key: hit_cnt, miss_key: miss_cnt, "body_detected": latched_body})  # 回寫統計

    # ---- 三 Gate 決定 ----
    gate_bar = _shared_get(shared_state, shared_lock, "bar_y_changed", False)                 # 槓 Gate
    should_record = gate_ui and latched_body and gate_bar                                     # 三 Gate 決定

    # ---- 開新段（只建暫存）----
    if should_record and not is_rec:                                                          # False→True
        seg_no += 1                                                                           # 段+1
        out, _, txt_file, tmp_paths = _start_segment_writers(                                 # 開 writer（無原始）
            folder, i, seg_no, frame, need_original=False, need_txt=True, txt_suffix="body"   # 疊圖+txt
        )
        _shared_set_many(shared_state, shared_lock, {
            cam_seg_key: seg_no, cam_rec_key: True, end_false_key: 0, tmp_paths_key: tmp_paths
        })                                                                                    # 回寫狀態
        print(f"[BODY] Start SEG {seg_no:03d} on cam{i+1} (latched_body={latched_body})")     # log

    # ---- 寫入或結束 ----
    if should_record:                                                                         # 錄影中
        if out is not None: out.write(frame)                                                  # 寫疊圖
        if txt_file is not None:                                                              # 寫 txt
            if body_detected_now and frame_rows:
                txt_file.write("\n".join(frame_rows) + "\n")                                  # 批次寫
            else:
                txt_file.write(f"{frame_count_for_detect},no detection\n")                    # 無偵測
        _shared_set_many(shared_state, shared_lock, {end_false_key: 0})                       # 重置關檔緩衝
    else:                                                                                     # Gate False
        end_false_cnt = min(END_GRACE_FRAMES, end_false_cnt + 1)                              # False 累+1
        _shared_set_many(shared_state, shared_lock, {end_false_key: end_false_cnt})           # 回寫
        is_rec = _shared_get(shared_state, shared_lock, cam_rec_key, False)                   # 重新讀 is_rec
        if is_rec and end_false_cnt >= END_GRACE_FRAMES:                                      # 達緩衝→結束
            _close_io(out, None, txt_file)                                                    # 關 I/O
            out, txt_file = None, None                                                        # 清 I/O 變數
            paths = _shared_get(shared_state, shared_lock, tmp_paths_key, {})                 # 取暫存
            _shared_set_many(shared_state, shared_lock, {cam_rec_key: False, tmp_paths_key: {}})  # 清狀態
            _end_and_move(folder, i, seg_no, paths, mapping={"v": "vision2.mp4", "t": "yolo_body_keypoints.txt"})  # 搬檔改名

    # ---- 顯示與同步 ----
    _qt_show(label, frame, fps)                                                               # 顯示
    barrier.wait()                                                                            # 同步
    return start_time, frame_count, fps, out, frame_count_for_detect, save_sig, txt_file      # 回傳


def benchpress_head_loop(i, frame, label, save_sig, folder,                                   # 頭部視角：依 Gate 分段錄影
                         start_time, frame_count, fps, out, original_out,                     # writer / 原始writer
                         frame_count_for_detect, barrier,                                     # 幀計數 / 柵欄
                         shared_state, shared_lock):                                          # 共享狀態 / 鎖
    import cv2                                                                                # 影像處理

    # ---- FPS ----
    start_time, frame_count, fps = _update_fps(start_time, frame_count, fps)                  # 刷新 FPS

    # 視角調整（如不需可移除）
    frame = cv2.rotate(frame, cv2.ROTATE_180)                                                 # 旋轉 180 度

    # ---- Gate 與分段狀態 ----
    gate_ui  = _shared_get(shared_state, shared_lock, "recording_sig", False)                 # UI Gate
    gate_body= _shared_get(shared_state, shared_lock, "body_detected", False)                 # 人體 Gate
    gate_bar = _shared_get(shared_state, shared_lock, "bar_y_changed", False)                 # 槓 Gate
    should_record = gate_ui and gate_body and gate_bar                                        # 三 Gate 決定

    cam_rec_key = f"rec_cam{i}"                                                               # 是否在錄 key
    cam_seg_key = f"seg_cam{i}"                                                               # 段號 key
    end_false_key = f"rec_end_false_cam{i}"                                                   # Gate False 連續幀 key
    tmp_paths_key = f"tmp_paths_cam{i}"                                                       # 暫存路徑 key

    is_rec = _shared_get(shared_state, shared_lock, cam_rec_key, False)                       # 是否在錄
    seg_no = _shared_get(shared_state, shared_lock, cam_seg_key, 0)                           # 段號
    end_false_cnt = _shared_get(shared_state, shared_lock, end_false_key, 0)                  # False 緩衝
    tmp_paths = _shared_get(shared_state, shared_lock, tmp_paths_key, {})                     # 暫存路徑

    # ---- 開新段（只建暫存）----
    if should_record and not is_rec:                                                          # False→True
        seg_no += 1                                                                           # 段+1
        out, original_out, _, tmp_paths = _start_segment_writers(                             # 開 writer（含原始）
            folder, i, seg_no, frame, need_original=True, need_txt=False, txt_suffix=""       # 無 txt
        )
        _shared_set_many(shared_state, shared_lock, {
            cam_seg_key: seg_no, cam_rec_key: True, end_false_key: 0, tmp_paths_key: tmp_paths
        })                                                                                    # 回寫狀態
        print(f"[HEAD] Start SEG {seg_no:03d} on cam{i+1}")                                   # log

    # ---- 寫入或結束 ----
    if should_record:                                                                         # 錄影中
        if original_out is not None: original_out.write(frame)                                # 寫原始
        if out is not None: out.write(frame)                                                  # 寫疊圖
        _shared_set_many(shared_state, shared_lock, {end_false_key: 0})                       # 重置關檔緩衝
    else:                                                                                     # Gate False
        end_false_cnt = min(END_GRACE_FRAMES, end_false_cnt + 1)                              # False 累+1
        _shared_set_many(shared_state, shared_lock, {end_false_key: end_false_cnt})           # 回寫
        if is_rec and end_false_cnt >= END_GRACE_FRAMES:                                      # 達緩衝→結束
            _close_io(out, original_out, None)                                                # 關 I/O
            out, original_out = None, None                                                    # 清 I/O 變數
            paths = _shared_get(shared_state, shared_lock, tmp_paths_key, {})                 # 取暫存
            _shared_set_many(shared_state, shared_lock, {cam_rec_key: False, tmp_paths_key: {}})  # 清狀態
            _end_and_move(folder, i, seg_no, paths, mapping={"o": "original_vision3.mp4", "v": "vision3.mp4"})  # 搬檔改名
        frame_count_for_detect = 0                                                            # 幀歸零（可選）

    # ---- 顯示與同步 ----
    _qt_show(label, frame, fps)                                                               # 顯示
    barrier.wait()                                                                            # 同步
    return start_time, frame_count, fps, out, original_out, save_sig, frame_count_for_detect  # 回傳
