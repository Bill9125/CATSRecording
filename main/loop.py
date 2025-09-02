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


def benchpress_bar_loop(i, frame, label, save_sig, folder,                                   # 槓視角：更新 bar_y_changed + 分段錄影
                        start_time, frame_count, fps, out, original_out, model, txt_file,    # writer / 原始writer / 模型 / txt
                        frame_count_for_detect, barrier,                                     # 幀計數 / 柵欄
                        shared_state, shared_lock, BAR_MOVE_THRESH):                         # 共享狀態 / 鎖 / 槓位移門檻
    import time, os, cv2, shutil                                                              # 需要搬檔用 shutil
    from PyQt5 import QtGui, QtCore                                                           # Qt 顯示

    # ---- FPS ----
    frame_count += 1                                                                          # 幀+1
    elapsed_time = time.time() - start_time                                                   # 距上次刷新秒數
    if elapsed_time >= 1:                                                                     # 每秒更新
        fps = frame_count / elapsed_time                                                      # 計算FPS
        frame_count = 0                                                                       # 幀歸零
        start_time = time.time()                                                              # 起點重設

    # ---- 分段/緩衝狀態 ----
    cam_rec_key = f"rec_cam{i}"                                                               # 是否在錄 key
    cam_seg_key = f"seg_cam{i}"                                                               # 段號 key
    end_false_key = f"rec_end_false_cam{i}"                                                   # Gate False 連續幀 key
    tmp_paths_key = f"tmp_paths_cam{i}"                                                       # 暫存檔路徑 key（字典）
    with shared_lock:                                                                          # 臨界區
        is_rec = shared_state.get(cam_rec_key, False)                                         # 現況：是否在錄
        seg_no = shared_state.get(cam_seg_key, 0)                                             # 現況：段號
        end_false_cnt = shared_state.get(end_false_key, 0)                                    # 關檔緩衝計數
        tmp_paths = shared_state.get(tmp_paths_key, {})                                       # 讀暫存檔路徑

    # ---- UI Gate ----
    with shared_lock:
        gate_ui = shared_state.get("recording_sig", False)                                    # 讀 UI Gate
    if not gate_ui:                                                                           # UI 未按錄影：全收乾淨
        if out is not None: out.release(); out = None                                         # 關疊圖 writer
        if original_out is not None: original_out.release(); original_out = None              # 關原始 writer
        if txt_file is not None: txt_file.close(); txt_file = None                            # 關 txt
        save_sig = False                                                                      # 清保存旗標
        frame_count_for_detect = 0                                                            # 幀歸零
        with shared_lock:
            shared_state[cam_rec_key] = False                                                 # 標記不在錄
        # 顯示
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # 疊FPS
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                    # BGR→RGB
        h, w, ch = frame.shape                                                                # 尺寸
        qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))  # QPixmap
        scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)  # 縮放
        label.setPixmap(scale_qpixmap)                                                        # 顯示
        barrier.wait()                                                                        # 同步
        return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file  # 回傳

    # ---- YOLO：取當幀槓 y ----
    results = model.predict(source=frame, imgsz=320, conf=0.5, verbose=False)                 # YOLO 推論
    boxes = results[0].boxes if len(results) > 0 else None                                    # 第1張結果
    current_bar_y = None                                                                       # 當幀槓 y
    if boxes is not None and boxes.xywh is not None and len(boxes.xywh) > 0:                  # 有框
        x_center, y_center, width, height = boxes.xywh[0]                                     # 取第一框
        current_bar_y = float(y_center)                                                       # 取 y
    for r in results:                                                                          # 疊圖
        frame = r.plot()                                                                      # 繪結果

    # ---- 槓 Gate（保持/遺失容忍）----
    with shared_lock:                                                                          # 臨界區
        prev_x = shared_state.get("prev_bar_x", None)                                         # 上一幀 x
        prev_y = shared_state.get("prev_bar_y", None)                                         # 上一幀 y
        bar_hold_key = f"bar_hold_cam{i}"                                                     # 保持 key
        bar_loss_key = f"bar_loss_cnt_cam{i}"                                                 # 遺失 key
        bar_hold = shared_state.get(bar_hold_key, 0)                                          # 讀保持
        bar_loss = shared_state.get(bar_loss_key, 0)                                          # 讀遺失

        if current_bar_y is not None:                                                         # 有偵測
            x_center, y_center, width, height = boxes.xywh[0]                                 # 拿當前框的中心
            bar_loss = 0                                                                      # 遺失歸零

            if prev_y is not None:
                # ① 正常 Y 閾值判斷
                if abs(float(y_center) - prev_y) >= BAR_MOVE_THRESH:                          
                    bar_hold = BAR_HOLD_FRAMES                                                # 達到主要門檻，保持滿格
                # ② 如果已經在保持狀態，則只要 X 或 Y 有 ≥1pixel 位移就續命
                elif bar_hold > 0 and (
                    (prev_x is not None and abs(float(x_center) - prev_x) >= 1) or
                    abs(float(y_center) - prev_y) >= 1):
                    bar_hold = BAR_HOLD_FRAMES                                                # 保持續命
                else:
                    bar_hold = max(0, bar_hold - 1)                                           # 否則遞減
            else:
                bar_hold = max(0, bar_hold - 1)

            shared_state["prev_bar_x"] = float(x_center)                                      # 更新 prev x
            shared_state["prev_bar_y"] = float(y_center)                                      # 更新 prev y

        else:                                                                                 # 無偵測
            bar_loss += 1                                                                     # 遺失+1
            bar_hold = max(0, bar_hold - 1) if bar_loss <= BAR_LOSS_TOL_FRAMES else 0         # 容忍內遞減，超過清零

        shared_state[bar_hold_key] = bar_hold                                                 # 回寫保持
        shared_state[bar_loss_key] = bar_loss                                                 # 回寫遺失
        shared_state["bar_y_changed"] = (bar_hold > 0)                                        # 決定 Gate
        gate_body = shared_state.get("body_detected", False)                                  # 人體 Gate
        gate_bar  = shared_state["bar_y_changed"]                                             # 槓 Gate
        should_record = gate_ui and gate_body and gate_bar                                    # 三 Gate 決定
            

    # ---- 開新段：只建「暫存檔」，資料夾留到結束再建 ----
    if should_record and not is_rec:                                                          # False→True
        seg_no += 1                                                                           # 段+1
        # 建暫存檔名（避免覆蓋，等結束再搬進時間資料夾）
        tmp_file_o = os.path.join(folder, f'_staging_cam{i}_seg{seg_no:03d}_original.mp4')    # 原始暫存
        tmp_file_v = os.path.join(folder, f'_staging_cam{i}_seg{seg_no:03d}_vision.mp4')      # 疊圖暫存
        tmp_file_t = os.path.join(folder, f'_staging_cam{i}_seg{seg_no:03d}_bar.txt')         # txt 暫存
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                              # 編碼
        frame_size = (frame.shape[1], frame.shape[0])                                         # 尺寸
        original_out = cv2.VideoWriter(tmp_file_o, fourcc, 29, frame_size)                    # 開原始 writer（暫存）
        out = cv2.VideoWriter(tmp_file_v, fourcc, 29, frame_size)                             # 開疊圖 writer（暫存）
        txt_file = open(tmp_file_t, "w")                                                      # 開 txt（暫存）
        with shared_lock:
            shared_state[cam_seg_key] = seg_no                                                # 回寫段號
            shared_state[cam_rec_key] = True                                                  # 標記開始錄
            shared_state[end_false_key] = 0                                                   # 清關檔緩衝
            shared_state[tmp_paths_key] = {"o": tmp_file_o, "v": tmp_file_v, "t": tmp_file_t} # 存暫存檔路徑
        frame_count_for_detect = 0                                                            # 幀歸零
        print(f"[BAR] Start SEG {seg_no:03d} on cam{i+1}")                                    # log

    # ---- 寫入或結束（含「結束時建立資料夾並搬檔」）----
    if should_record:                                                                          # 錄影中
        if original_out is not None: original_out.write(frame)                                 # 寫原始
        if out is not None: out.write(frame)                                                   # 寫疊圖
        if txt_file is not None:                                                               # 寫txt
            if boxes is not None and boxes.xywh is not None and len(boxes.xywh) > 0:
                x_center, y_center, width, height = boxes.xywh[0]                              # 第一框
                frame_count_for_detect += 1                                                    # 幀+1
                txt_file.write(f"{frame_count_for_detect},{x_center},{y_center},{width},{height}\n")  # 座標
            else:
                frame_count_for_detect += 1                                                    # 幀+1
                txt_file.write(f"{frame_count_for_detect},no detection\n")                     # 無偵測
        with shared_lock:
            shared_state[end_false_key] = 0                                                    # 重置關檔緩衝
    else:                                                                                      # Gate False
        with shared_lock:
            end_false_cnt = min(END_GRACE_FRAMES, end_false_cnt + 1)                           # False 累+1
            shared_state[end_false_key] = end_false_cnt                                        # 回寫
        if is_rec and end_false_cnt >= END_GRACE_FRAMES:                                       # 達緩衝→結束
            # 關檔
            if txt_file is not None: txt_file.close(); txt_file = None                         # 關txt
            if out is not None: out.release(); out = None                                      # 關疊圖
            if original_out is not None: original_out.release(); original_out = None           # 關原始

            # 以「結束時間」建立資料夾（建立在 recordings 根目錄）並搬檔
            end_ts = time.strftime("%Y%m%d_%H%M%S")                                            # 結束時間戳
            with shared_lock:
                paths = shared_state.get(tmp_paths_key, {})                                    # 取暫存路徑
                shared_state[cam_rec_key] = False                                              # 標記不在錄
                shared_state[tmp_paths_key] = {}                                               # 清暫存

            # 以「結束時間」建立 recording_{end_ts} 資料夾
            root_dir = os.path.dirname(folder)
            rec_folder = os.path.join(root_dir, f"recording_{end_ts}")
            os.makedirs(rec_folder, exist_ok=True)

            # 搬移 + 重新命名
            mapping = {
                "o": "original_vision1.mp4",    # 原始
                "v": "vision1.mp4",             # 疊圖
                "t": "yolo_coordinates.txt"     # 槓座標
            }
            for k, new_name in mapping.items():
                p = paths.get(k)
                if p and os.path.exists(p):
                    shutil.move(p, os.path.join(rec_folder, new_name))

            print(f"[BAR] End SEG {seg_no:03d} on cam{i+1} -> {rec_folder}")
                            # log
        if not is_rec:                                                                          # 原本沒錄
            frame_count_for_detect = 0                                                         # 幀歸零

    # ---- 顯示與同步 ----
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # 疊FPS
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                          # BGR→RGB
    h, w, ch = frame.shape                                                                      # 尺寸
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))  # QPixmap
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)    # 縮放
    label.setPixmap(scale_qpixmap)                                                              # 顯示
    barrier.wait()                                                                              # 同步
    return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file  # 回傳


def benchpress_body_loop(i, frame, label, save_sig, folder,                                   # 人體視角：更新 body_detected + 分段錄影
                         start_time, frame_count, fps, out, model, txt_file,                  # writer / 模型 / txt
                         frame_count_for_detect, skeleton_connections, barrier,               # 幀計數 / 連線 / 柵欄
                         shared_state, shared_lock):                                          # 共享狀態 / 鎖
    import time, os, cv2, numpy as np, shutil                                                 # 需要搬檔用 shutil
    from PyQt5 import QtGui, QtCore                                                           # Qt 顯示

    # ---- FPS ----
    frame_count += 1                                                                          # 幀+1
    elapsed_time = time.time() - start_time                                                   # 距上次刷新秒數
    if elapsed_time >= 1:                                                                     # 每秒更新
        fps = frame_count / elapsed_time                                                      # 計算FPS
        frame_count = 0                                                                       # 幀歸零
        start_time = time.time()                                                              # 起點重設

    # ---- 預設骨架連線 ----
    if not skeleton_connections:                                                              # 未傳入
        skeleton_connections = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(4,6),(5,7)]              # 簡化預設

    # ---- 分段/緩衝狀態 ----
    cam_rec_key = f"rec_cam{i}"                                                               # 是否在錄 key
    cam_seg_key = f"seg_cam{i}"                                                               # 段號 key
    hit_cnt_key = f"body_hit_cnt_cam{i}"                                                      # 命中 key
    miss_cnt_key = f"body_miss_cnt_cam{i}"                                                    # 未中 key
    end_false_key = f"rec_end_false_cam{i}"                                                   # Gate False key
    tmp_paths_key = f"tmp_paths_cam{i}"                                                       # 暫存檔路徑 key
    with shared_lock:
        is_rec   = shared_state.get(cam_rec_key, False)                                       # 是否在錄
        seg_no   = shared_state.get(cam_seg_key, 0)                                           # 段號
        gate_ui  = shared_state.get("recording_sig", False)                                   # UI Gate
        hit_cnt  = shared_state.get(hit_cnt_key, 0)                                           # 命中幀
        miss_cnt = shared_state.get(miss_cnt_key, 0)                                          # 未中幀
        latched_body = shared_state.get("body_detected", False)                               # 鎖存人體旗標
        end_false_cnt = shared_state.get(end_false_key, 0)                                    # 關檔緩衝
        tmp_paths = shared_state.get(tmp_paths_key, {})                                       # 暫存路徑

    # ---- UI 未啟動：收乾淨 ----
    if not gate_ui:
        if out is not None: out.release(); out = None                                         # 關 writer
        if txt_file is not None: txt_file.close(); txt_file = None                            # 關 txt
        save_sig = False                                                                      # 清保存
        frame_count_for_detect = 0                                                            # 幀歸零
        with shared_lock:
            shared_state[cam_rec_key] = False                                                 # 標記不在錄
        # 顯示
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # 疊FPS
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                    # BGR→RGB
        h, w, ch = frame.shape                                                                # 尺寸
        qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))  # QPixmap
        scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)  # 縮放
        label.setPixmap(scale_qpixmap)                                                        # 顯示
        barrier.wait()                                                                        # 同步
        return start_time, frame_count, fps, out, frame_count_for_detect, save_sig, txt_file  # 回傳

    # ---- YOLO 關鍵點偵測 ----
    try:
        results = list(model(source=frame, stream=True, verbose=False))                      # 逐幀 keypoints
    except Exception as e:
        results = []                                                                          # 失敗視為無偵測
        print(f"[benchpress_body_loop] model call error: {e}")                                # log

    frame_count_for_detect += 1                                                              # 幀+1
    body_detected_now = False                                                                # 當幀人體旗標
    frame_rows = []                                                                           # txt 行暫存

    if results and getattr(results[0], "keypoints", None) is not None:                       # 有 keypoints
        r0 = results[0]                                                                       # 第一個結果
        kpts = r0.keypoints                                                                   # 關鍵點物件
        first_xy = None                                                                       # 初始化
        if hasattr(kpts, "xy"):                                                               # 有 xy
            xy = kpts.xy                                                                      # 取 xy
            if hasattr(xy, "detach"): xy = xy.detach().cpu().numpy()                          # tensor→numpy
            first_xy = xy[0] if len(xy) > 0 else None                                         # 第一人
        if first_xy is not None:                                                              # 有人體
            body_detected_now = True                                                          # 當幀有偵測
            kp_coords = []                                                                    # 畫圖點集
            K = first_xy.shape[0]                                                             # 點數
            for idx in range(K):                                                              # 逐點
                x_kp, y_kp = int(first_xy[idx,0]), int(first_xy[idx,1])                      # 轉 int
                if x_kp==0 and y_kp==0:                                                       # 無效點
                    kp_coords.append(None)                                                    # 記 None
                else:
                    kp_coords.append((x_kp,y_kp))                                             # 記有效點
                    cv2.circle(frame, (x_kp,y_kp), 5, (0,255,0), cv2.FILLED)                 # 畫點
                frame_rows.append(f"{frame_count_for_detect},{idx},{x_kp},{y_kp}")           # 記錄 txt
            for a,b in skeleton_connections:                                                  # 逐線
                if a<len(kp_coords) and b<len(kp_coords) and kp_coords[a] and kp_coords[b]:   # 檢查
                    cv2.line(frame, kp_coords[a], kp_coords[b], (0,255,255), 2)              # 畫線

    # ---- 20 幀遲滯（人體緩衝）----
    if body_detected_now:                                                                     # 當幀有偵測
        hit_cnt  += 1                                                                         # 命中+1
        miss_cnt  = 0                                                                         # 未中歸零
        if hit_cnt >= BODY_BUF_FRAMES:                                                        # 達閾
            latched_body = True                                                               # 鎖存 True
    else:                                                                                     # 當幀無偵測
        miss_cnt += 1                                                                         # 未中+1
        hit_cnt   = 0                                                                         # 命中歸零
        if miss_cnt >= BODY_BUF_FRAMES:                                                       # 達閾
            latched_body = False                                                              # 鎖存 False

    with shared_lock:                                                                         # 回寫共享
        shared_state[hit_cnt_key]  = hit_cnt                                                  # 回寫命中
        shared_state[miss_cnt_key] = miss_cnt                                                 # 回寫未中
        shared_state["body_detected"] = latched_body                                          # 回寫人體Gate
        gate_bar = shared_state.get("bar_y_changed", False)                                   # 槓 Gate
        should_record = gate_ui and latched_body and gate_bar                                 # 三 Gate 決定

    # ---- 開新段：只建「暫存檔」，資料夾留到結束再建 ----
    if should_record and not is_rec:                                                          # False→True
        seg_no += 1                                                                           # 段+1
        tmp_file_v = os.path.join(folder, f'_staging_cam{i}_seg{seg_no:03d}_vision.mp4')      # 影像暫存
        tmp_file_t = os.path.join(folder, f'_staging_cam{i}_seg{seg_no:03d}_body.txt')        # txt 暫存
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                              # 編碼
        frame_size = (frame.shape[1], frame.shape[0])                                         # 尺寸
        out = cv2.VideoWriter(tmp_file_v, fourcc, 29, frame_size)                             # 開 writer（暫存）
        txt_file = open(tmp_file_t, "w")                                                      # 開 txt（暫存）
        with shared_lock:
            shared_state[cam_seg_key] = seg_no                                                # 回寫段號
            shared_state[cam_rec_key] = True                                                  # 標記在錄
            shared_state[end_false_key] = 0                                                   # 清關檔緩衝
            shared_state[tmp_paths_key] = {"v": tmp_file_v, "t": tmp_file_t}                  # 存暫存路徑
        print(f"[BODY] Start SEG {seg_no:03d} on cam{i+1} (latched_body={latched_body})")     # log

    # ---- 寫入或結束（含「結束時建立資料夾並搬檔」）----
    if should_record:                                                                          # 錄影中
        if out is not None: out.write(frame)                                                   # 寫影像
        if txt_file is not None:                                                               # 寫txt
            if body_detected_now and frame_rows:
                txt_file.write("\n".join(frame_rows) + "\n")                                   # 批次寫
            else:
                txt_file.write(f"{frame_count_for_detect},no detection\n")                     # 無偵測
        with shared_lock:
            shared_state[end_false_key] = 0                                                    # 重置關檔緩衝
    else:                                                                                      # Gate False
        with shared_lock:
            end_false_cnt = min(END_GRACE_FRAMES, end_false_cnt + 1)                           # False 累+1
            shared_state[end_false_key] = end_false_cnt                                        # 回寫
        if is_rec and end_false_cnt >= END_GRACE_FRAMES:                                       # 達緩衝→結束
            # 關檔
            if txt_file is not None: txt_file.close(); txt_file = None                         # 關txt
            if out is not None: out.release(); out = None                                      # 關writer

            # 以「結束時間」建立資料夾（建立在 recordings 根目錄）並搬檔
            end_ts = time.strftime("%Y%m%d_%H%M%S")                                            # 結束時間戳
            with shared_lock:
                paths = shared_state.get(tmp_paths_key, {})                                    # 取暫存路徑
                shared_state[cam_rec_key] = False                                              # 標記不在錄
                shared_state[tmp_paths_key] = {}                                               # 清暫存

            root_dir = os.path.dirname(folder)
            rec_folder = os.path.join(root_dir, f"recording_{end_ts}")
            os.makedirs(rec_folder, exist_ok=True)

            mapping = {
                "v": "vision2.mp4",                   # 疊圖
                "t": "yolo_body_keypoints.txt"        # 骨架關鍵點
            }
            for k, new_name in mapping.items():
                p = paths.get(k)
                if p and os.path.exists(p):
                    shutil.move(p, os.path.join(rec_folder, new_name))

            print(f"[BODY] End SEG {seg_no:03d} on cam{i+1} -> {rec_folder}")


    # ---- 顯示與同步 ----
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # 疊FPS
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                          # BGR→RGB
    h, w, ch = frame.shape                                                                      # 尺寸
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))  # QPixmap
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)    # 縮放
    label.setPixmap(scale_qpixmap)                                                              # 顯示
    barrier.wait()                                                                              # 同步
    return start_time, frame_count, fps, out, frame_count_for_detect, save_sig, txt_file        # 回傳


def benchpress_head_loop(i, frame, label, save_sig, folder,                                   # 頭部視角：僅依 Gate 分段錄影
                         start_time, frame_count, fps, out, original_out,                     # writer / 原始writer
                         frame_count_for_detect, barrier,                                     # 幀計數 / 柵欄
                         shared_state, shared_lock):                                          # 共享狀態 / 鎖
    import time, os, cv2, shutil                                                               # 需要搬檔用 shutil
    from PyQt5 import QtGui, QtCore                                                           # Qt 顯示

    # ---- FPS ----
    frame_count += 1                                                                          # 幀+1
    elapsed_time = time.time() - start_time                                                   # 距上次刷新秒數
    if elapsed_time >= 1:                                                                     # 每秒更新
        fps = frame_count / elapsed_time                                                      # 計算FPS
        frame_count = 0                                                                       # 幀歸零
        start_time = time.time()                                                              # 起點重設

    frame = cv2.rotate(frame, cv2.ROTATE_180)                                                 # 視角調整（如不需可移除）

    # ---- Gate 與分段狀態 ----
    cam_rec_key = f"rec_cam{i}"                                                               # 是否在錄 key
    cam_seg_key = f"seg_cam{i}"                                                               # 段號 key
    end_false_key = f"rec_end_false_cam{i}"                                                   # Gate False 連續幀 key
    tmp_paths_key = f"tmp_paths_cam{i}"                                                       # 暫存檔路徑 key
    with shared_lock:
        gate_ui  = shared_state.get("recording_sig", False)                                   # UI Gate
        gate_body= shared_state.get("body_detected", False)                                   # 人體 Gate
        gate_bar = shared_state.get("bar_y_changed", False)                                   # 槓 Gate
        should_record = gate_ui and gate_body and gate_bar                                    # 三 Gate 決定
        is_rec = shared_state.get(cam_rec_key, False)                                         # 是否在錄
        seg_no = shared_state.get(cam_seg_key, 0)                                             # 段號
        end_false_cnt = shared_state.get(end_false_key, 0)                                    # 關檔緩衝
        tmp_paths = shared_state.get(tmp_paths_key, {})                                       # 暫存路徑

    # ---- 開新段：只建「暫存檔」，資料夾留到結束再建 ----
    if should_record and not is_rec:                                                          # False→True
        seg_no += 1                                                                           # 段+1
        tmp_file_o = os.path.join(folder, f'_staging_cam{i}_seg{seg_no:03d}_original.mp4')    # 原始暫存
        tmp_file_v = os.path.join(folder, f'_staging_cam{i}_seg{seg_no:03d}_vision.mp4')      # 疊圖暫存
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                              # 編碼
        frame_size = (frame.shape[1], frame.shape[0])                                         # 尺寸
        original_out = cv2.VideoWriter(tmp_file_o, fourcc, 29, frame_size)                    # 開原始 writer（暫存）
        out = cv2.VideoWriter(tmp_file_v, fourcc, 29, frame_size)                             # 開疊圖 writer（暫存）
        with shared_lock:
            shared_state[cam_seg_key] = seg_no                                                # 回寫段號
            shared_state[cam_rec_key] = True                                                  # 標記在錄
            shared_state[end_false_key] = 0                                                   # 清關檔緩衝
            shared_state[tmp_paths_key] = {"o": tmp_file_o, "v": tmp_file_v}                  # 存暫存路徑
        print(f"[HEAD] Start SEG {seg_no:03d} on cam{i+1}")                                   # log

    # ---- 寫入或結束（含「結束時建立資料夾並搬檔」）----
    if should_record:                                                                          # 錄影中
        if original_out is not None: original_out.write(frame)                                 # 寫原始
        if out is not None: out.write(frame)                                                   # 寫疊圖
        with shared_lock:
            shared_state[end_false_key] = 0                                                    # 重置關檔緩衝
    else:                                                                                      # Gate False
        with shared_lock:
            end_false_cnt = min(END_GRACE_FRAMES, end_false_cnt + 1)                           # False 累+1
            shared_state[end_false_key] = end_false_cnt                                        # 回寫
        if is_rec and end_false_cnt >= END_GRACE_FRAMES:                                       # 達緩衝→結束
            # 關檔
            if out is not None: out.release(); out = None                                      # 關疊圖
            if original_out is not None: original_out.release(); original_out = None           # 關原始

            # 以「結束時間」建立資料夾（建立在 recordings 根目錄）並搬檔
            end_ts = time.strftime("%Y%m%d_%H%M%S")                                            # 結束時間戳
            with shared_lock:
                paths = shared_state.get(tmp_paths_key, {})                                    # 取暫存路徑
                shared_state[cam_rec_key] = False                                              # 標記不在錄
                shared_state[tmp_paths_key] = {}                                               # 清暫存

            root_dir = os.path.dirname(folder)
            rec_folder = os.path.join(root_dir, f"recording_{end_ts}")
            os.makedirs(rec_folder, exist_ok=True)

            mapping = {
                "o": "original_vision3.mp4",    # 原始
                "v": "vision3.mp4"              # 疊圖
            }
            for k, new_name in mapping.items():
                p = paths.get(k)
                if p and os.path.exists(p):
                    shutil.move(p, os.path.join(rec_folder, new_name))

            print(f"[HEAD] End SEG {seg_no:03d} on cam{i+1} -> {rec_folder}")

        frame_count_for_detect = 0                                                             # 幀歸零（可選）

    # ---- 顯示與同步 ----
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                         # BGR→RGB
    cv2.putText(frame_rgb, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # 疊FPS
    h, w, ch = frame_rgb.shape                                                                 # 尺寸
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))      # QPixmap
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)    # 縮放
    label.setPixmap(scale_qpixmap)                                                             # 顯示
    barrier.wait()                                                                             # 同步
    return start_time, frame_count, fps, out, original_out, save_sig, frame_count_for_detect   # 回傳

