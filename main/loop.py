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

def benchpress_bar_loop(i, frame, label, save_sig, recording_sig, folder,                 # Bar 視角：偵測槓 y 位移、更新 bar_y_changed  # 介面沿用
                        start_time, frame_count, fps, out, original_out, model, txt_file, # 視訊/模型/文字輸出 handler                     # 介面沿用
                        frame_count_for_detect, barrier,                                  # 偵測幀計數、同步柵欄                          # 介面沿用
                        shared_state, shared_lock, BAR_MOVE_THRESH):                      # ★ 共享狀態/鎖/位移閾值                        # 新增
    frame_count += 1                                                                      # 幀數+1
    elapsed_time = time.time() - start_time                                               # 距離上次計時
    if elapsed_time >= 1:                                                                 # 每秒更新一次
        fps = frame_count / elapsed_time                                                  # 計算FPS
        frame_count = 0                                                                   # 幀數歸零
        start_time = time.time()                                                          # 重設起點

    # --- 早退：UI 未按下錄影時，不跑 YOLO、僅顯示，並確保資源關閉 ---                               # 省算力
    with shared_lock:                                                                     # 臨界區讀 gate
        gate_ui = shared_state.get("recording_sig", False)                                # 讀 UI 旗標
    if not gate_ui:                                                                       # 未按下錄影
        # 關閉已開啟的 writer/txt（若有）                                                     # 收尾
        if out is not None: out.release(); out = None                                     # 關閉疊圖 writer
        if original_out is not None: original_out.release(); original_out = None          # 關閉原始 writer
        if txt_file is not None: txt_file.close(); txt_file = None                        # 關閉 txt
        save_sig = False                                                                  # 清除保存旗標
        frame_count_for_detect = 0                                                        # 歸零偵測幀
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # 疊FPS
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                # BGR→RGB
        h, w, ch = frame.shape                                                            # 尺寸
        qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))  # QPixmap
        scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)  # 縮放
        label.setPixmap(scale_qpixmap)                                                    # 顯示
        barrier.wait()                                                                    # 同步
        return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file  # 回傳

    # --- UI 要求錄影：這裡才跑 YOLO 算當幀槓的 y 中心 ---                                           # 省資源
    results = model.predict(source=frame, imgsz=320, conf=0.5, verbose=False)             # YOLO 推論
    boxes = results[0].boxes if len(results) > 0 else None                                # 取第一張結果
    current_bar_y = None                                                                  # 當幀槓 y
    if boxes is not None and boxes.xywh is not None and len(boxes.xywh) > 0:              # 有框結果
        x_center, y_center, width, height = boxes.xywh[0]                                 # 取第一框（可依類別過濾）
        current_bar_y = float(y_center)                                                   # 設當幀 y
    for r in results:                                                                      # 疊圖
        frame = r.plot()                                                                  # 繪製結果

    # --- 更新 shared_state：bar_y_changed 與 prev_bar_y，並組 should_record ---                   # 單一臨界區
    with shared_lock:                                                                     # 進入鎖
        prev_y = shared_state.get("prev_bar_y", None)                                     # 取上一幀 y
        if prev_y is not None and current_bar_y is not None:                              # 兩幀都有值
            shared_state["bar_y_changed"] = abs(current_bar_y - prev_y) >= BAR_MOVE_THRESH# 是否超過閾值
        else:
            shared_state["bar_y_changed"] = False                                         # 無法比較→False
        if current_bar_y is not None:                                                     # 有當幀值
            shared_state["prev_bar_y"] = current_bar_y                                    # 更新上一幀

        gate_ui = shared_state.get("recording_sig", False)                                # 重新取 UI 旗標
        gate_body = shared_state.get("body_detected", False)                              # 取人體旗標
        gate_bar  = shared_state.get("bar_y_changed", False)                              # 取槓旗標
        should_record = gate_ui and gate_body and gate_bar                                # 三旗標 Gate

    # --- Gate 通過：建立/寫入 writer 與 txt；Gate 不通過：安全關閉 ---                               # 核心
    if should_record:                                                                      # 應錄影
        if original_out is None:                                                           # 原始 writer
            file = os.path.join(folder, f'original_vision{i + 1}.mp4')                    # 路徑
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                       # 編碼
            frame_size = (frame.shape[1], frame.shape[0])                                  # 尺寸
            original_out = cv2.VideoWriter(file, fourcc, 29, frame_size)                   # 建立
            print(f"Initialized VideoWriter for origin camera {i + 1}")                    # 訊息
        original_out.write(frame)                                                          # 寫入原始畫面

        if out is None:                                                                    # 疊圖 writer
            file = os.path.join(folder, f'vision{i + 1}.mp4')                              # 路徑
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                       # 編碼
            frame_size = (frame.shape[1], frame.shape[0])                                  # 尺寸
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)                            # 建立
            print(f"Initialized VideoWriter for camera {i + 1}")                           # 訊息
        out.write(frame)                                                                   # 寫入疊圖

        if txt_file is None:                                                               # txt 尚未開啟
            txt_file_path = os.path.join(folder, 'yolo_coordinates.txt')                   # 槓框輸出檔
            txt_file = open(txt_file_path, "w")                                            # 開檔覆寫
            frame_count_for_detect = 0                                                     # 偵測幀歸零
            print(f"Started writing data to {txt_file_path}")                               # 訊息

        # 寫入當幀偵測結果                                                                       # 記錄
        if boxes is not None and boxes.xywh is not None and len(boxes.xywh) > 0:           # 有框
            x_center, y_center, width, height = boxes.xywh[0]                              # 取第一框
            frame_count_for_detect += 1                                                    # 幀+1
            txt_file.write(f"{frame_count_for_detect},{x_center},{y_center},{width},{height}\n")  # 寫檔
        else:
            frame_count_for_detect += 1                                                    # 幀+1
            txt_file.write(f"{frame_count_for_detect},no detection\n")                     # 無偵測
    else:                                                                                  # 不應錄影
        frame_count_for_detect = 0                                                         # 偵測幀歸零
        if out is not None: out.release(); out = None                                      # 關閉疊圖 writer
        if original_out is not None: original_out.release(); original_out = None           # 關閉原始 writer
        if txt_file is not None: txt_file.close(); txt_file = None                         # 關閉 txt
        save_sig = False                                                                   # 清保存旗標

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # 疊FPS
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                     # BGR→RGB
    h, w, ch = frame.shape                                                                 # 尺寸
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))  # QPixmap
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)  # 縮放
    label.setPixmap(scale_qpixmap)                                                         # 顯示
    barrier.wait()                                                                         # 同步
    return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file  # 回傳

def benchpress_body_loop(i, frame, label, save_sig, recording_sig, folder,                # Body 視角：偵測人體、更新 body_detected         # 介面沿用
                         start_time, frame_count, fps, out, model, txt_file,             # 視訊/模型/文字輸出 handler                     # 介面沿用
                         frame_count_for_detect, skeleton_connections, barrier,          # 偵測幀計數/骨架連線/同步柵欄                  # 介面沿用
                         shared_state, shared_lock):                                     # ★ 共享狀態/鎖                                  # 新增
    frame_count += 1                                                                     # 幀+1
    elapsed_time = time.time() - start_time                                              # 經過秒數
    if elapsed_time >= 1:                                                                # 每秒更新
        fps = frame_count / elapsed_time                                                 # 計算FPS
        frame_count = 0                                                                  # 歸零
        start_time = time.time()                                                         # 重設

    if not skeleton_connections:                                                         # 預設骨架連線
        skeleton_connections = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(4,6),(5,7)]         # 連線

    # --- 早退：UI 未按下錄影時，不跑 YOLO、不更新人體旗標 ---                                      # 省算力
    with shared_lock:                                                                    # 臨界區讀 gate
        gate_ui = shared_state.get("recording_sig", False)                               # 讀 UI 旗標
    if not gate_ui:                                                                      # 未按下錄影
        if out is not None: out.release(); out = None                                    # 關閉 writer
        if txt_file is not None: txt_file.close(); txt_file = None                       # 關閉 txt
        save_sig = False                                                                 # 清保存旗標
        frame_count_for_detect = 0                                                       # 幀歸零
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # 疊FPS
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                               # BGR→RGB
        h, w, ch = frame.shape                                                           # 尺寸
        qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)) # QPixmap
        scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation) # 縮放
        label.setPixmap(scale_qpixmap)                                                   # 顯示
        barrier.wait()                                                                   # 同步
        return start_time, frame_count, fps, out, frame_count_for_detect, save_sig, txt_file  # 回傳

    # --- UI 要求錄影：跑 YOLO keypoints 偵測人體 ---                                               # 省資源
    try:
        results = list(model(source=frame, stream=True, verbose=False))                  # YOLO 關鍵點
    except Exception as e:
        results = []                                                                      # 失敗視為無
        print(f"[benchpress_body_loop] model call error: {e}")                            # 診斷

    frame_count_for_detect += 1                                                          # 偵測幀+1
    body_detected_now = False                                                            # 本幀人體旗標
    frame_rows = []                                                                       # txt 行列
    if results and getattr(results[0], "keypoints", None) is not None:                   # 有 keypoints
        r0 = results[0]                                                                   # 第一結果
        kpts = r0.keypoints                                                               # Keypoints 物件
        first_xy = None                                                                   # 初始化
        if hasattr(kpts, "xy"):                                                           # 取 xy
            xy = kpts.xy                                                                  # (num,K,2)
            if hasattr(xy, "detach"): xy = xy.detach().cpu().numpy()                     # tensor→numpy
            first_xy = xy[0] if len(xy) > 0 else None                                     # 第一人
        if first_xy is not None:                                                          # 有人
            body_detected_now = True                                                      # 設 True
            kp_coords = []                                                                # 畫圖用座標
            K = first_xy.shape[0]                                                         # 點數
            for idx in range(K):                                                          # 逐點
                x_kp, y_kp = int(first_xy[idx,0]), int(first_xy[idx,1])                  # 取座標
                if x_kp==0 and y_kp==0:                                                   # 無效點
                    kp_coords.append(None)                                                # 記 None
                else:
                    kp_coords.append((x_kp,y_kp))                                         # 記座標
                    cv2.circle(frame, (x_kp,y_kp), 5, (0,255,0), cv2.FILLED)             # 畫點
                frame_rows.append(f"{frame_count_for_detect},{idx},{x_kp},{y_kp}")       # 準備 txt
            for a,b in skeleton_connections:                                              # 畫線
                if a<len(kp_coords) and b<len(kp_coords) and kp_coords[a] and kp_coords[b]:  # 索引/有效檢查
                    cv2.line(frame, kp_coords[a], kp_coords[b], (0,255,255), 2)          # 畫線
    else:
        body_detected_now = False                                                         # 無結果→False

    # --- 寫回人體旗標並組 should_record ---                                                       # 臨界區
    with shared_lock:                                                                     # 進入鎖
        shared_state["body_detected"] = body_detected_now                                 # 更新人體旗標
        gate_ui  = shared_state.get("recording_sig", False)                               # 取 UI
        gate_bar = shared_state.get("bar_y_changed", False)                               # 取槓
        should_record = gate_ui and shared_state["body_detected"] and gate_bar            # 三旗標 Gate

    # --- Gate 控制寫檔/關檔 ---                                                                      # 核心
    if should_record:                                                                     # 應錄影
        if out is None:                                                                   # writer
            file = os.path.join(folder, f'vision{i + 1}.mp4')                             # 路徑
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                      # 編碼
            frame_size = (frame.shape[1], frame.shape[0])                                 # 尺寸
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)                           # 建立
            print(f"Initialized VideoWriter for camera {i + 1}")                          # 訊息
        out.write(frame)                                                                  # 寫入畫面

        if txt_file is None:                                                              # txt 開檔
            txt_path = os.path.join(folder, 'yolo_body_keypoints.txt')                    # 檔名
            txt_file = open(txt_path, "w")                                                # 開啟
            print(f"Started writing data to {txt_path}")                                   # 訊息

        if body_detected_now and frame_rows and txt_file is not None:                     # 有人且有資料
            txt_file.write("\n".join(frame_rows) + "\n")                                  # 批次寫入
        elif txt_file is not None:                                                        # 無人
            txt_file.write(f"{frame_count_for_detect},no detection\n")                    # 記無偵測
    else:                                                                                 # 不應錄影
        if txt_file is not None: txt_file.close(); txt_file = None                        # 關閉 txt
        if out is not None: out.release(); out = None                                     # 關閉 writer
        save_sig = False                                                                  # 清保存旗標

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # 疊FPS
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                    # BGR→RGB
    h, w, ch = frame.shape                                                                # 尺寸
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)) # QPixmap
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation) # 縮放
    label.setPixmap(scale_qpixmap)                                                        # 顯示
    barrier.wait()                                                                        # 同步
    return start_time, frame_count, fps, out, frame_count_for_detect, save_sig, txt_file  # 回傳

def benchpress_head_loop(i, frame, label, save_sig, recording_sig, folder,                # Head 視角：僅跟 Gate 錄影，不更新旗標           # 介面沿用
                         start_time, frame_count, fps, out, original_out, frame_count_for_detect, barrier,
                         shared_state, shared_lock):                                      # ★ 共享狀態/鎖                                  # 新增
    frame_count += 1                                                                      # 幀+1
    elapsed_time = time.time() - start_time                                               # 經過秒數
    if elapsed_time >= 1:                                                                 # 每秒更新
        fps = frame_count / elapsed_time                                                  # 計算FPS
        frame_count = 0                                                                   # 歸零
        start_time = time.time()                                                          # 重設

    frame = cv2.rotate(frame, cv2.ROTATE_180)                                             # 旋轉頭部畫面

    # --- 讀 Gate（should_record） ---                                                           # 三旗標 Gate
    with shared_lock:                                                                     # 臨界區
        gate_ui  = shared_state.get("recording_sig", False)                               # UI 旗標
        gate_body= shared_state.get("body_detected", False)                               # 人體旗標
        gate_bar = shared_state.get("bar_y_changed", False)                               # 槓旗標
        should_record = gate_ui and gate_body and gate_bar                                # 三旗標

    # --- Gate 控制 writer 建立/寫入 ---                                                             # 核心
    if should_record:                                                                     # 應錄影
        if original_out is None:                                                          # 原始 writer
            file = os.path.join(folder, f'original_vision{i + 1}.mp4')                    # 路徑
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                       # 編碼
            frame_size = (frame.shape[1], frame.shape[0])                                  # 尺寸
            original_out = cv2.VideoWriter(file, fourcc, 29, frame_size)                   # 建立
            print(f"Initialized VideoWriter for origin camera {i + 1}")                    # 訊息
        original_out.write(frame)                                                          # 寫入原始

        if out is None:                                                                    # 疊圖 writer
            file = os.path.join(folder, f'vision{i + 1}.mp4')                              # 路徑
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                       # 編碼
            frame_size = (frame.shape[1], frame.shape[0])                                  # 尺寸
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)                            # 建立
            print(f"Initialized VideoWriter for camera {i + 1}")                           # 訊息
        out.write(frame)                                                                   # 寫入疊圖
    else:                                                                                 # 不應錄影
        frame_count_for_detect = 0                                                        # 幀歸零（可視需求保留）
        if out is not None: out.release(); out = None                                     # 關閉疊圖 writer
        if original_out is not None: original_out.release(); original_out = None          # 關閉原始 writer
        save_sig = False                                                                  # 清保存旗標

    barrier.wait()                                                                        # 同步

    # --- 顯示畫面（即使不錄影也顯示） ---                                                          # UI 顯示
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                    # BGR→RGB
    cv2.putText(frame_rgb, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # 疊FPS
    h, w, ch = frame_rgb.shape                                                            # 尺寸
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))      # QPixmap
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)   # 縮放
    label.setPixmap(scale_qpixmap)                                                        # 顯示

    return start_time, frame_count, fps, out, original_out, save_sig, frame_count_for_detect  # 回傳
