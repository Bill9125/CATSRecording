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

def benchpress_bar_loop(i, frame, label, save_sig, recording_sig, folder,                 # Bar 視角：偵測槓 y 位移、更新 bar_y_changed     # 介面沿用
                        start_time, frame_count, fps, out, original_out, model, txt_file, # writer / 原始writer / 模型 / txt               # 介面沿用
                        frame_count_for_detect, barrier,                                  # 偵測幀計數 / 柵欄                              # 介面沿用
                        shared_state, shared_lock, BAR_MOVE_THRESH):                      # 共享狀態 / 鎖 / 位移閾值                        # 介面沿用
    # ---- FPS ----
    frame_count += 1                                                                       # 幀+1
    elapsed_time = time.time() - start_time                                                # 經過秒數
    if elapsed_time >= 1:                                                                  # 每秒更新
        fps = frame_count / elapsed_time                                                   # 算FPS
        frame_count = 0                                                                    # 幀歸零
        start_time = time.time()                                                           # 起點重設

    # ---- 讀/設分段狀態（每台相機各自維護）----
    cam_rec_key = f"rec_cam{i}"                                                            # 該相機是否正在錄影 的 key
    cam_seg_key = f"seg_cam{i}"                                                            # 該相機目前段號 的 key
    with shared_lock:                                                                       # 臨界區
        is_rec = shared_state.get(cam_rec_key, False)                                      # 預設不在錄
        seg_no = shared_state.get(cam_seg_key, 0)                                          # 預設段號0（尚未開檔）

    # ---- 若 UI 未開啟，直接收乾淨（不跑YOLO）----
    with shared_lock:
        gate_ui = shared_state.get("recording_sig", False)                                 # 讀 UI Gate
    if not gate_ui:                                                                        # UI 未按錄影
        if out is not None: out.release(); out = None                                      # 關閉疊圖 writer
        if original_out is not None: original_out.release(); original_out = None           # 關閉原始 writer
        if txt_file is not None: txt_file.close(); txt_file = None                         # 關閉 txt
        save_sig = False                                                                   # 清保存旗標
        frame_count_for_detect = 0                                                         # 幀歸零
        with shared_lock:
            shared_state[cam_rec_key] = False                                              # 標記：目前不在錄
        # 照片顯示
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # 疊FPS
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                 # BGR→RGB
        h, w, ch = frame.shape                                                             # 尺寸
        qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))  # QPixmap
        scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)  # 縮放
        label.setPixmap(scale_qpixmap)                                                     # 顯示
        barrier.wait()                                                                     # 同步
        return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file  # 回傳

    # ---- UI=ON：跑 YOLO 算當幀槓的 y ----
    results = model.predict(source=frame, imgsz=320, conf=0.5, verbose=False)              # YOLO 推論
    boxes = results[0].boxes if len(results) > 0 else None                                 # 取第一張結果
    current_bar_y = None                                                                    # 當幀 y
    if boxes is not None and boxes.xywh is not None and len(boxes.xywh) > 0:               # 有框
        x_center, y_center, width, height = boxes.xywh[0]                                  # 取第一框
        current_bar_y = float(y_center)                                                    # 當幀 y
    for r in results:                                                                       # 疊圖
        frame = r.plot()                                                                   # 繪製結果

    # ---- 更新 bar_y_changed 與 prev_bar_y，並算 Gate ----
    with shared_lock:                                                                       # 臨界區
        prev_y = shared_state.get("prev_bar_y", None)                                      # 取上一幀 y
        if prev_y is not None and current_bar_y is not None:                               # 可比較
            shared_state["bar_y_changed"] = abs(current_bar_y - prev_y) >= BAR_MOVE_THRESH # 判斷是否超閾
        else:
            shared_state["bar_y_changed"] = False                                          # 否則=False
        if current_bar_y is not None:                                                      # 有當幀 y
            shared_state["prev_bar_y"] = current_bar_y                                     # 更新 prev
        gate_ui  = shared_state.get("recording_sig", False)                                # 讀 UI
        gate_body= shared_state.get("body_detected", False)                                # 讀人體
        gate_bar = shared_state.get("bar_y_changed", False)                                # 讀槓
        should_record = gate_ui and gate_body and gate_bar                                 # 三 Gate

    # ---- 分段邏輯：轉場偵測 ----
    if should_record and not is_rec:                                                       # False→True：開新段
        seg_no += 1                                                                        # 段號+1
        with shared_lock:
            shared_state[cam_seg_key] = seg_no                                             # 回寫段號
            shared_state[cam_rec_key] = True                                               # 標記：開始錄
        frame_count_for_detect = 0                                                         # 新段幀計數歸零
        # 建立 writer 與 txt（以段號結尾）
        file_o = os.path.join(folder, f'original_vision{i + 1}_seg{seg_no:03d}.mp4')      # 原始檔名（分段）
        file_v = os.path.join(folder, f'vision{i + 1}_seg{seg_no:03d}.mp4')               # 疊圖檔名（分段）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                           # 編碼
        frame_size = (frame.shape[1], frame.shape[0])                                      # 尺寸
        original_out = cv2.VideoWriter(file_o, fourcc, 29, frame_size)                     # 建原始 writer
        out = cv2.VideoWriter(file_v, fourcc, 29, frame_size)                              # 建疊圖 writer
        txt_file_path = os.path.join(folder, f'yolo_coordinates_seg{seg_no:03d}.txt')      # txt 檔名（分段）
        txt_file = open(txt_file_path, "w")                                                # 開 txt
        print(f"[BAR] Start SEG {seg_no:03d} on cam{i+1}")                                 # 訊息

    if should_record:                                                                      # 真正在錄：持續寫
        if original_out is not None: original_out.write(frame)                             # 寫原始
        if out is not None: out.write(frame)                                               # 寫疊圖
        if txt_file is not None:                                                           # 寫 yolo座標
            if boxes is not None and boxes.xywh is not None and len(boxes.xywh) > 0:      # 有框
                x_center, y_center, width, height = boxes.xywh[0]                          # 第一框
                frame_count_for_detect += 1                                                # 幀+1
                txt_file.write(f"{frame_count_for_detect},{x_center},{y_center},{width},{height}\n")  # 寫入
            else:
                frame_count_for_detect += 1                                                # 幀+1
                txt_file.write(f"{frame_count_for_detect},no detection\n")                 # 無偵測
    else:
        if is_rec:                                                                         # True→False：段落結束
            if txt_file is not None: txt_file.close(); txt_file = None                     # 關 txt
            if out is not None: out.release(); out = None                                  # 關疊圖
            if original_out is not None: original_out.release(); original_out = None       # 關原始
            save_sig = False                                                               # 清保存旗標
            with shared_lock:
                shared_state[cam_rec_key] = False                                          # 標記：不在錄
            print(f"[BAR] End SEG {seg_no:03d} on cam{i+1}")                               # 訊息
        frame_count_for_detect = 0                                                         # 幀歸零

    # ---- 顯示與同步 ----
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # 疊FPS
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                     # BGR→RGB
    h, w, ch = frame.shape                                                                 # 尺寸
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))  # QPixmap
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)  # 縮放
    label.setPixmap(scale_qpixmap)                                                         # 顯示
    barrier.wait()                                                                         # 同步
    return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file  # 回傳

def benchpress_body_loop(i, frame, label, save_sig, recording_sig, folder,                # Body 視角：偵測人體並分段                       # 介面沿用
                         start_time, frame_count, fps, out, model, txt_file,             # writer / 模型 / txt                               # 介面沿用
                         frame_count_for_detect, skeleton_connections, barrier,          # 偵測幀計數 / 連線 / 柵欄                          # 介面沿用
                         shared_state, shared_lock):                                     # 共享狀態 / 鎖                                      # 介面沿用
    # ---- FPS ----
    frame_count += 1                                                                     # 幀+1
    elapsed_time = time.time() - start_time                                              # 經過秒數
    if elapsed_time >= 1:                                                                # 每秒更新
        fps = frame_count / elapsed_time                                                 # 算FPS
        frame_count = 0                                                                  # 幀歸零
        start_time = time.time()                                                         # 起點重設

    if not skeleton_connections:                                                         # 預設骨架連線
        skeleton_connections = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(4,6),(5,7)]         # 連線

    # ---- 讀/設分段狀態 ----
    cam_rec_key = f"rec_cam{i}"                                                          # 該相機是否正在錄
    cam_seg_key = f"seg_cam{i}"                                                          # 該相機段號
    with shared_lock:
        is_rec = shared_state.get(cam_rec_key, False)                                    # 既有錄影狀態
        seg_no = shared_state.get(cam_seg_key, 0)                                        # 既有段號
        gate_ui = shared_state.get("recording_sig", False)                               # 讀 UI Gate

    # ---- UI 未開 → 收乾淨 ----
    if not gate_ui:
        if out is not None: out.release(); out = None                                    # 關 writer
        if txt_file is not None: txt_file.close(); txt_file = None                       # 關 txt
        save_sig = False                                                                 # 清保存
        frame_count_for_detect = 0                                                       # 幀歸零
        with shared_lock:
            shared_state[cam_rec_key] = False                                            # 標記不在錄
        # 顯示
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # 疊FPS
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                               # BGR→RGB
        h, w, ch = frame.shape                                                           # 尺寸
        qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)) # QPixmap
        scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation) # 縮放
        label.setPixmap(scale_qpixmap)                                                   # 顯示
        barrier.wait()                                                                   # 同步
        return start_time, frame_count, fps, out, frame_count_for_detect, save_sig, txt_file  # 回傳

    # ---- YOLO 關鍵點偵測 ----
    try:
        results = list(model(source=frame, stream=True, verbose=False))                  # YOLO keypoints
    except Exception as e:
        results = []                                                                      # 失敗視為無
        print(f"[benchpress_body_loop] model call error: {e}")                            # 診斷

    frame_count_for_detect += 1                                                          # 偵測幀+1
    body_detected_now = False                                                            # 本幀是否有人
    frame_rows = []                                                                       # txt 行列
    if results and getattr(results[0], "keypoints", None) is not None:                   # 有 keypoints
        r0 = results[0]                                                                   # 第一結果
        kpts = r0.keypoints                                                               # Keypoints
        first_xy = None                                                                   # 預設
        if hasattr(kpts, "xy"):                                                           # 取 xy
            xy = kpts.xy                                                                  # (num,K,2)
            if hasattr(xy, "detach"): xy = xy.detach().cpu().numpy()                     # tensor→numpy
            first_xy = xy[0] if len(xy) > 0 else None                                     # 第一人
        if first_xy is not None:
            body_detected_now = True                                                      # 有人
            kp_coords = []                                                                # 畫圖用
            K = first_xy.shape[0]                                                         # 點數
            for idx in range(K):
                x_kp, y_kp = int(first_xy[idx,0]), int(first_xy[idx,1])                  # 座標
                if x_kp==0 and y_kp==0:
                    kp_coords.append(None)                                                # 無效
                else:
                    kp_coords.append((x_kp,y_kp))                                         # 有效
                    cv2.circle(frame, (x_kp,y_kp), 5, (0,255,0), cv2.FILLED)             # 畫點
                frame_rows.append(f"{frame_count_for_detect},{idx},{x_kp},{y_kp}")       # 記錄
            for a,b in skeleton_connections:                                              # 畫線
                if a<len(kp_coords) and b<len(kp_coords) and kp_coords[a] and kp_coords[b]:
                    cv2.line(frame, kp_coords[a], kp_coords[b], (0,255,255), 2)

    # ---- 寫回人體旗標並計算 Gate ----
    with shared_lock:
        shared_state["body_detected"] = body_detected_now                                 # 更新人體旗標
        gate_bar = shared_state.get("bar_y_changed", False)                               # 讀槓
        should_record = gate_ui and shared_state["body_detected"] and gate_bar            # 三 Gate

    # ---- 分段邏輯 ----
    if should_record and not is_rec:                                                      # False→True：開新段
        seg_no += 1                                                                       # 段+1
        with shared_lock:
            shared_state[cam_seg_key] = seg_no                                            # 回寫段號
            shared_state[cam_rec_key] = True                                              # 標記在錄
        # 建 writer/txt（段號後綴）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                          # 編碼
        frame_size = (frame.shape[1], frame.shape[0])                                     # 尺寸
        file_v = os.path.join(folder, f'vision{i + 1}_seg{seg_no:03d}.mp4')              # 檔名
        out = cv2.VideoWriter(file_v, fourcc, 29, frame_size)                             # 建 writer
        txt_path = os.path.join(folder, f'yolo_body_keypoints_seg{seg_no:03d}.txt')       # txt 檔名
        txt_file = open(txt_path, "w")                                                    # 開 txt
        print(f"[BODY] Start SEG {seg_no:03d} on cam{i+1}")                               # 訊息

    if should_record:                                                                     # 寫入
        if out is not None: out.write(frame)                                              # 寫畫面
        if txt_file is not None:
            if body_detected_now and frame_rows:
                txt_file.write("\n".join(frame_rows) + "\n")                              # 寫點列
            else:
                txt_file.write(f"{frame_count_for_detect},no detection\n")                # 無偵測
    else:
        if is_rec:                                                                        # True→False：結束一段
            if txt_file is not None: txt_file.close(); txt_file = None                    # 關 txt
            if out is not None: out.release(); out = None                                 # 關 writer
            save_sig = False                                                              # 清保存
            with shared_lock:
                shared_state[cam_rec_key] = False                                         # 標記不在錄
            print(f"[BODY] End SEG {seg_no:03d} on cam{i+1}")                             # 訊息

    # ---- 顯示與同步 ----
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # 疊FPS
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                    # BGR→RGB
    h, w, ch = frame.shape                                                                # 尺寸
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)) # QPixmap
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation) # 縮放
    label.setPixmap(scale_qpixmap)                                                        # 顯示
    barrier.wait()                                                                        # 同步
    return start_time, frame_count, fps, out, frame_count_for_detect, save_sig, txt_file  # 回傳

def benchpress_head_loop(i, frame, label, save_sig, recording_sig, folder,                # Head 視角：跟 Gate 錄影並分段                    # 介面沿用
                         start_time, frame_count, fps, out, original_out, frame_count_for_detect, barrier,
                         shared_state, shared_lock):                                      # 共享狀態 / 鎖                                      # 介面沿用
    # ---- FPS ----
    frame_count += 1                                                                      # 幀+1
    elapsed_time = time.time() - start_time                                               # 經過秒數
    if elapsed_time >= 1:                                                                 # 每秒更新
        fps = frame_count / elapsed_time                                                  # 算FPS
        frame_count = 0                                                                   # 幀歸零
        start_time = time.time()                                                          # 起點重設

    frame = cv2.rotate(frame, cv2.ROTATE_180)                                             # 視角調整

    # ---- 讀 Gate ----
    with shared_lock:
        gate_ui  = shared_state.get("recording_sig", False)                               # 讀 UI
        gate_body= shared_state.get("body_detected", False)                               # 讀人體
        gate_bar = shared_state.get("bar_y_changed", False)                               # 讀槓
        should_record = gate_ui and gate_body and gate_bar                                # 三 Gate

    # ---- 讀/設分段狀態 ----
    cam_rec_key = f"rec_cam{i}"                                                           # 該相機是否正在錄
    cam_seg_key = f"seg_cam{i}"                                                           # 段號
    with shared_lock:
        is_rec = shared_state.get(cam_rec_key, False)                                     # 既有錄影狀態
        seg_no = shared_state.get(cam_seg_key, 0)                                         # 既有段號

    # ---- 分段邏輯 ----
    if should_record and not is_rec:                                                      # False→True：開新段
        seg_no += 1                                                                       # 段+1
        with shared_lock:
            shared_state[cam_seg_key] = seg_no                                            # 回寫段號
            shared_state[cam_rec_key] = True                                              # 標記在錄
        # 建 writer（段號後綴）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                          # 編碼
        frame_size = (frame.shape[1], frame.shape[0])                                     # 尺寸
        file_o = os.path.join(folder, f'original_vision{i + 1}_seg{seg_no:03d}.mp4')     # 原始分段
        file_v = os.path.join(folder, f'vision{i + 1}_seg{seg_no:03d}.mp4')              # 疊圖分段
        original_out = cv2.VideoWriter(file_o, fourcc, 29, frame_size)                    # 建原始 writer
        out = cv2.VideoWriter(file_v, fourcc, 29, frame_size)                             # 建疊圖 writer
        print(f"[HEAD] Start SEG {seg_no:03d} on cam{i+1}")                               # 訊息

    if should_record:                                                                     # 寫入
        if original_out is not None: original_out.write(frame)                            # 原始
        if out is not None: out.write(frame)                                              # 疊圖
    else:
        if is_rec:                                                                        # True→False：結束一段
            if out is not None: out.release(); out = None                                 # 關疊圖
            if original_out is not None: original_out.release(); original_out = None      # 關原始
            save_sig = False                                                              # 清保存
            with shared_lock:
                shared_state[cam_rec_key] = False                                         # 標記不在錄
            print(f"[HEAD] End SEG {seg_no:03d} on cam{i+1}")                             # 訊息
        frame_count_for_detect = 0                                                        # 幀歸零（可選）

    # ---- 顯示 ----
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                    # BGR→RGB
    cv2.putText(frame_rgb, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)  # 疊FPS
    h, w, ch = frame_rgb.shape                                                            # 尺寸
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))      # QPixmap
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)   # 縮放
    label.setPixmap(scale_qpixmap)                                                        # 顯示

    barrier.wait()                                                                        # 同步
    return start_time, frame_count, fps, out, original_out, save_sig, frame_count_for_detect  # 回傳
