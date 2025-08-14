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
            file = os.path.join(folder, f'vision{i + 1}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
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
            file = os.path.join(folder, f'vision{i + 1}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
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
            file = os.path.join(folder, f'vision{i + 1}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
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
            file = os.path.join(folder, f'vision{i + 1}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
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
            file = os.path.join(folder, f'vision{i + 1}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
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
            file = os.path.join(folder, f'vision{i + 1}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
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
    
def benchpress_bar_loop(i, frame, label, save_sig, recording_sig, folder,
                        start_time, frame_count, fps, out, original_out, model, txt_file, frame_count_for_detect, barrier):
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        
    # Save the original video frame
    if recording_sig:
        if original_out is None:
            file = os.path.join(folder, f'original_vision{i + 1}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_size = (frame.shape[1], frame.shape[0])  # 幀大小 (width, height)
            original_out = cv2.VideoWriter(file, fourcc, 29, frame_size)
            print(f"Initialized VideoWriter for origin camera {i + 1}")
        original_out.write(frame)

        if txt_file is None:
            txt_file_path = os.path.join(folder, 'yolo_coordinates.txt')
            txt_file = open(txt_file_path, "w")  # ✅ 錄影開始時開啟檔案
            frame_count_for_detect = 0  # ✅ 只在錄影開始時歸零
            print(f"Started writing data to {txt_file_path}")

    # frame 處理
    results = model.predict(source=frame, imgsz=320, conf=0.5, verbose=False)
    boxes = results[0].boxes
    detected = False  # Initialize detected to False at the start of each frame
    for result in results:
        frame = result.plot()
        
    # 錄影開始
    if recording_sig:
        if out is None:  # 初始化 VideoWriter
            file = os.path.join(folder, f'vision{i + 1}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_size = (frame.shape[1], frame.shape[0])  # 幀大小 (width, height)
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)
            print(f"Initialized VideoWriter for camera {i + 1}")
        out.write(frame)

    if recording_sig or txt_file is not None:
        for box in boxes.xywh:
            detected = True
            x_center, y_center, width, height = box
            frame_count_for_detect += 1
            txt_file.write(f"{frame_count_for_detect},{x_center},{y_center},{width},{height}\n")
            
        if not detected:
            frame_count_for_detect += 1
            txt_file.write(f"{frame_count_for_detect},no detection\n")

    if not recording_sig:
        frame_count_for_detect = 0
        # 錄影結束
        if save_sig and out is not None:
            out.release()
            original_out.release()
            print(f"Released VideoWriter for camera {i + 1}")
            save_sig = False
        out = None
        original_out = None
        if txt_file is not None:
            txt_file.close()
            txt_file = None  # ✅ 確保 `txt_file` 被正確關閉
            print(f"Closed txt_file for camera {i + 1}")

    barrier.wait()
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file
       
def benchpress_body_loop(i, frame, label, save_sig, recording_sig, folder,               # 與 head 相同簽名  # 同簽名
                         start_time, frame_count, fps, out, original_out,                # 與 head 相同簽名  # 同簽名
                         txt_file, model, frame_count_for_detect, barrier):              # ★ 改成接 YOLO model  # 吃YOLO
    frame_count += 1  # 幀計數+1
    elapsed_time = time.time() - start_time  # 距上次計時秒數
    if elapsed_time >= 1:  # 每秒更新 FPS
        fps = frame_count / elapsed_time  # 計算 FPS
        frame_count = 0  # 歸零
        start_time = time.time()  # 重設起點

    # 儲存原始影像
    if recording_sig:  # 錄影中才寫檔
        if original_out is None:  # 建立原始影片 writer
            file = os.path.join(folder, f'original_vision{i + 1}.avi')  # 路徑
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 編碼
            frame_size = (frame.shape[1], frame.shape[0])  # (w,h)
            original_out = cv2.VideoWriter(file, fourcc, 29, frame_size)  # 建立 writer
            print(f"Initialized VideoWriter for origin camera {i + 1}")  # 訊息
        original_out.write(frame)  # 寫原始幀

        if txt_file is None:  # 首次開啟 txt
            txt_file_path = os.path.join(folder, 'yolo_skeleton_top_11m.txt')  # 檔名
            txt_file = open(txt_file_path, "w")  # 覆寫開啟
            frame_count_for_detect = 0  # 偵測幀數歸零
            print(f"Started writing data to {txt_file_path}")  # 訊息

    # ---- YOLO 關鍵點推論（同 head 作法）----
    results = model.predict(source=frame, conf=0.5, verbose=False)  # YOLO 推論
    frame_count_for_detect += 1  # 幀+1

    wrote_any = False  # 是否有寫入關鍵點
    if results and results[0].keypoints:  # 有 keypoints
        for kp_det in results[0].keypoints:  # 逐個偵測對象
            xy_list = kp_det.xy.tolist()  # (K,2) → list
            if not xy_list or not xy_list[0]:  # 無資料
                continue  # 跳過

            for idx, (x, y) in enumerate(xy_list[0]):  # 逐關節
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)  # 畫點
                if recording_sig and txt_file is not None:  # 寫txt
                    txt_file.write(f"{frame_count_for_detect},{idx},{x},{y}\n")  # 幀,索引,x,y
                    wrote_any = True  # 有寫入

    if recording_sig and txt_file is not None and not wrote_any:  # 本幀無任何點
        txt_file.write(f"{frame_count_for_detect},no detection\n")  # 記錄無偵測

    # 疊畫面輸出
    if recording_sig:  # 錄影中
        if out is None:  # 建立疊畫影片 writer
            file = os.path.join(folder, f'vision{i + 1}.avi')  # 檔名
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 編碼
            frame_size = (frame.shape[1], frame.shape[0])  # (w,h)
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)  # 建立 writer
            print(f"Initialized VideoWriter for camera {i + 1}")  # 訊息
        out.write(frame)  # 寫幀

    # 停止錄影：關閉資源
    if not recording_sig:  # 停止狀態
        frame_count_for_detect = 0  # 歸零
        if save_sig and out is not None:  # 釋放 writer
            out.release()  # 關閉疊畫影片
            original_out.release()  # 關閉原始影片
            print(f"Released VideoWriter for camera {i + 1}")  # 訊息
            save_sig = False  # 清旗標
        out = None  # 置空
        original_out = None  # 置空
        if txt_file is not None:  # 關閉txt
            txt_file.close()  # 關閉
            txt_file = None  # 置空
            print(f"Closed txt_file for camera {i + 1}")  # 訊息

    barrier.wait()  # 與其他視角同步
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR→RGB
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2, cv2.LINE_AA)  # 疊 FPS
    h, w, ch = frame.shape  # 尺寸
    qpixmap = QtGui.QPixmap.fromImage(  # 轉QImage
        QtGui.QImage(frame.data, w, h, ch*w, QtGui.QImage.Format_RGB888))  # QImage
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(),
                                   QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)  # 等比縮放
    label.setPixmap(scale_qpixmap)  # 顯示於 UI
    return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file  # 回傳

    
def benchpress_head_loop(i, frame, label, save_sig, recording_sig, folder,
                           start_time, frame_count, fps, out, original_out, txt_file, 
                           model, frame_count_for_detect, barrier):
    connections = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5)]
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    # 儲存原始影像幀
    if recording_sig:
        if original_out is None:
            file = os.path.join(folder, f'original_vision{i + 1}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_size = (frame.shape[1], frame.shape[0])  # 幀大小 (width, height)
            original_out = cv2.VideoWriter(file, fourcc, 29, frame_size)
            print(f"Initialized VideoWriter for origin camera {i + 1}")
        original_out.write(frame)

        if txt_file is None:
            txt_file_path = os.path.join(folder, 'yolo_skeleton.txt')
            txt_file = open(txt_file_path, "w")  # ✅ 錄影開始時開啟檔案
            frame_count_for_detect = 0  # ✅ 只在錄影開始時歸零
            print(f"Started writing data to {txt_file_path}")
        
    # frame 處理
    results = model.predict(source=frame, conf=0.5, verbose = False)
    frame_count_for_detect += 1  # Increment frame count for each frame

    frame_data = []
    if results[0].keypoints:
        for result in results[0].keypoints:
            keypoints = result.xy.tolist()

            if not keypoints or not keypoints[0]:
                if recording_sig and txt_file is not None:
                    txt_file.write(f"{frame_count_for_detect},no detection\n")
                pass
            
            keypoint_list = []
            for keypoint in keypoints[0]:  
                if len(keypoint) == 2:  
                    x, y = keypoint
                    keypoint_list.append((x, y))
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            for (start_idx, end_idx) in connections:
                if start_idx < len(keypoint_list) and end_idx < len(keypoint_list):
                    start_point = keypoint_list[start_idx]
                    end_point = keypoint_list[end_idx]
                    
                    if start_point != (0, 0) and end_point != (0, 0):
                        cv2.line(frame, (int(start_point[0]), int(start_point[1])),
                                (int(end_point[0]), int(end_point[1])), (255, 0, 0), 2)
            frame_data.append(keypoint_list)
            
    # 錄影開始
    if recording_sig:
        if out is None:  # 初始化 VideoWriter
            file = os.path.join(folder, f'vision{i + 1}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_size = (frame.shape[1], frame.shape[0])  # 幀大小 (width, height)
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)
            print(f"Initialized VideoWriter for camera {i + 1}")
        out.write(frame)
        
        if txt_file is not None:
            txt_file.write(f"Frame {frame_count_for_detect}: {frame_data}\n")
    
    if not recording_sig:
        frame_count_for_detect = 0
        # 錄影結束
        if save_sig and out is not None:
            out.release()
            original_out.release()
            print(f"Released VideoWriter for camera {i + 1}")
            save_sig = False
        out = None
        original_out = None
        if txt_file is not None:
            txt_file.close()
            txt_file = None  # ✅ 確保 `txt_file` 被正確關閉
            print(f"Closed txt_file for camera {i + 1}")
    
    barrier.wait()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file