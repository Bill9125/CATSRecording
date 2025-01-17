import time, os, cv2
from PyQt5 import QtCore, QtGui

def deadlift_bar_loop(i, frame, label, save_sig, recording_sig, folder,
                      start_time, frame_count, fps, out, model, txt_file, frame_count_for_detect):
    # fps 計算
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    
    # frame 處理
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    results = model(source=frame, imgsz=320, conf=0.5, verbose=False)
    boxes = results[0].boxes
    detected = False
    for result in results:
        frame = result.plot()
    
    # write result
    for box in boxes.xywh:
        detected = True
        x_center, y_center, width, height = box
        if recording_sig and txt_file is not None:
            frame_count_for_detect += 1
            txt_file.write(f"{frame_count_for_detect},{x_center},{y_center},{width},{height}\n")
            
    if not detected and recording_sig and txt_file is not None:
        frame_count_for_detect += 1
        txt_file.write(f"{frame_count_for_detect},no detection\n")

    # 錄影開始
    if recording_sig:
        if out is None:  # 初始化 VideoWriter
            file = os.path.join(folder, f'vision{i + 1}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_size = (frame.shape[1], frame.shape[0])  # 幀大小 (width, height)
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)
            print(f"Initialized VideoWriter for camera {i + 1}")
        
        if out is not None:
            out.write(frame)
            
    elif not recording_sig:
        frame_count_for_detect = 0

    # 錄影結束
    if save_sig and out is not None:
        out.release()
        out = None
        print(f"Released VideoWriter for camera {i + 1}")
        save_sig = False

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out, frame_count_for_detect

def deadlift_bone_loop(i, frame, label, save_sig, recording_sig, folder,
                       start_time, frame_count, fps, out, model, txt_file, frame_count_for_detect, skeleton_connections):
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        
    # frame 處理
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    results = model(source=frame, stream=True, verbose=False)
    frame_count_for_detect += 1
    for r2 in results:
        boxes = r2.boxes
        keypoints = r2.keypoints
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # cv2.rectangle(frame2, (x1, y1), (x2, y2), (255, 0, 255), 3)
            confidence = round(box.conf[0].item(), 2)
            cls2 = int(box.cls[0].item())
            cv2.putText(frame, f"person {confidence}",
                        (max(0, x1), max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            kpts = keypoints[i]
                            # 獲取關鍵點的座標和置信度
            keypoints_xy = kpts.xy  # shape: (1, 17, 2) -> 1 組 17 個關鍵點，每個關鍵點有 (x, y) 座標

            for j in range(keypoints_xy.shape[1]):  # keypoints_xy.shape[1] 為 17，表示有 17 個關鍵點
                
                if results:
                    if recording_sig and txt_file is not None:
                        txt_file.write(f"{frame_count_for_detect},{j},{int(keypoints_xy[0, j, 0])},{int(keypoints_xy[0, j, 1])}\n")
                else:
                    # Write "no detection" if no landmarks are detected
                    if recording_sig and txt_file is not None:
                        txt_file.write(f"{frame_count_for_detect},no detection\n")
            # 繪製骨架
            kp_coords = []
            for kp in keypoints.xy[i]:
                x_kp, y_kp = int(kp[0].item()), int(kp[1].item())  # Get x, y coordinates
                kp_coords.append((x_kp, y_kp))
                cv2.circle(frame, (x_kp, y_kp), 5, (0, 255, 0), cv2.FILLED)
                print(x_kp, y_kp)

            # Draw skeleton
            for start_idx, end_idx in skeleton_connections:
                if start_idx < len(kp_coords) and end_idx < len(kp_coords):
                    # Skip lines that connect to (0, 0)
                    if kp_coords[start_idx] == (0, 0) or kp_coords[end_idx] == (0, 0):
                        continue
                    cv2.line(frame, kp_coords[start_idx], kp_coords[end_idx], (0, 255, 255), 2)

    
    # 錄影開始
    if recording_sig:
        if out is None:  # 初始化 VideoWriter
            file = os.path.join(folder, f'vision{i + 1}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_size = (frame.shape[1], frame.shape[0])  # 幀大小 (width, height)
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)
            print(f"Initialized VideoWriter for camera {i + 1}")
        
        if out is not None:
            out.write(frame)
    elif not recording_sig:
        frame_count_for_detect = 0
        
    # 錄影結束
    if save_sig and out is not None:
        out.release()
        out = None
        print(f"Released VideoWriter for camera {i + 1}")
        save_sig = False

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out
    
def deadlift_general_loop(i, frame, label, save_sig, recording_sig, folder,
                          start_time, frame_count, fps, out):
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
        
        if out is not None:
            out.write(frame)
    
    # 錄影結束    
    if save_sig and out is not None:
        out.release()
        out = None
        print(f"Released VideoWriter for camera {i + 1}")
        save_sig = False

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out
    
def benchpress_bar_loop(i, frame, label, save_sig, recording_sig, folder,
                        start_time, frame_count, fps, out):
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        
    # 錄影開始
    if recording_sig:
        if out is None:  # 初始化 VideoWriter
            file = os.path.join(folder, f'vision{i + 1}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_size = (frame.shape[1], frame.shape[0])  # 幀大小 (width, height)
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)
            print(f"Initialized VideoWriter for camera {i + 1}")
        
        if out is not None:
            out.write(frame)
        
    # 錄影結束
    if save_sig and out is not None:
        out.release()
        out = None
        print(f"Released VideoWriter for camera {i + 1}")
        save_sig = False

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out
    
def benchpress_bone_loop_1(i, frame, label, save_sig, recording_sig, folder,
                           start_time, frame_count, fps, out):
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        
    # 錄影開始
    if recording_sig:
        if out is None:  # 初始化 VideoWriter
            file = os.path.join(folder, f'vision{i + 1}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_size = (frame.shape[1], frame.shape[0])  # 幀大小 (width, height)
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)
            print(f"Initialized VideoWriter for camera {i + 1}")
        
        if out is not None:
            out.write(frame)
    
    # 錄影結束
    if save_sig and out is not None:
        out.release()
        out = None
        print(f"Released VideoWriter for camera {i + 1}")
        save_sig = False

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out
    
def benchpress_bone_loop_2(i, frame, label, save_sig, recording_sig, folder,
                           start_time, frame_count, fps, out):
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        
    # 錄影開始
    if recording_sig:
        if out is None:  # 初始化 VideoWriter
            file = os.path.join(folder, f'vision{i + 1}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_size = (frame.shape[1], frame.shape[0])  # 幀大小 (width, height)
            out = cv2.VideoWriter(file, fourcc, 29, frame_size)
            print(f"Initialized VideoWriter for camera {i + 1}")
        
        if out is not None:
            out.write(frame)
    
    # 錄影結束
    if save_sig and out is not None:
        out.release()
        out = None
        print(f"Released VideoWriter for camera {i + 1}")
        save_sig = False

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out