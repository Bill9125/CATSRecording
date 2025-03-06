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

    # 錄影開始
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

    finalizing = not recording_sig and save_sig

    # frame 處理
    if recording_sig or finalizing:
        results = list(model(source=frame, stream=True, verbose=False))
        frame_count_for_detect += 1

        if results and results[0].boxes:  # ✅ 確保有偵測到人
            r2 = results[0]  # ✅ 只取第一個偵測結果（限制為單一人）
            boxes = r2.boxes
            keypoints = r2.keypoints

            x1, y1, x2, y2 = map(int, boxes.xyxy[0])  # ✅ 只取第一個人的 bounding box
            confidence = round(boxes.conf[0].item(), 2)
            cv2.putText(frame, f"person {confidence}",
                        (max(0, x1), max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            kpts = keypoints[0]  # ✅ 只取第一個人的骨架點
            keypoints_xy = kpts.xy  # shape: (1, 17, 2) -> 17 個關鍵點

            if txt_file is not None:
                for j in range(keypoints_xy.shape[1]):
                    txt_file.write(f"{frame_count_for_detect},{j},{int(keypoints_xy[0, j, 0])},{int(keypoints_xy[0, j, 1])}\n")

            # **繪製骨架**
            kp_coords = []
            for idx, kp in enumerate(keypoints.xy[0]):
                if idx == 2:  # 🔹 **跳過右眼**
                    continue
                x_kp, y_kp = int(kp[0].item()), int(kp[1].item())
                kp_coords.append((x_kp, y_kp))
                cv2.circle(frame, (x_kp, y_kp), 5, (0, 255, 0), cv2.FILLED)

            # **畫出骨架**
            for start_idx, end_idx in skeleton_connections:
                if start_idx == 2 or end_idx == 2:  # 🔹 **跳過右眼的骨架線**
                    continue
                if start_idx < len(kp_coords) and end_idx < len(kp_coords):
                    cv2.line(frame, kp_coords[start_idx], kp_coords[end_idx], (0, 255, 255), 2)

        else:
            # ❌ 沒有偵測到人，寫入 "no detection"
            if txt_file is not None:
                txt_file.write(f"{frame_count_for_detect},no detection\n")

    # ✅ 錄影完全結束後才關閉 txt_file
    if finalizing:
        if txt_file is not None:
            txt_file.close()
            txt_file = None
            print(f"Closed txt_file for camera {i + 1}")

        if out is not None:
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

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file
    
def benchpress_body_loop(i, frame, label, save_sig, recording_sig, folder,
                           start_time, frame_count, fps, out, original_out, excluded_indices, txt_file, pose, frame_count_for_detect, connections):
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        
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
            txt_file_path = os.path.join(folder, 'mediapipe_landmarks.txt')
            txt_file = open(txt_file_path, "w")  # ✅ 錄影開始時開啟檔案
            frame_count_for_detect = 0  # ✅ 只在錄影開始時歸零
            print(f"Started writing data to {txt_file_path}")
        
    # frame 處理
    # MediaPipe Pose Detection
    results = pose.process(frame)
    frame_count_for_detect += 1  # Increment frame count for each frame

    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx not in excluded_indices:
                if recording_sig and txt_file is not None:
                    x, y, z = landmark.x, landmark.y, landmark.z
                    txt_file.write(f"{frame_count_for_detect},{idx},{x},{y},{z}\n")
                # 只畫出不在排除範圍內的 landmarks
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)  # Example: Draw a blue circle for landmarks
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx not in excluded_indices and end_idx not in excluded_indices:
                start_landmark = results.pose_landmarks.landmark[start_idx]
                end_landmark = results.pose_landmarks.landmark[end_idx]
                x1, y1 = int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])
                x2, y2 = int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    if not results.pose_landmarks:
        # Write "no detection" if no landmarks are detected
        if recording_sig and txt_file is not None:
            txt_file.write(f"{frame_count_for_detect},no detection\n")

    frame = cv2.rotate(frame, cv2.ROTATE_180)
    
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
    if not recording_sig:
        frame_count_for_detect = 0
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
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file
    
def benchpress_head_loop(i, frame, label, save_sig, recording_sig, folder,
                           start_time, frame_count, fps, out, original_out, txt_file, 
                           model, frame_count_for_detect):
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
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    h, w, ch = frame.shape
    qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
    scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scale_qpixmap)
    return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file