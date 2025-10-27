import time, os, cv2
from PyQt5 import QtCore, QtGui
import os, tempfile, shutil, uuid                         # 檔案/暫存/搬移/隨機ID  # 
from datetime import datetime                             # 只引入類別，便於 datetime.now()  # 

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
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
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
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
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
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
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
# ====== Squat ======
# ====== 全域設定與共用物件 ======
import os, time, cv2, threading                                                          # 基本模組
from PyQt5 import QtGui, QtCore                                                          # Qt 顯示

# 推論序列化鎖（cam1/cam2 用）：避免兩路同時搶 GPU 導致抖動/崩潰                        # 全域鎖
inference_lock = threading.Lock()                                                        # 全域推論鎖

# 統一 YOLO 參數                                                                         # 統一參數
YOLO_IMGSZ = 288                                                                         # 輸入影像邊長
YOLO_CONF  = 0.65                                                                        # 置信度閾值
YOLO_MAXDET= 1                                                                           # 只取一人/一物體

# 錄影設定（全部改 MJPG）                                                                 # MJPG 設定
FOURCC_MJPG = cv2.VideoWriter_fourcc(*'MJPG')                                            # MJPG fourcc
# FOURCC_MJPG = cv2.VideoWriter_fourcc(*'MJPG')                                            # MJPG fourcc
REC_FPS     = 29                                                                         # 錄影 FPS


def squat_bar_loop(i, frame, label, save_sig, recording_sig, folder,                      # 槓視角（cam1）  # 簽名不變，呼叫端免改
                   start_time, frame_count, fps, out, original_out,                       # original_out 不用，保留相容
                   model, txt_file, frame_count_for_detect, barrier):                     # 狀態/同步參數
    # ===== FPS =====
    frame_count += 1                                                                      # 幀+1
    elapsed_time = time.time() - start_time                                               # 距上次秒數
    if elapsed_time >= 1:                                                                 # 每秒刷新
        fps = frame_count / elapsed_time                                                  # 計算 FPS
        frame_count = 0                                                                   # 歸零
        start_time = time.time()                                                          # 重置

    # ===== 旋轉與乾淨幀 =====
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)                                    # 旋轉 90°
    clean_frame = frame.copy()                                                            # 乾淨幀（寫檔用）
    vis_frame   = frame.copy()                                                            # 可視幀（UI 疊圖用）

    # ===== 錄影：只寫乾淨畫面 =====
    if recording_sig:                                                                     # 錄影中
        if out is None:                                                                   # 初始化 writer
            path = os.path.join(folder, f'vision{i+1}.avi')                               # 檔名
            h, w = frame.shape[:2]                                                        # 幀高寬
            out = cv2.VideoWriter(path, FOURCC_MJPG, REC_FPS, (w, h))                     # MJPG writer
        out.write(clean_frame)                                                            # 先寫乾淨畫面（不疊圖）

        if txt_file is None:                                                              # 初始化 txt（一次）
            txt_path = os.path.join(folder, 'yolo_coordinates.txt')                       # Bar 偵測輸出
            txt_file = open(txt_path, "w")                                                # 開檔覆寫
            frame_count_for_detect = 0                                                    # 幀計數歸零

    # ===== 推論：只為 UI 與 txt，不寫回檔案 =====
    boxes = None                                                                          # 預設無框
    if recording_sig:                                                                     # 錄影時才推論
        with inference_lock:                                                              # 序列化避免搶 GPU
            results = model(source=frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF,               # YOLO 單幀推論
                             max_det=YOLO_MAXDET, verbose=False)                          # 只取一人
        boxes = results[0].boxes if (results and len(results) > 0) else None              # 取 boxes

        # ===== 寫 txt（逐幀遞增；無偵測也寫） =====
        if txt_file is not None:                                                          # 有檔案才寫
            if boxes is not None and boxes.xywh is not None and len(boxes.xywh) > 0:      # 有偵測
                for xywh in boxes.xywh:                                                   # 逐框（max_det=1）
                    x_c, y_c, w_b, h_b = xywh                                             # 中心與寬高
                    frame_count_for_detect += 1                                           # 幀+1
                    txt_file.write(f"{frame_count_for_detect},{x_c},{y_c},{w_b},{h_b}\n") # 寫一行
            else:
                frame_count_for_detect += 1                                               # 幀+1
                txt_file.write(f"{frame_count_for_detect},no detection\n")                # 無偵測

        # ===== 畫到 UI 幀（不寫檔） =====
        try:
            vis_frame = results[0].plot()                                                 # Ultralytics 內建疊圖（僅 UI）
        except Exception:
            vis_frame = frame.copy()                                                      # 保底：用原幀

    # ===== 畫 FPS 並顯示到 Qt（只顯示 UI，不影響檔案） =====
    cv2.putText(vis_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,         # FPS 文字
                1, (0, 255, 0), 2, cv2.LINE_AA)                                           # 樣式
    frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)                                # BGR→RGB
    h, w, ch = vis_frame.shape                                                            # 尺寸
    qimg = QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)           # QImage
    scale_qpix = QtGui.QPixmap.fromImage(qimg).scaled(                                    # 縮放
        label.width(), label.height(), QtCore.Qt.KeepAspectRatio,                         # 等比
        QtCore.Qt.SmoothTransformation)                                                   # 平滑
    label.setPixmap(scale_qpix)                                                           # 顯示

    # ===== 多執行緒同步 =====
    barrier.wait()                                                                        # 柵欄

    # ===== 結束立即收尾（避免 pending） =====
    if not recording_sig:                                                                 # 停止錄影
        frame_count_for_detect = 0                                                        # 幀計歸零
        if txt_file is not None:                                                          # 關 txt
            txt_file.close()                                                              # 關閉
            txt_file = None                                                               # 置空
        if out is not None:                                                               # 關 writer
            out.release()                                                                 # 釋放
            out = None                                                                    # 置空
        original_out = None                                                               # 不使用 original
        save_sig = False                                                                  # 清旗標

    return start_time, frame_count, fps, out, original_out, frame_count_for_detect, save_sig, txt_file  # 回傳相容

def squat_bone_loop(i, frame, label, save_sig, recording_sig, folder,                      # 骨架視角（cam2）
                    start_time, frame_count, fps, out, original_out,                       # original_out 不用，保留相容
                    model, txt_file, frame_count_for_detect,                               # 狀態/推論
                    skeleton_connections, barrier):                                        # 連線/同步
    # ===== FPS =====
    frame_count += 1                                                                       # 幀+1
    elapsed_time = time.time() - start_time                                                # 距上次
    if elapsed_time >= 1:                                                                  # 每秒
        fps = frame_count / elapsed_time                                                   # FPS
        frame_count = 0                                                                    # 歸零
        start_time = time.time()                                                           # 重置

    # ===== 旋轉與乾淨/可視幀 =====
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)                                     # 旋轉 90°
    clean_frame = frame.copy()                                                             # 乾淨幀（寫檔）
    vis_frame   = frame.copy()                                                             # 可視幀（UI 疊圖）

    # ===== 錄影：只寫乾淨畫面 =====
    if recording_sig:                                                                      # 錄影中
        if out is None:                                                                    # 初始化 writer
            path = os.path.join(folder, f'vision{i+1}.avi')                                # 檔名
            h, w = frame.shape[:2]                                                         # 幀高寬
            out = cv2.VideoWriter(path, FOURCC_MJPG, REC_FPS, (w, h))                      # MJPG writer
        out.write(clean_frame)                                                             # 先寫乾淨畫面

        if txt_file is None:                                                               # 初始化 txt（一次）
            txt_path = os.path.join(folder, 'mediapipe_landmarks.txt')                     # 骨架輸出
            txt_file = open(txt_path, "w", buffering=1)                                    # 行緩衝
            frame_count_for_detect = 0                                                     # 幀計數歸零

    # ===== 骨架推論：畫到 UI 幀、寫到 txt（不寫回檔案） =====
    kpts_xy = None                                                                         # 預設無點
    if recording_sig:                                                                      # 錄影時才推論
        with inference_lock:                                                               # 序列化避免搶 GPU
            results = list(model(source=frame, stream=True, verbose=False))                # YOLO 骨架推論
        if results and results[0].keypoints:                                               # 有偵測
            try:
                kpts = results[0].keypoints[0]                                             # 取第一個人
                kpts_xy = kpts.xy                                                          # (1,K,2)
            except Exception:
                kpts_xy = None                                                             # 容錯

        # ===== txt：逐幀遞增；無偵測也寫 =====
        if txt_file is not None:                                                           # 檔案存在
            frame_count_for_detect += 1                                                    # 幀+1
            wrote = False                                                                  # 是否寫點
            if kpts_xy is not None and hasattr(kpts_xy, "shape") and len(kpts_xy.shape) == 3 and kpts_xy.shape[0] >= 1:  # 形狀檢查
                for idx, kp in enumerate(kpts_xy[0]):                                      # 逐關節
                    x_kp, y_kp = int(kp[0].item()), int(kp[1].item())                      # 轉 int
                    txt_file.write(f"{frame_count_for_detect},{idx},{x_kp},{y_kp}\n")      # 寫一行
                    wrote = True                                                           # 標記
            if not wrote:                                                                  # 無點
                txt_file.write(f"{frame_count_for_detect},no detection\n")                 # 無偵測

        # ===== UI：畫點與連線到 vis_frame（不寫回檔案） =====
        if kpts_xy is not None and hasattr(kpts_xy, "shape") and len(kpts_xy.shape) == 3 and kpts_xy.shape[0] >= 1:  # 有點
            kp_coords = []                                                                 # 座標列表
            for kp in kpts_xy[0]:                                                          # 逐關節
                x_kp, y_kp = int(kp[0].item()), int(kp[1].item())                          # 座標
                if x_kp == 0 and y_kp == 0:                                                # 無效點
                    kp_coords.append(None)                                                 # 記 None
                    continue                                                               # 跳過畫點
                kp_coords.append((x_kp, y_kp))                                             # 記有效點
                cv2.circle(vis_frame, (x_kp, y_kp), 5, (0, 255, 0), cv2.FILLED)            # 畫關節點
            for s_idx, e_idx in skeleton_connections:                                      # 逐邊
                if s_idx < len(kp_coords) and e_idx < len(kp_coords):                      # 邊界檢查
                    ps, pe = kp_coords[s_idx], kp_coords[e_idx]                            # 端點
                    if ps is None or pe is None:                                           # 任一無效
                        continue                                                           # 不畫線
                    cv2.line(vis_frame, ps, pe, (0, 255, 255), 2)                           # 畫連線

    # ===== Qt 顯示（只顯示 UI，不影響檔案） =====
    cv2.putText(vis_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,          # FPS
                1, (0, 255, 0), 2, cv2.LINE_AA)                                            # 樣式
    frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)                                 # BGR→RGB
    h, w, ch = vis_frame.shape                                                             # 尺寸
    qimg = QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)            # QImage
    scale_qpix = QtGui.QPixmap.fromImage(qimg).scaled(                                     # 縮放
        label.width(), label.height(), QtCore.Qt.KeepAspectRatio,                          # 等比
        QtCore.Qt.SmoothTransformation)                                                    # 平滑
    label.setPixmap(scale_qpix)                                                            # 顯示

    # ===== 同步 =====
    barrier.wait()                                                                         # 柵欄

    # ===== 收尾（避免 pending） =====
    if not recording_sig:                                                                  # 停止錄影
        frame_count_for_detect = 0                                                         # 幀計歸零
        if txt_file is not None:                                                           # 關 txt
            txt_file.close()                                                               # 關閉
            txt_file = None                                                                # 置空
        if out is not None:                                                                # 關 writer
            out.release()                                                                  # 釋放
            out = None                                                                     # 置空
        original_out = None                                                                # 不使用 original
        save_sig = False                                                                   # 清旗標

    return start_time, frame_count, fps, out, original_out, frame_count_for_detect, save_sig, txt_file  # 回傳相容

def squat_general_loop(i, frame, label, save_sig, recording_sig, folder,                  # 一般視角（cam3~6）
                       start_time, frame_count, fps, out, barrier):                       # writer／柵欄
    # ---- FPS 計算 ----                                                                   # FPS
    frame_count += 1                                                                     # 幀+1
    elapsed_time = time.time() - start_time                                              # 距上次秒數
    if elapsed_time >= 1:                                                                # 每秒刷新
        fps = frame_count / elapsed_time                                                 # FPS
        frame_count = 0                                                                  # 歸零
        start_time = time.time()                                                         # 重置

    # ---- 旋轉 ----                                                                         # 旋轉
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)                                   # 順時針 90°

    # ---- 錄影（MJPG） ----                                                                  # 錄影
    if recording_sig:                                                                    # 若在錄影
        if out is None:                                                                  # 初始化 writer
            path = os.path.join(folder, f'vision{i+1}.avi')                              # 檔名
            h, w = frame.shape[:2]                                                       # 高寬
            out = cv2.VideoWriter(path, FOURCC_MJPG, REC_FPS, (w, h))                    # 建立 writer
        out.write(frame)                                                                 # 寫一幀

    # ---- 同步（安全：仍每幀 wait） ----                                                          # 柵欄
    barrier.wait()                                                                       # 每幀同步

    # ---- 錄影結束處理 ----                                                                    # 收尾
    if not recording_sig:                                                                # 未錄影
        if out is not None:                                                              # 若 writer 存在
            out.release()                                                                # 關閉
            out = None                                                                   # 置空
        save_sig = False                                                                 # 清旗標

    # ---- 顯示到 Qt ----                                                                       # 顯示
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,            # 畫 FPS
                1, (0, 255, 0), 2, cv2.LINE_AA)                                          # 樣式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                   # 顏色轉換
    h, w, ch = frame.shape                                                               # 尺寸
    qimg = QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)          # QImage
    scale_qpix = QtGui.QPixmap.fromImage(qimg).scaled(                                   # 縮放
        label.width(), label.height(), QtCore.Qt.KeepAspectRatio,                        # 等比
        QtCore.Qt.SmoothTransformation)                                                  # 平滑
    label.setPixmap(scale_qpix)                                                          # 顯示
    return start_time, frame_count, fps, out, save_sig                                   # 回傳


# benchpress
# ====== 緩衝常數（可依需求調整）======
BODY_BUF_FRAMES = 20                     # 人體偵測命中/未命中緩衝幀數                       # 遲滯
START_LATCH_FRAMES = 10                  # 三 Gate 必須連續命中 N 幀才允許『開始錄影』        # 防抖（只影響開段）
BAR_LOSS_TOL_FRAMES = 20                 # 槓暫時偵測不到時可容忍的連續幀數                   # 偵測遺失容忍
END_GRACE_FRAMES = 30                    # Gate 轉 False 後需連續幀數才真正結束分段           # 關檔緩衝
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

def _start_segment_writers(folder, i, seg_no, frame, need_original, need_txt, txt_suffix):  # 開啟暫存 writer 與 txt
    import os, cv2  # 檔案與影像
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPG 編碼
    size = (frame.shape[1], frame.shape[0])  # 幀尺寸
    tmp = {}  # 暫存路徑字典
    out = None  # 疊圖 writer
    original_out = None  # 原始 writer
    txt_file = None  # txt 物件

    if _safe_folder(folder):  # 若最終資料夾可用則直接寫入
        base_dir = folder  # 目的地
        _ensure_dir(base_dir)  # 確保存在
        prefix = os.path.join(base_dir, f"_staging_cam{i}_seg{seg_no:03d}")  # 統一前綴
    else:  # 否則寫到暫存區
        base_dir = _staging_dir_for_cam(i)  # cam 專屬暫存
        prefix = os.path.join(base_dir, f"_staging_cam{i}_seg{seg_no:03d}")  # 暫存前綴

    if need_original:  # 原始影片
        tmp['o'] = f"{prefix}_original.avi"  # 原始檔名
        original_out = cv2.VideoWriter(tmp['o'], fourcc, 29, size)  # 建立原始 writer

    tmp['v'] = f"{prefix}_vision.avi"  # 疊圖檔名
    out = cv2.VideoWriter(tmp['v'], fourcc, 29, size)  # 建立疊圖 writer

    if need_txt:  # 需要 txt 就建
        tmp['t'] = f"{prefix}_{txt_suffix}.txt"  # txt 檔名
        txt_file = open(tmp['t'], 'w', encoding='utf-8')  # 開啟 txt

    return out, original_out, txt_file, tmp  # 回傳 I/O 與路徑


def _close_io(out=None, original_out=None, txt_file=None):                                    # 關閉 I/O
    if txt_file is not None: txt_file.close()                                                 # 關 txt
    if out is not None: out.release()                                                         # 關疊圖
    if original_out is not None: original_out.release()                                       # 關原始
    return None                                                                               # 無回傳

def _end_and_move(folder, i, seg_no, tmp_paths, mapping):                                     # 結束段落、建立最終資料夾並搬檔  #
    import os, shutil, time                                                                     # 檔案與時間  #
    end_ts = time.strftime("%Y%m%d_%H%M%S")                                                     # 以結束時刻命名  #
    # ==== 決定最終根目錄：優先使用呼叫端傳入的 folder，否則就用 _final_root() ====
    try:
        base_root = folder if _safe_folder(folder) else _final_root()                           # 目的根目錄  #
    except Exception:
        base_root = _final_root()                                                               # 防呆退回 _FINAL_BASE_DIR  #
    # ==== 最終資料夾命名：recording_YYYYMMDD_HHMMSS（不加 seg 編號） ====
    rec_folder = os.path.join(base_root, f"recording_{end_ts}")                                 # 最終錄影資料夾  #
    os.makedirs(rec_folder, exist_ok=True)                                                      # 確保存在  #
    # ==== 逐檔搬移並改名 ====
    for k, new_name in mapping.items():                                                         # 逐檔搬移  #
        p = tmp_paths.get(k)                                                                    # 暫存檔路徑  #
        if p and os.path.exists(p):                                                             # 檔案存在才搬  #
            dst = os.path.join(rec_folder, new_name)                                            # 目標完整路徑  #
            if os.path.exists(dst):                                                             # 若已存在先刪  #
                try: os.remove(dst)                                                             # 刪除舊檔  #
                except Exception: pass                                                          # 忽略刪除失敗  #
            shutil.move(p, dst)                                                                 # 搬到最終資料夾  #
    print(f"[SEG] End SEG {seg_no:03d} on cam{i+1} -> {rec_folder}")                            # 紀錄路徑  #
    return rec_folder                                                                            # 回傳最終資料夾  #

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

def _debounce_three_gate(is_rec, gate_ui, gate_body, gate_bar,                              # 是否已在錄、三個 gate 當幀值
                         shared_state, shared_lock, gate3_cnt_key):                         # 共享狀態與計數 key
    three_now = bool(gate_ui and gate_body and gate_bar)                                    # 本幀三 Gate 是否同時 True
    cnt = _shared_get(shared_state, shared_lock, gate3_cnt_key, 0)                          # 取連續命中計數
    cnt = cnt + 1 if three_now else 0                                                       # 命中 +1；任一掉則歸零
    _shared_set_many(shared_state, shared_lock, {gate3_cnt_key: cnt})                       # 回寫最新計數
    # 尚未在錄 → 需要連續 START_LATCH_FRAMES 幀才允許開錄；已在錄 → 直接看本幀三 Gate 是否維持
    should_record = (is_rec and three_now) or ((not is_rec) and (cnt >= START_LATCH_FRAMES))# 防抖判斷
    return should_record                                                                    # 回傳是否應該錄影


# ============================== 進一步共用工具 ==============================

def _idle_if_ui_off(gate_ui, label, frame, fps, barrier,
                    shared_state, shared_lock, cam_rec_key,
                    io_tuple, counters_tuple):                                               # UI 關時統一收尾  # 用在三視角
    # io_tuple = (out, original_out, txt_file)                                               # I/O 三件組
    # counters_tuple = (save_sig, frame_count_for_detect)                                    # 計數雙件組
    out, original_out, txt_file = io_tuple                                                    # 解包 I/O
    save_sig, frame_count_for_detect = counters_tuple                                         # 解包計數
    if not gate_ui:                                                                           # UI 未按錄影直接收
        _close_io(out, original_out, txt_file)                                                # 關 I/O
        _shared_set_many(shared_state, shared_lock, {cam_rec_key: False})                     # 標記不在錄
        _qt_show(label, frame, fps)                                                           # 顯示
        barrier.wait()                                                                        # 同步
        return True, (None, None, None), (False, 0)                                           # 早退 + 重置回傳
    return False, io_tuple, counters_tuple                                                    # 繼續 + 原封不動

def _segment_start_if_needed(should_record, is_rec, folder, i, seg_no,
                             frame, need_original, need_txt, txt_suffix,
                             shared_state, shared_lock, cam_seg_key, cam_rec_key,
                             end_false_key, tmp_paths_key):                                   # 開段統一化
    if should_record and not is_rec:                                                          # False→True
        seg_no += 1                                                                           # 段+1
        out, original_out, txt_file, tmp_paths = _start_segment_writers(                      # 開 writer + 路徑
            folder, i, seg_no, frame, need_original, need_txt, txt_suffix                    # 依需求
        )
        _shared_set_many(shared_state, shared_lock, {                                         # 回寫狀態
            cam_seg_key: seg_no, cam_rec_key: True, end_false_key: 0, tmp_paths_key: tmp_paths
        })
        return True, seg_no, out, original_out, txt_file                                     # 有開段
    return False, seg_no, None, None, None                                                   # 沒開段

def _segment_end_if_needed(should_record, is_rec, end_false_cnt,
                           END_GRACE_FRAMES, out, original_out, txt_file,
                           shared_state, shared_lock, tmp_paths_key, cam_rec_key,
                           folder, i, seg_no, mapping, *,
                           end_false_key, reset_frame_counter=False):  # ← 加上 * 後面都要用命名參數
    ended = False                                                                            # 預設未結束
    if not should_record:                                                                    # Gate 為 False
        end_false_cnt = min(END_GRACE_FRAMES, end_false_cnt + 1)                             # False 累 +1
        _shared_set_many(shared_state, shared_lock, {end_false_key: end_false_cnt})          # 回寫累計
        if is_rec and end_false_cnt >= END_GRACE_FRAMES:                                     # 達緩衝 → 結束一段
            _close_io(out, original_out, txt_file)                                           # 關 I/O
            paths = _shared_get(shared_state, shared_lock, tmp_paths_key, {})                # 取暫存路徑
            _shared_set_many(shared_state, shared_lock, {cam_rec_key: False, tmp_paths_key: {}})  # 清狀態
            _end_and_move(folder, i, seg_no, paths, mapping)                                 # 搬檔改名
            ended = True                                                                     # 標記結束
        if reset_frame_counter and not is_rec:                                               # 若沒在錄且需要歸零
            return ended, end_false_cnt, 0                                                   # 回傳並把幀數清零
    else:                                                                                    # Gate 為 True（錄影中）
        _shared_set_many(shared_state, shared_lock, {end_false_key: 0})                      # 清 False 緩衝
    return ended, end_false_cnt, None                                                        # 無需重設幀數則回 None

def benchpress_bar_loop(i, frame, label, save_sig, folder,                                     # 槓視角主迴圈  # 說明
                        start_time, frame_count, fps, out, original_out, model,               # FPS/Writer/Model  # 說明
                        txt_file, frame_count_for_detect, barrier,                             # TXT/段內幀/同步  # 說明
                        shared_state, shared_lock, BAR_MOVE_THRESH):                           # 共享狀態/鎖/位移門檻  # 說明
    import cv2                                                                                # 影像處理  # 說明

    # -------- 共用 key --------
    cam_rec_key   = f"rec_cam{i}"                                                             # 是否在錄 key  # 說明
    cam_seg_key   = f"seg_cam{i}"                                                             # 段號 key  # 說明
    end_false_key = f"rec_end_false_cam{i}"                                                   # Gate False 緩衝 key  # 說明
    tmp_paths_key = f"tmp_paths_cam{i}"                                                       # 暫存路徑 key  # 說明

    # -------- FPS 更新 --------
    start_time, frame_count, fps = _update_fps(start_time, frame_count, fps)                  # 每秒刷新 FPS  # 說明

    # -------- 狀態讀取 --------
    # benchpress_bar_loop 內的 UI gate 讀取
    gate_ui   = _shared_get(shared_state, shared_lock, "auto_recording_sig", False)           # UI Gate 改讀 auto_recording_sig
    is_rec    = _shared_get(shared_state, shared_lock, cam_rec_key, False)                    # 是否在錄  # 說明
    seg_no    = _shared_get(shared_state, shared_lock, cam_seg_key, 0)                        # 段號  # 說明
    end_false_cnt = _shared_get(shared_state, shared_lock, end_false_key, 0)                  # False 緩衝幀  # 說明

    # -------- UI 未啟動 → 早退 --------
    early, (out, original_out, txt_file), (save_sig, frame_count_for_detect) = \
        _idle_if_ui_off(gate_ui, label, frame, fps, barrier,                                  # UI 關就收尾  # 說明
                        shared_state, shared_lock, cam_rec_key,
                        (out, original_out, txt_file), (save_sig, frame_count_for_detect))
    if early:                                                                                 # 若早退  # 說明
        return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file  # 回傳  # 說明

    # -------- 保留原始幀 --------
    try:
        original_frame = frame.copy()                                                         # 乾淨原圖  # 說明
    except Exception:
        original_frame = frame                                                                # 退化保護  # 說明

    # -------- YOLO 推論 + 疊圖 --------
    try:
        results = model.predict(source=frame, imgsz=320, conf=0.5, verbose=False)             # YOLO 推論  # 說明
    except Exception as e:
        results = []                                                                          # 失敗視為無偵測  # 說明
        print(f"[benchpress_bar_loop] model error: {e}")                                      # 紀錄錯誤  # 說明

    xywh = _yolo_first_box_xywh(results)                                                      # 取第一個框中心 xywh  # 說明
    for r in results:
        try:
            frame = r.plot()                                                                  # 疊框到顯示層  # 說明
        except Exception:
            pass                                                                              # 疊圖錯誤忽略  # 說明

    # -------- 小框設定（只在框內判定位移） --------
    GATE_X1, GATE_X2 = 420, 500                                                               # 觸發區 X 範圍  # 說明
    GATE_Y1, GATE_Y2 = 160, 225                                                               # 觸發區 Y 範圍  # 說明
    cv2.rectangle(frame, (GATE_X1, GATE_Y1), (GATE_X2, GATE_Y2), (0, 0, 255), 2)              # 畫紅框供校對  # 說明

    # -------- session 狀態（出槓一路錄；回框穩定停住才關） --------
    session_key   = "bar_session_active"                                                      # 出槓鎖存（一路錄）  # 說明
    back_idle_key = "bar_back_idle_cnt"                                                       # 回框且未達門檻之累計  # 說明
    prev_in_key   = "prev_bar_in_gate"                                                        # 上一幀是否在框內  # 說明
    bar_session_active = _shared_get(shared_state, shared_lock, session_key, False)           # 本段是否 active  # 說明
    bar_back_idle_cnt  = _shared_get(shared_state, shared_lock, back_idle_key, 0)             # 回框無位移累計  # 說明
    prev_in_gate       = _shared_get(shared_state, shared_lock, prev_in_key, None)            # 上一幀 in/out  # 說明

    prev_x = _shared_get(shared_state, shared_lock, "prev_bar_x", None)                       # 上一幀 x  # 說明
    prev_y = _shared_get(shared_state, shared_lock, "prev_bar_y", None)                       # 上一幀 y  # 說明
    bar_loss_key = f"bar_loss_cnt_cam{i}"                                                     # 遺失計數 key  # 說明
    bar_loss = _shared_get(shared_state, shared_lock, bar_loss_key, 0)                        # 當前遺失  # 說明

    in_gate = False                                                                           # 本幀是否在框內  # 說明
    moved   = False                                                                           # 本幀是否達位移門檻（僅在框內有效）  # 說明

    if xywh is not None:                                                                      # 有偵測到槓  # 說明
        x, y, w, h = xywh                                                                     # 解析 xywh  # 說明
        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), cv2.FILLED)                       # 畫中心點  # 說明

        in_gate = (GATE_X1 <= x <= GATE_X2) and (GATE_Y1 <= y <= GATE_Y2)                     # 是否在框內  # 說明
        cv2.putText(frame, "IN" if in_gate else "OUT", (GATE_X1, GATE_Y1-8),                  # IN/OUT 提示  # 說明
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

        # 先處理「剛回框」的基準重設，避免回框第一幀誤判 moved=True
        if prev_in_gate is False and in_gate:                                                 # 外→內 的轉換  # 說明
            _shared_set_many(shared_state, shared_lock, {"prev_bar_x": x, "prev_bar_y": y})   # 基準重設  # 說明
            prev_x, prev_y = x, y                                                             # 同步本地  # 說明

        # 只有「在框內」才計算位移門檻（出槓觸發、回槓關檔都是看框內位移）
        if in_gate and (prev_y is not None):                                                  # 框內且有上一幀  # 說明
            moved = (abs(y - prev_y) >= BAR_MOVE_THRESH)                                      # 是否過門檻  # 說明

        # ===== 狀態機：以「框內位移門檻」觸發/關閉 =====
        if not bar_session_active:                                                            # 尚未出槓  # 說明
            if in_gate and moved:                                                             # ★ 框內達門檻 → 出槓  # 說明
                bar_session_active = True                                                     # 鎖存整段 active  # 說明
                bar_back_idle_cnt = 0                                                         # 清回框累計  # 說明
        else:                                                                                 # 已出槓（active）  # 說明
            if in_gate:                                                                       # 回到框內  # 說明
                if not moved:                                                                 # 未達位移門檻  # 說明
                    bar_back_idle_cnt = min(END_GRACE_FRAMES, bar_back_idle_cnt + 1)          # 無位移累+1  # 說明
                else:
                    bar_back_idle_cnt = 0                                                     # 有動就清零  # 說明
                if bar_back_idle_cnt >= END_GRACE_FRAMES:                                     # 框內穩定停住  # 說明
                    bar_session_active = False                                                # 結束本段  # 說明
                    bar_back_idle_cnt = 0                                                     # 歸零  # 說明
            else:
                bar_back_idle_cnt = 0                                                         # 框外不累計關段條件  # 說明

        # 更新上一幀座標與遺失
        _shared_set_many(shared_state, shared_lock, {"prev_bar_x": x, "prev_bar_y": y})       # 更新 prev_*  # 說明
        bar_loss = 0                                                                          # 清遺失  # 說明

    else:
        in_gate = False                                                                       # 視為在框外  # 說明
        bar_loss = min(BAR_LOSS_TOL_FRAMES+1, bar_loss + 1)                                   # 遺失累計  # 說明
        cv2.putText(frame, "OUT", (GATE_X1, GATE_Y1-8),                                       # 標示 OUT  # 說明
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

    # -------- 回寫 bar 狀態（供其他視角/三 Gate 使用） --------
    _shared_set_many(shared_state, shared_lock, {
        "bar_y_changed": bar_session_active,                                                  # bar Gate = session active  # 說明
        session_key: bar_session_active,                                                      # 存 session  # 說明
        back_idle_key: bar_back_idle_cnt,                                                     # 存回框無位移累計  # 說明
        prev_in_key: in_gate,                                                                 # 存本幀 in/out  # 說明
        bar_loss_key: bar_loss                                                                # 存遺失計數  # 說明
    })

    # -------- 三 Gate 決策（UI + 人體骨架 + 槓） --------
    gate3_cnt_key = f"gate3_true_cnt_cam{i}"                                                    # 三 Gate 連續命中計數 key
    gate_body = _shared_get(shared_state, shared_lock, "body_detected", False)                  # 人體 gate 取自 body_loop
    # bar 這段的 gate 是你的 bar_session_active（已在上文計算）
    should_record = _debounce_three_gate(                                                       # 呼叫共用防抖函式
        is_rec, gate_ui, gate_body, bar_session_active,                                         # 三 Gate 值
        shared_state, shared_lock, gate3_cnt_key                                                # 共享與計數 key
    )


    # -------- 視段開啟（只建暫存 writer/txt） --------
    opened, seg_no, out_new, original_new, txt_new = _segment_start_if_needed(                # 視需要開段  # 說明
        should_record, is_rec, folder, i, seg_no, frame,
        need_original=True, need_txt=True, txt_suffix="bar",                                  # 槓端要原始+txt  # 說明
        shared_state=shared_state, shared_lock=shared_lock,
        cam_seg_key=cam_seg_key, cam_rec_key=cam_rec_key,
        end_false_key=end_false_key, tmp_paths_key=tmp_paths_key)
    if opened:                                                                                # 剛開段  # 說明
        out, original_out, txt_file = out_new, original_new, txt_new                          # 接手 I/O  # 說明
        frame_count_for_detect = 0                                                            # 段內幀歸零  # 說明

    # -------- 寫入 / 結束控制 --------
    if should_record:                                                                         # 錄影中  # 說明
        if original_out is not None:
            try: original_out.write(original_frame)                                           # 寫原始  # 說明
            except Exception as e: print(f"[benchpress_bar_loop] original write err: {e}")    # 保護  # 說明
        if out is not None:
            out.write(frame)                                                                  # 寫疊圖  # 說明

        frame_count_for_detect += 1                                                           # 段內幀+1  # 說明
        if txt_file is not None:                                                              # 寫座標 TXT  # 說明
            if xywh is not None:
                x, y, w, h = xywh                                                             # 解析  # 說明
                txt_file.write(f"{frame_count_for_detect},{x},{y},{w},{h}\n")                 # 記錄 xywh  # 說明
            else:
                txt_file.write(f"{frame_count_for_detect},no detection\n")                    # 無偵測  # 說明

        _shared_set_many(shared_state, shared_lock, {end_false_key: 0})                       # 清 False 緩衝  # 說明

    else:                                                                                     # Gate False → 可能觸發關段  # 說明
        ended, end_false_cnt, frame_reset = _segment_end_if_needed(                           # 判斷關段  # 說明
            should_record, is_rec, end_false_cnt, END_GRACE_FRAMES,
            out, original_out, txt_file,
            shared_state, shared_lock, tmp_paths_key, cam_rec_key,
            folder, i, seg_no,
            mapping={"o": "original_vision1.avi", "v": "vision1.avi", "t": "yolo_coordinates.txt"},  # 搬檔命名  # 說明
            end_false_key=end_false_key,                                                      # 必帶  # 說明
            reset_frame_counter=True)                                                         # 關段後可歸零  # 說明
        if ended:                                                                             # 若關段  # 說明
            out, original_out, txt_file = None, None, None                                    # 釋放 I/O  # 說明
        if frame_reset is not None:                                                           # 需要歸零  # 說明
            frame_count_for_detect = frame_reset                                              # 段內幀歸零  # 說明

    # -------- 顯示與同步 --------
    _qt_show(label, frame, fps)                                                               # 疊 FPS 並顯示  # 說明
    barrier.wait()                                                                            # 多執行緒同步  # 說明

    # -------- 回傳（介面一致） --------
    return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file  # 回傳  # 說明


def benchpress_body_loop(i, frame, label, save_sig, folder,                                   # 人體視角主迴圈（攝影機 i、畫面、UI元件、旗標、輸出資料夾）  # 說明
                         start_time, frame_count, fps, out, model, txt_file,                  # FPS 與 Writer/Model/TXT 句柄（維持介面一致）  # 說明
                         frame_count_for_detect, skeleton_connections, barrier,               # 錄影內幀計數 / 骨架連線 / 執行緒柵欄  # 說明
                         shared_state, shared_lock):                                          # 共享狀態與鎖  # 說明
    # ============== 可調參數 ==============
    BOX_CONF_TH   = 0.60                                                                      # 人框信心值下限  # 說明
    AREA_MIN_RATE = 0.02                                                                      # 人框佔畫面最小比例  # 說明
    AREA_MAX_RATE = 0.90                                                                      # 人框佔畫面最大比例  # 說明
    EDGE_PX_TH    = 6                                                                         # 邊界像素閾值（防角落假點）  # 說明
    EDGE_RATE_MAX = 0.25                                                                      # 邊緣點比例上限  # 說明
    MIN_VALID_KP  = 4                                                                         # 最少有效關鍵點數  # 說明
    INDEX_MAP     = [1,0,3,2,5,4,7,6]                                                         # 左右對調（符合你畫線邏輯）  # 說明

    # ---- ROI（方法 A：直接裁切 ROI） ----
    ROI_X1_RATE, ROI_Y1_RATE = 0.20, 0.15                                                     # ROI 左上相對座標  # 說明
    ROI_X2_RATE, ROI_Y2_RATE = 0.80, 0.80                                                     # ROI 右下相對座標  # 說明
    DRAW_ROI = True                                                                           # 除錯時畫 ROI 框  # 說明

    # ---- 常用 key ----
    cam_rec_key   = f"rec_cam{i}"                                                             # 是否在錄影 key  # 說明
    cam_seg_key   = f"seg_cam{i}"                                                             # 段號 key  # 說明
    end_false_key = f"rec_end_false_cam{i}"                                                   # 關段緩衝計數 key  # 說明
    tmp_paths_key = f"tmp_paths_cam{i}"                                                       # 暫存路徑 key  # 說明
    hit_key       = f"body_hit_cnt_cam{i}"                                                    # 命中幀數 key  # 說明
    miss_key      = f"body_miss_cnt_cam{i}"                                                   # 未命中幀數 key  # 說明
    orig_wr_key   = f"original_out_cam{i}"                                                    # 原始 writer 句柄 key  # 說明

    # ---- FPS 更新 ----
    start_time, frame_count, fps = _update_fps(start_time, frame_count, fps)                  # 刷新 FPS 計算  # 說明

    # ---- 讀共享狀態 ----
    gate_ui   = _shared_get(shared_state, shared_lock, "auto_recording_sig", False)           # UI Gate 改讀 auto_recording_sig
    is_rec    = _shared_get(shared_state, shared_lock, cam_rec_key, False)                    # 是否在錄影中  # 說明
    seg_no    = _shared_get(shared_state, shared_lock, cam_seg_key, 0)                        # 當前段號  # 說明
    end_false_cnt = _shared_get(shared_state, shared_lock, end_false_key, 0)                  # 關段緩衝幀數  # 說明
    hit_cnt   = _shared_get(shared_state, shared_lock, hit_key, 0)                            # 命中幀數  # 說明
    miss_cnt  = _shared_get(shared_state, shared_lock, miss_key, 0)                           # 未命中幀數  # 說明
    original_out = _shared_get(shared_state, shared_lock, orig_wr_key, None)                  # 原始 writer 取出  # 說明

    # ---- UI 未開啟時的早退收尾 ----
    early, (out, _unused_original, txt_file), (save_sig, frame_count_for_detect) = \
        _idle_if_ui_off(gate_ui, label, frame, fps, barrier,                                  # 呼叫保護：UI 關就早退  # 說明
                        shared_state, shared_lock, cam_rec_key,
                        (out, None, txt_file), (save_sig, frame_count_for_detect))
    if early:                                                                                 # 若早退  # 說明
        return start_time, frame_count, fps, out, frame_count_for_detect, save_sig, txt_file  # 直接回傳結束  # 說明

    # ---- 保留原始畫面（給 original_vision2） ----
    try:
        original_frame = frame.copy()                                                         # 複製原圖  # 說明
    except Exception:
        original_frame = frame                                                                # 退化處理  # 說明

    # ---- ROI 轉 px 並可視化 ----
    import cv2                                                                                # 匯入繪圖/影像庫  # 說明
    H, W = frame.shape[:2]                                                                    # 取得畫面大小  # 說明
    rx1, ry1 = int(W*ROI_X1_RATE), int(H*ROI_Y1_RATE)                                         # ROI 左上 px  # 說明
    rx2, ry2 = int(W*ROI_X2_RATE), int(H*ROI_Y2_RATE)                                         # ROI 右下 px  # 說明
    if DRAW_ROI:                                                                              # 是否畫 ROI 框  # 說明
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)                        # 在顯示層畫黃框  # 說明

    # ================== 核心改動：裁切 ROI → 推論 → 加回偏移 ==================
    roi_img = original_frame[ry1:ry2, rx1:rx2].copy()                                         # 直接裁切 ROI 影像  # 說明
    try:
        results = list(model(source=roi_img, stream=True, verbose=False))                     # 只在 ROI 圖上跑 YOLO（較快且乾淨）  # 說明
    except Exception as e:
        results = []                                                                          # 發生錯誤視為無偵測  # 說明
        print(f"[benchpress_body_loop] model error: {e}")                                     # 列印錯誤  # 說明

    # ---- 小工具：轉 numpy 與加偏移 ----
    def _to_numpy(x):                                                                         # 將 tensor 轉 numpy  # 說明
        if hasattr(x, "detach"):                                                              # 若是 torch tensor  # 說明
            return x.detach().cpu().numpy()                                                   # 轉 numpy  # 說明
        return x                                                                              # 已是 numpy 則直接回傳  # 說明

    def _offset_kps(xy_arr, offx, offy):                                                      # kpoints 陣列整體加偏移  # 說明
        xy = _to_numpy(xy_arr)                                                                # 轉 numpy  # 說明
        xy[..., 0] += offx                                                                    # x 加 rx1 偏移  # 說明
        xy[..., 1] += offy                                                                    # y 加 ry1 偏移  # 說明
        return xy                                                                             # 回傳偏移後座標  # 說明

    def _offset_box_xyxy(box_xyxy, offx, offy):                                               # 單一框加偏移  # 說明
        x1, y1, x2, y2 = [float(v) for v in box_xyxy[0].tolist()]                             # 取 xyxy  # 說明
        return (x1+offx, y1+offy, x2+offx, y2+offy)                                           # 全圖座標  # 說明

    # ---- 關鍵點解析與過濾 ----
    body_now = False                                                                          # 本幀是否有人  # 說明
    frame_points = None                                                                       # 本幀輸出點  # 說明

    if results and getattr(results[0], "keypoints", None) is not None and \
       getattr(results[0], "boxes", None) is not None:                                        # 需同時有 boxes 與 keypoints  # 說明
        det_boxes = results[0].boxes                                                          # 取出 boxes  # 說明
        det_kpts  = results[0].keypoints                                                      # 取出 keypoints  # 說明

        xy_all_roi = det_kpts.xy                                                              # ROI 座標系的 (N,K,2)  # 說明
        kconf      = getattr(det_kpts, "conf", None)                                          # (N,K) 或 None  # 說明
        if kconf is not None:                                                                 # 若存在 kp conf  # 說明
            kconf = _to_numpy(kconf)                                                          # 轉 numpy  # 說明

        xy_all = _offset_kps(xy_all_roi, rx1, ry1)                                            # 將所有 kps 加回全圖偏移  # 說明

        best_idx, best_score = -1, -1.0                                                       # 最佳框索引與分數  # 說明
        for idx in range(len(det_boxes)):                                                     # 逐框挑選  # 說明
            box = det_boxes[idx]                                                              # 第 idx 個框  # 說明
            conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0                        # 框信心值  # 說明
            if conf < BOX_CONF_TH:                                                            # 低於門檻略過  # 說明
                continue                                                                      # 跳過  # 說明
            if not hasattr(box, "xyxy"):                                                      # 無座標略過  # 說明
                continue                                                                      # 跳過  # 說明

            x1, y1, x2, y2 = _offset_box_xyxy(box.xyxy, rx1, ry1)                             # 框加回全圖偏移  # 說明
            bw, bh = max(0.0, x2-x1), max(0.0, y2-y1)                                         # 框寬高  # 說明
            area   = bw * bh                                                                   # 面積  # 說明
            rate   = area / float(W*H + 1e-6)                                                  # 佔比  # 說明
            if not (AREA_MIN_RATE <= rate <= AREA_MAX_RATE):                                  # 大小不合理  # 說明
                continue                                                                      # 跳過  # 說明

            # 取該人的所有關節點（已是全圖座標）
            if idx >= xy_all.shape[0]:                                                        # 越界保護  # 說明
                continue                                                                      # 跳過  # 說明
            kps = xy_all[idx]                                                                 # (K,2)  # 說明
            valid_mask = ~((kps[:,0] == 0) & (kps[:,1] == 0))                                 # 有效點遮罩  # 說明
            valid_cnt  = int(valid_mask.sum())                                                # 有效點數  # 說明
            if valid_cnt < MIN_VALID_KP:                                                      # 太少則跳過  # 說明
                continue                                                                      # 跳過  # 說明

            # 邊緣點比例過濾（全圖座標）
            edge_mask = (kps[:,0] < EDGE_PX_TH) | (kps[:,0] > (W-1-EDGE_PX_TH)) | \
                        (kps[:,1] < EDGE_PX_TH) | (kps[:,1] > (H-1-EDGE_PX_TH))               # 邊緣判定  # 說明
            edge_rate = float(edge_mask.sum()) / float(kps.shape[0])                          # 邊緣比例  # 說明
            if edge_rate > EDGE_RATE_MAX:                                                     # 邊緣點太多丟棄  # 說明
                continue                                                                      # 跳過  # 說明

            # 綜合分數：以 box conf 為主，kp conf 為輔
            score = conf                                                                      # 初始用框信心  # 說明
            if kconf is not None and idx < kconf.shape[0] and valid_cnt > 0:                  # 若有 kp conf  # 說明
                score = 0.7*conf + 0.3*float(kconf[idx][valid_mask].mean())                   # 加權  # 說明

            if score > best_score:                                                            # 更新最佳  # 說明
                best_score, best_idx = score, idx                                             # 紀錄  # 說明

        if best_idx >= 0:                                                                     # 找到合格人框  # 說明
            first = xy_all[best_idx]                                                          # 該人的 (K,2) 全圖座標  # 說明
            if first.shape[0] >= 8:                                                           # 至少 8 點  # 說明
                body_now = True                                                               # 記錄有人  # 說明
                reordered = first[INDEX_MAP, :2]                                              # 左右對調後的 8 點  # 說明
                frame_points = [(float(x), float(y)) for (x, y) in reordered]                 # 存成 list 供寫檔  # 說明

                # 視覺化（在顯示層的 frame 上）
                draw_pts = [(int(x), int(y)) for (x, y) in frame_points]                      # 轉 int 畫圖  # 說明
                if not skeleton_connections:                                                  # 若未提供連線  # 說明
                    skeleton_connections = [(0,1),(0,2),(1,3),(2,3),(4,6),(5,7),(0,4),(1,5)]  # 簡化骨架連線  # 說明
                for p in draw_pts: cv2.circle(frame, p, 5, (0,255,0), cv2.FILLED)             # 畫關節點  # 說明
                for a, b in skeleton_connections:                                             # 畫骨架線段  # 說明
                    if a < len(draw_pts) and b < len(draw_pts):
                        cv2.line(frame, draw_pts[a], draw_pts[b], (0,255,255), 2)             # 畫線  # 說明

    # ---- 人體 gate（遲滯） ----
    hit_cnt, miss_cnt, latched_now = _latch_by_buffer(hit_cnt, miss_cnt, body_now, BODY_BUF_FRAMES)  # 命中未命中緩衝  # 說明
    _shared_set_many(shared_state, shared_lock, {hit_key: hit_cnt, miss_key: miss_cnt})       # 回寫命中統計  # 說明
    _shared_set_many(shared_state, shared_lock, {"body_detected": latched_now})               # 更新人體 gate  # 說明

    # ---- 三 Gate 決策 ----
    gate3_cnt_key = f"gate3_true_cnt_cam{i}"                                                    # 三 Gate 連續命中計數 key
    gate_bar = _shared_get(shared_state, shared_lock, "bar_y_changed", False)                   # 槓 gate 取自 bar_loop
    # 這段的人體 gate 是 latched_now（上文已用 BODY_BUF_FRAMES 防抖）
    should_record = _debounce_three_gate(                                                       # 呼叫共用防抖函式
        is_rec, gate_ui, latched_now, gate_bar,                                                 # 三 Gate 值
        shared_state, shared_lock, gate3_cnt_key                                                # 共享與計數 key
    )

    # ---- 開段（建立疊圖 out + 原始 original_out + TXT） ----
    opened, seg_no, out_new, original_out_new, txt_new = _segment_start_if_needed(            # 進入錄影段  # 說明
        should_record, is_rec, folder, i, seg_no, frame,
        need_original=True, need_txt=True, txt_suffix="body",                                 # 要原始影片與 txt  # 說明
        shared_state=shared_state, shared_lock=shared_lock,
        cam_seg_key=cam_seg_key, cam_rec_key=cam_rec_key,
        end_false_key=end_false_key, tmp_paths_key=tmp_paths_key)
    if opened:                                                                                # 剛開段  # 說明
        out, txt_file = out_new, txt_new                                                      # 更新 writer 與 txt  # 說明
        if original_out_new is not None:                                                      # 若有原始 writer  # 說明
            _shared_set_many(shared_state, shared_lock, {orig_wr_key: original_out_new})      # 存入共享  # 說明
            original_out = original_out_new                                                   # 更新本地變數  # 說明
        frame_count_for_detect = 0                                                            # 段內幀數歸零  # 說明

    # ---- 寫入 / 關段 ----
    if should_record:                                                                         # 錄影中才寫入  # 說明
        frame_count_for_detect += 1                                                           # 段內幀+1  # 說明

        if original_out is not None:                                                          # 原始影片 writer  # 說明
            try:
                original_out.write(original_frame)                                            # 寫原始畫面  # 說明
            except Exception as e:
                print(f"[benchpress_body_loop] original write err: {e}")                      # 錯誤記錄  # 說明

        if out is not None:                                                                   # 疊圖 writer  # 說明
            out.write(frame)                                                                  # 寫上疊圖（含骨架、ROI 框）  # 說明

        if txt_file is not None:                                                              # TXT 檔（關鍵點）  # 說明
            if frame_points is not None:                                                      # 有偵測  # 說明
                line = "Frame {}: [[{}]]\n".format(                                           # 組字串  # 說明
                    frame_count_for_detect,
                    ", ".join(f"({x:.6f}, {y:.6f})" for (x, y) in frame_points))
                txt_file.write(line)                                                          # 寫入  # 說明
            else:
                txt_file.write(f"Frame {frame_count_for_detect}: [[no detection]]\n")         # 無偵測記錄  # 說明

        _shared_set_many(shared_state, shared_lock, {end_false_key: 0})                       # 關段緩衝清零  # 說明

    else:                                                                                     # 不錄影 → 檢查關段  # 說明
        ended, end_false_cnt, frame_reset = _segment_end_if_needed(                           # 判定關段  # 說明
            should_record, is_rec, end_false_cnt, END_GRACE_FRAMES,
            out, original_out, txt_file,
            shared_state, shared_lock, tmp_paths_key, cam_rec_key,
            folder, i, seg_no,
            mapping={"o":"original_vision2.avi","v":"vision2.avi","t":"yolo_skeleton_top.txt"},   # 關段時檔名對應  # 說明
            end_false_key=end_false_key,
            reset_frame_counter=True)                                                         # 關段後幀數歸零  # 說明
        if ended:                                                                             # 已關段  # 說明
            out, txt_file = None, None                                                        # 釋放 writer  # 說明
            _shared_set_many(shared_state, shared_lock, {orig_wr_key: None})                  # 清共享原始 writer  # 說明
            original_out = None                                                               # 釋放本地原始 writer  # 說明
        if frame_reset is not None:                                                           # 需要重置段內幀  # 說明
            frame_count_for_detect = frame_reset                                              # 重置  # 說明

    # ---- 顯示與同步 ----
    _qt_show(label, frame, fps)                                                               # UI 顯示  # 說明
    barrier.wait()                                                                            # 與其他相機同步  # 說明
    return start_time, frame_count, fps, out, frame_count_for_detect, save_sig, txt_file      # 回傳（介面一致）  # 說明


def benchpress_head_loop(i, frame, label, save_sig, folder,                                  # 頭部視角：跟三 Gate 錄影
                         start_time, frame_count, fps, out, original_out,                    # I/O 與 FPS 狀態
                         frame_count_for_detect, barrier,                                    # 偵測幀 / 柵欄
                         shared_state, shared_lock):                                         # shared 狀態
    import cv2                                                                               # 影像處理

    # ---- Key 與初始狀態 -----------------------------------------------------------
    cam_rec_key   = f"rec_cam{i}"                                                            # 是否在錄 key
    cam_seg_key   = f"seg_cam{i}"                                                            # 段號 key
    end_false_key = f"rec_end_false_cam{i}"                                                  # Gate False 緩衝 key
    tmp_paths_key = f"tmp_paths_cam{i}"                                                      # 暫存路徑 key

    # ---- FPS 更新 -----------------------------------------------------------------
    start_time, frame_count, fps = _update_fps(start_time, frame_count, fps)                 # 刷新 FPS
    # ---- Frame 前處理（視角旋轉） --------------------------------------------------
    frame = cv2.rotate(frame, cv2.ROTATE_180)                                               # 旋轉 180°（如不需可移除）
    # ---- 錄影狀態讀取 --------------------------------------------------------------
    is_rec        = _shared_get(shared_state, shared_lock, cam_rec_key, False)              # 是否在錄
    seg_no        = _shared_get(shared_state, shared_lock, cam_seg_key, 0)                  # 段號
    end_false_cnt = _shared_get(shared_state, shared_lock, end_false_key, 0)                # False 緩衝計數

    # ---- 三 Gate 讀取 --------------------------------------------------------------
    gate3_cnt_key = f"gate3_true_cnt_cam{i}"                                                    # 三 Gate 連續命中計數 key
    gate_ui   = _shared_get(shared_state, shared_lock, "auto_recording_sig", False)           # UI Gate 改讀 auto_recording_sig
    gate_body = _shared_get(shared_state, shared_lock, "body_detected", False)                  # 人體 gate
    gate_bar  = _shared_get(shared_state, shared_lock, "bar_y_changed", False)                  # 槓 gate
    should_record = _debounce_three_gate(                                                       # 呼叫共用防抖函式
        is_rec, gate_ui, gate_body, gate_bar,                                                   # 三 Gate 值
        shared_state, shared_lock, gate3_cnt_key                                                # 共享與計數 key
        )

    # ---- UI 未啟動：統一早退 ------------------------------------------------------
    early, (out, original_out, _), (save_sig, frame_count_for_detect) = \
        _idle_if_ui_off(gate_ui, label, frame, fps, barrier,                                # 共用：UI 關就收尾與顯示
                        shared_state, shared_lock, cam_rec_key,
                        (out, original_out, None), (save_sig, frame_count_for_detect))
    if early:                                                                                # 若已早退
        return start_time, frame_count, fps, out, original_out, save_sig, frame_count_for_detect  # 回傳


    # ---- 嘗試開新段（只建暫存） ----------------------------------------------------
    opened, seg_no, out_new, original_new, _ = _segment_start_if_needed(                    # 共用：開段
        should_record, is_rec, folder, i, seg_no, frame,
        need_original=True, need_txt=False, txt_suffix="",                                  # Head 無 txt
        shared_state=shared_state, shared_lock=shared_lock,
        cam_seg_key=cam_seg_key, cam_rec_key=cam_rec_key,
        end_false_key=end_false_key, tmp_paths_key=tmp_paths_key)
    if opened:                                                                               # 若剛開段
        out, original_out = out_new, original_new                                           # 接手 I/O

    # ---- 錄影 / 結束 段控 ----------------------------------------------------------
    if should_record:                                                                        # 錄影中
        if original_out is not None: original_out.write(frame)                               # 寫原始
        if out is not None: out.write(frame)                                                 # 寫疊圖（此視角等同原始）
        _shared_set_many(shared_state, shared_lock, {end_false_key: 0})                      # 清 False 緩衝
    else:                                                                                    # Gate False：可能觸發結束
        ended, end_false_cnt, frame_reset = _segment_end_if_needed(                          # 共用：關段
            should_record, is_rec, end_false_cnt, END_GRACE_FRAMES,
            out, original_out, None,
            shared_state, shared_lock, tmp_paths_key, cam_rec_key,
            folder, i, seg_no,
            mapping={"o": "original_vision3.avi", "v": "vision3.avi"},                       # Head 檔名規則
            end_false_key=end_false_key,
            reset_frame_counter=True)
        if ended:                                                                            # 若已關段
            out, original_out = None, None                                                   # 清 I/O
        if frame_reset is not None:                                                          # 若需重設幀
            frame_count_for_detect = frame_reset                                             # 歸零

    # ---- 顯示與同步 ----------------------------------------------------------------
    _qt_show(label, frame, fps)                                                              # 疊 FPS 並顯示
    barrier.wait()                                                                           # 多執行緒同步

    # ---- 回傳狀態 ------------------------------------------------------------------
    return start_time, frame_count, fps, out, original_out, save_sig, frame_count_for_detect # 與既有介面一致

def _ensure_dir(p):                                       # 確保資料夾存在  # 
    os.makedirs(p, exist_ok=True)                         # 不存在就建立  # 

def _safe_folder(folder):                                 # 檢查最終 folder 是否可用  # 
    return isinstance(folder, (str, bytes, os.PathLike)) and str(folder) != ""  # 可用回 True  # 

_STAGE_SESSION = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"  # 本次程式階段ID  # 

def _staging_root():                                      # 取得暫存根目錄  # 
    root = os.path.join(tempfile.gettempdir(), "CATS_stage")  # 放在系統暫存區  # 
    _ensure_dir(root)                                     # 確保存在  # 
    return root                                           # 回傳路徑  # 

def _staging_dir_for_cam(i):                              # 依相機編號分開的暫存資料夾  # 
    d = os.path.join(_staging_root(), _STAGE_SESSION, f"cam{i}")  # cam 專屬暫存  # 
    _ensure_dir(d)                                        # 確保存在  # 
    return d                                              # 回傳路徑  # 

# ====== 新增：最終輸出根目錄與命名規則 ======

_FINAL_BASE_DIR = r"C:\Users\92A27\benchpress\recordings" # 最終成品儲存根目錄  # 

def _final_root():                                        # 取得最終輸出根目錄  # 
    _ensure_dir(_FINAL_BASE_DIR)                          # 確保根目錄存在  # 
    return _FINAL_BASE_DIR                                # 回傳根目錄  # 

from typing import Optional  # ← 加在檔案開頭
def make_final_recording_dir(ts_str: Optional[str] = None):  # 依規則建立 recording_YYYYMMDD_HHMMSS 資料夾  #
    ts = ts_str or datetime.now().strftime('%Y%m%d_%H%M%S')  # 若未指定則用現在時間  # 
    d = os.path.join(_final_root(), f"recording_{ts}")    # 最終資料夾完整路徑  # 
    _ensure_dir(d)                                        # 確保存在  # 
    return d                                              # 回傳最終資料夾路徑  # 

def commit_stage_to_final(ts_str: Optional[str] = None):  # 將本階段暫存整批搬到最終資料夾  #
    src = os.path.join(_staging_root(), _STAGE_SESSION)   # 本階段暫存根目錄  # 
    if not os.path.isdir(src):                            # 若暫存不存在則略過  # 
        return None                                       # 無可搬資料回傳 None  # 
    dst = make_final_recording_dir(ts_str)                # 先建立最終錄影資料夾  # 
    for name in os.listdir(src):                          # 逐一搬移暫存內容  # 
        shutil.move(os.path.join(src, name), os.path.join(dst, name))  # cam 子資料夾整批搬  # 
    shutil.rmtree(src, ignore_errors=True)                # 清理暫存  # 
    return dst                                            # 回傳最終資料夾路徑  # 
