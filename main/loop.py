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
# FOURCC_MJPG = cv2.VideoWriter_fourcc(*'MJPG')                                            # MJPG fourcc
FOURCC_MJPG = cv2.VideoWriter_fourcc(*'mp4v')                                            # MJPG fourcc
REC_FPS     = 29                                                                         # 錄影 FPS


def squat_bar_loop(i, frame, label, save_sig, recording_sig, folder,                     # 槓視角（cam1）
                   start_time, frame_count, fps, out, original_out,                      # 疊圖 writer／原始 writer
                   model, txt_file, frame_count_for_detect, barrier):                    # 模型／txt／幀計數／柵欄
    # ---- FPS 計算 ----                                                                   # FPS
    frame_count += 1                                                                     # 幀數+1
    elapsed_time = time.time() - start_time                                              # 距上次秒數
    if elapsed_time >= 1:                                                                # 每秒刷新
        fps = frame_count / elapsed_time                                                 # 計算 FPS
        frame_count = 0                                                                  # 幀數歸零
        start_time = time.time()                                                         # 重置起點

    # ---- 影像前處理（旋轉） ----                                                          # 旋轉
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)                                   # 順時針 90°

    # ---- 初始化 VideoWriter（在開始錄影時） ----                                             # 初始化 writer
    if recording_sig:
        
        if original_out is None:                                                         # 原始影像 writer
            ori_path = os.path.join(folder, f'original_vision{i+1}.mp4')                 # 檔名：original_vision*.mp4
            h, w = frame.shape[:2]                                                       # 高寬
            original_out = cv2.VideoWriter(ori_path, FOURCC_MJPG, REC_FPS, (w, h))       # 建立 writer
        if out is None:                                                                  # 疊圖影像 writer
            vis_path = os.path.join(folder, f'vision{i+1}.mp4')                          # 檔名：vision*.mp4
            h, w = frame.shape[:2]                                                       # 高寬
            out = cv2.VideoWriter(vis_path, FOURCC_MJPG, REC_FPS, (w, h))                # 建立 writer
        if txt_file is None:                                                             # 槓座標 txt（沿用你的格式）
            txt_path = os.path.join(folder, 'yolo_coordinates.txt')                      # txt 檔案路徑
            txt_file = open(txt_path, "w")                                               # 開啟檔案
            frame_count_for_detect = 0                                                   # 幀計數歸零（定義為：每幀+1）
    
    # ---- 生成原始幀副本（給原始檔用） ----                                                   # 原始副本
    original_frame = frame.copy()                                                        # 複製一份未疊圖影像

    # ---- YOLO 推論（只鎖推論本身；不鎖其他工作） ----                                          # 推論序列化
    results = None                                                                       # 預設無結果
    if recording_sig:                                                                    # 只在錄影時推論（可視需求改成永遠推）
        with inference_lock:                                                             # 進入推論臨界區（cam1/cam2 輪流）
            results = model(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF,                     # 單幀推論
                             max_det=YOLO_MAXDET, verbose=False)                         # 不用 stream=True

    # ---- 疊圖（bar 用內建 plot） + 寫 txt（每幀只加一次計數） ----                               # 疊圖與記錄
    if recording_sig and results is not None and len(results) > 0:                       # 有結果才處理
        r0 = results[0]                                                                  # 取第一筆
        frame = r0.plot()                                                                # 疊框到 frame（只畫一次）
        boxes_ok = hasattr(r0, "boxes") and r0.boxes is not None and r0.boxes.xywh is not None  # 檢查 boxes
        # 幀序號 +1（不論是否有 box；每幀只加一次）                                              
        frame_count_for_detect += 1                                                      # 幀計數+1（統一定義）
        if txt_file is not None:                                                         # 有檔案才寫
            if boxes_ok and r0.boxes.xywh.shape[0] > 0:                                  # 有偵測到
                for xywh in r0.boxes.xywh.cpu().numpy():                                 # 逐 box（通常 max_det=1）
                    x_c, y_c, w_b, h_b = xywh                                            # 取 xywh
                    txt_file.write(f"{frame_count_for_detect},{x_c},{y_c},{w_b},{h_b}\n")# 寫一行
            else:
                txt_file.write(f"{frame_count_for_detect},no detection\n")               # 寫無偵測
    else:
        # 沒推論或沒錄影時，不動 frame_count_for_detect（保持只在錄影期間計數）                      # 非錄影不計
        pass                                                                             # 保持現狀

    # ---- 寫入影片（原始 + 疊圖） ----                                                          # 寫檔
    if recording_sig:
        if original_out is not None:                                                     # 原始 writer 存在
            original_out.write(original_frame)                                           # 寫原始幀
        if out is not None:                                                              # 疊圖 writer 存在
            out.write(frame)                                                             # 寫疊圖幀

    # ---- 同步（為安全仍每幀 wait；如要真每 2 幀同調，需加共用幀計數） ----                           # 柵欄
    barrier.wait()                                                                       # 每幀同步（避免死鎖）

    # ---- 錄影結束處理 ----                                                                    # 收尾
    if not recording_sig:                                                                # 未錄影狀態
        frame_count_for_detect = 0                                                       # 幀計數歸零
        if save_sig and out is not None:                                                 # 若要保存且 writer 存在
            out.release()                                                                # 關閉疊圖 writer
            out = None                                                                   # 置空
        if save_sig and original_out is not None:                                        # 原始 writer
            original_out.release()                                                       # 關閉原始 writer
            original_out = None                                                          # 置空
        save_sig = False                                                                 # 清保存旗標
        if txt_file is not None:                                                         # 關閉 txt
            txt_file.close()                                                             # 關
            txt_file = None                                                              # 置空

    # ---- 顯示到 Qt ----                                                                       # 顯示
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,            # 畫 FPS
                1, (0, 255, 0), 2, cv2.LINE_AA)                                          # 樣式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                   # BGR→RGB
    h, w, ch = frame.shape                                                               # 取尺寸
    qimg = QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)          # 做 QImage
    scale_qpix = QtGui.QPixmap.fromImage(qimg).scaled(                                   # 縮放
        label.width(), label.height(), QtCore.Qt.KeepAspectRatio,                        # 等比
        QtCore.Qt.SmoothTransformation)                                                  # 平滑
    label.setPixmap(scale_qpix)                                                          # 顯示
    return start_time, frame_count, fps, out, original_out, frame_count_for_detect, save_sig, txt_file  # 回傳


def squat_bone_loop(i, frame, label, save_sig, recording_sig, folder,                     # 骨架視角（cam2）
                    start_time, frame_count, fps, out, original_out,                      # 疊圖／原始 writer
                    model, txt_file, frame_count_for_detect,                              # 模型／txt／幀計數
                    skeleton_connections, barrier):                                       # 骨架連線／柵欄
    import os, time                                                                       # 檔案/時間
    import cv2                                                                            # 影像處理
    from PyQt5 import QtGui, QtCore                                                       # Qt 顯示

    # ---- FPS 計算 ----
    frame_count += 1                                                                      # 幀+1
    elapsed_time = time.time() - start_time                                               # 距上次秒數
    if elapsed_time >= 1:                                                                 # 每秒刷新
        fps = frame_count / elapsed_time                                                  # 計算 FPS
        frame_count = 0                                                                   # 計數歸零
        start_time = time.time()                                                          # 重置起點

    # ---- 旋轉 ----
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)                                    # 影像順時針 90°
    original_frame = frame.copy()                                                         # 保留原始幀

    # ---- 初始化錄影與 txt（僅在錄影中建立）----
    if recording_sig:                                                                     # 僅錄影時建立輸出
        if original_out is None:                                                          # 原始 writer
            h, w = frame.shape[:2]                                                        # 幀高寬
            ori_path = os.path.join(folder, f'original_vision{i+1}.mp4')                  # 原始檔名
            original_out = cv2.VideoWriter(ori_path, FOURCC_MJPG, REC_FPS, (w, h))        # 建立原始 writer
        if out is None:                                                                   # 疊圖 writer
            h, w = frame.shape[:2]                                                        # 幀高寬
            vis_path = os.path.join(folder, f'vision{i+1}.mp4')                           # 疊圖檔名
            out = cv2.VideoWriter(vis_path, FOURCC_MJPG, REC_FPS, (w, h))                 # 建立疊圖 writer
        if txt_file is None:                                                              # 骨架 txt
            txt_path = os.path.join(folder, 'mediapipe_landmarks.txt')                    # txt 檔名（沿用）
            txt_file = open(txt_path, "w", buffering=1)                                   # 開檔（行緩衝）
            frame_count_for_detect = 0                                                    # 幀計數歸零

    # ---- YOLO 推論（僅錄影時推）----
    keypoints_xy = None                                                                   # 預設無點
    if recording_sig:                                                                     # 錄影時才推論
        with inference_lock:                                                              # 進入推論臨界區
            results = model(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF,                      # 單幀推論
                             max_det=YOLO_MAXDET, verbose=False)                          # 控制輸出
        if results and len(results) > 0 and getattr(results[0], "keypoints", None):       # 確保有結果與骨架
            r0 = results[0]                                                               # 取第一筆
            kobj = r0.keypoints                                                           # Keypoints 物件
            try:
                kpts = kobj[0]                                                            # 只取第一個人的骨架
                keypoints_xy = getattr(kpts, "xy", None)                                  # 取 xy 張量
            except Exception:
                keypoints_xy = None                                                       # 容錯：無法索引時視為無點

    # ---- txt 寫入 + 點位過濾/繪製 ----
    if recording_sig:                                                                     # 僅錄影時處理
        frame_count_for_detect += 1                                                       # 幀+1（統一定義）
        kp_coords = []                                                                    # 畫線用的點列表（含 None）
        frame_data = []                                                                   # 該幀輸出行

        if keypoints_xy is not None:                                                      # 有骨架張量
            xy = keypoints_xy                                                             # 形如 (1, K, 2)
            if hasattr(xy, "shape") and len(xy.shape) == 3 and xy.shape[0] >= 1:         # 基本形狀檢查
                for idx, kp in enumerate(xy[0]):                                          # 遍歷 K 個關節
                    x_kp, y_kp = int(kp[0].item()), int(kp[1].item())                     # 轉 int
                    if x_kp == 0 and y_kp == 0:                                           # (0,0) 視為無效
                        kp_coords.append(None)                                            # 記 None
                    else:
                        kp_coords.append((x_kp, y_kp))                                    # 留有效點
                        cv2.circle(frame, (x_kp, y_kp), 5, (0, 255, 0), cv2.FILLED)       # 畫關節點
                    frame_data.append(f"{frame_count_for_detect},{idx},{x_kp},{y_kp}")    # 紀錄一列
            else:
                frame_data.append(f"{frame_count_for_detect},no detection")               # 形狀不符視為無偵測
        else:
            frame_data.append(f"{frame_count_for_detect},no detection")                   # 完全無偵測

        # 繪製連線（若任一端為 None 則跳過）
        if kp_coords:
            for s_idx, e_idx in skeleton_connections:                                     # 逐邊
                if s_idx < len(kp_coords) and e_idx < len(kp_coords):                     # 邊界檢查
                    ps, pe = kp_coords[s_idx], kp_coords[e_idx]                           # 端點
                    if ps is None or pe is None:                                          # 任一無效則略過
                        continue
                    cv2.line(frame, ps, pe, (0, 255, 255), 2)                             # 畫連線

        # 寫入 txt（逐行寫入）
        if txt_file is not None:                                                          # 確保檔案存在
            if frame_data and isinstance(frame_data[0], str):                             # 正常行
                for row in frame_data:                                                    # 逐行
                    txt_file.write(row + "\n")                                            # 寫入一行
            else:
                txt_file.write(f"{frame_count_for_detect},no detection\n")                # 保底

    # ---- 寫入影片（原始 + 疊圖）----
    if recording_sig:                                                                     # 錄影中才寫
        if original_out is not None:                                                      # 原始 writer
            original_out.write(original_frame)                                            # 寫原始幀
        if out is not None:                                                               # 疊圖 writer
            out.write(frame)                                                              # 寫疊圖幀

    # ---- 同步（每幀阻塞）----
    barrier.wait()                                                                        # 多執行緒對齊

    # ---- 錄影結束處理 ----
    if not recording_sig:                                                                 # 未錄影
        frame_count_for_detect = 0                                                        # 幀歸零
        if save_sig and out is not None:                                                  # 關疊圖
            out.release()                                                                 # 釋放
            out = None                                                                    # 置空
        if save_sig and original_out is not None:                                         # 關原始
            original_out.release()                                                        # 釋放
            original_out = None                                                           # 置空
        if txt_file is not None:                                                          # 關 txt
            txt_file.close()                                                              # 釋放
            txt_file = None                                                               # 置空
        save_sig = False                                                                  # 清旗標

    # ---- 顯示到 Qt ----
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,             # 顯示 FPS
                1, (0, 255, 0), 2, cv2.LINE_AA)                                           # 樣式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                    # BGR→RGB
    h, w, ch = frame.shape                                                                # 幀尺寸
    qimg = QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)           # QImage
    scale_qpix = QtGui.QPixmap.fromImage(qimg).scaled(                                    # 縮放顯示
        label.width(), label.height(), QtCore.Qt.KeepAspectRatio,                         # 等比縮放
        QtCore.Qt.SmoothTransformation)                                                   # 平滑
    label.setPixmap(scale_qpix)                                                           # 設圖到 QLabel

    return start_time, frame_count, fps, out, original_out, frame_count_for_detect, save_sig, txt_file  # 回傳狀態


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
            path = os.path.join(folder, f'vision{i+1}.mp4')                              # 檔名
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

def benchpress_bar_loop(i, frame, label, save_sig, folder,                                   # 槓視角
                        start_time, frame_count, fps, out, original_out, model, txt_file,    # I/O 與模型
                        frame_count_for_detect, barrier,                                     # 幀計數 / 柵欄
                        shared_state, shared_lock, BAR_MOVE_THRESH):                         # 共享狀態 / 門檻
    # 常用 key                                                                           # 事先把 key 固定好
    cam_rec_key = f"rec_cam{i}"                                                               # 是否在錄
    cam_seg_key = f"seg_cam{i}"                                                               # 段號
    end_false_key = f"rec_end_false_cam{i}"                                                   # False 緩衝
    tmp_paths_key = f"tmp_paths_cam{i}"                                                       # 暫存路徑

    # FPS 更新                                                                          # 每幀維護 FPS
    start_time, frame_count, fps = _update_fps(start_time, frame_count, fps)                  # 刷新 FPS

    # 讀 gate 與狀態                                                                    # 集中讀取
    gate_ui = _shared_get(shared_state, shared_lock, "recording_sig", False)                  # UI Gate
    is_rec  = _shared_get(shared_state, shared_lock, cam_rec_key, False)                      # 是否在錄
    seg_no  = _shared_get(shared_state, shared_lock, cam_seg_key, 0)                          # 段號
    end_false_cnt = _shared_get(shared_state, shared_lock, end_false_key, 0)                  # False 緩衝

    # UI 未啟動 → 統一收尾                                                             # 用共用工具早退
    early, (out, original_out, txt_file), (save_sig, frame_count_for_detect) = \
        _idle_if_ui_off(gate_ui, label, frame, fps, barrier,
                        shared_state, shared_lock, cam_rec_key,
                        (out, original_out, txt_file), (save_sig, frame_count_for_detect))
    if early:                                                                                # 若早退
        return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file   # 回

    # YOLO 推論 + 疊圖                                                                 # 偵測第一框
    results = model.predict(source=frame, imgsz=320, conf=0.5, verbose=False)                 # YOLO
    xywh = _yolo_first_box_xywh(results)                                                      # 取第一框
    for r in results: frame = r.plot()                                                        # 疊圖

    # 更新 bar gate（保持/遺失容忍）                                                   # 只專注 gate 計算
    prev_x = _shared_get(shared_state, shared_lock, "prev_bar_x", None)                       # 上一幀x
    prev_y = _shared_get(shared_state, shared_lock, "prev_bar_y", None)                       # 上一幀y
    bar_hold_key = f"bar_hold_cam{i}"                                                         # 保持
    bar_loss_key = f"bar_loss_cnt_cam{i}"                                                     # 遺失
    bar_hold = _shared_get(shared_state, shared_lock, bar_hold_key, 0)                        # 讀保持
    bar_loss = _shared_get(shared_state, shared_lock, bar_loss_key, 0)                        # 讀遺失
    if xywh is not None:                                                                      # 有偵測
        x, y, w, h = xywh                                                                     # 解包
        bar_loss = 0                                                                          # 遺失清零
        if prev_y is not None and abs(y - prev_y) >= BAR_MOVE_THRESH:                         # 主要門檻
            bar_hold = BAR_HOLD_FRAMES                                                        # 續滿
        elif bar_hold > 0 and ((prev_x is not None and abs(x - prev_x) >= 1) or (prev_y is not None and abs(y - prev_y) >= 1)):
            bar_hold = BAR_HOLD_FRAMES                                                        # 續命
        else:
            bar_hold = max(0, bar_hold - 1)                                                   # 遞減
        _shared_set_many(shared_state, shared_lock, {"prev_bar_x": x, "prev_bar_y": y})       # 更新 prev
    else:                                                                                     # 無偵測
        bar_loss += 1                                                                         # 遺失+1
        bar_hold = max(0, bar_hold - 1) if bar_loss <= BAR_LOSS_TOL_FRAMES else 0            # 超過容忍清零
    _shared_set_many(shared_state, shared_lock, {bar_hold_key: bar_hold, bar_loss_key: bar_loss, "bar_y_changed": (bar_hold>0)})  # 回寫 gate

    # 三 Gate 決策                                                                     # 統一三 gate
    gate_body = _shared_get(shared_state, shared_lock, "body_detected", False)                # 人體 gate
    should_record = gate_ui and gate_body and (bar_hold > 0)                                  # 三者同時

    # 開段（只建暫存）                                                                 # 共用開段
    opened, seg_no, out_new, original_new, txt_new = _segment_start_if_needed(
        should_record, is_rec, folder, i, seg_no, frame,
        need_original=True, need_txt=True, txt_suffix="bar",
        shared_state=shared_state, shared_lock=shared_lock,
        cam_seg_key=cam_seg_key, cam_rec_key=cam_rec_key,
        end_false_key=end_false_key, tmp_paths_key=tmp_paths_key)
    if opened:                                                                                # 若剛開段
        out, original_out, txt_file = out_new, original_new, txt_new                          # 接手 I/O
        frame_count_for_detect = 0                                                            # 清幀

    # 寫入 / 關段                                                                       # 共用模板
    if should_record:                                                                         # 錄影中
        if original_out is not None: original_out.write(frame)                                # 寫原始
        if out is not None: out.write(frame)                                                  # 寫疊圖
        frame_count_for_detect += 1                                                           # 偵測幀+1
        if txt_file is not None:                                                              # 寫 txt
            if xywh is not None:
                x,y,w,h = xywh                                                                # 解包
                txt_file.write(f"{frame_count_for_detect},{x},{y},{w},{h}\n")                 # 記錄
            else:
                txt_file.write(f"{frame_count_for_detect},no detection\n")                    # 無偵測
        _shared_set_many(shared_state, shared_lock, {end_false_key: 0})                       # 清緩衝
    else:                                                                                     # Gate False
# --- 在 benchpress_bar_loop() 內，關段的那一段，補 end_false_key ---
        ended, end_false_cnt, reset = _segment_end_if_needed(
            should_record, is_rec, end_false_cnt, END_GRACE_FRAMES,
            out, original_out, txt_file,
            shared_state, shared_lock, tmp_paths_key, cam_rec_key,
            folder, i, seg_no,
            mapping={"o":"original_vision1.mp4","v":"vision1.mp4","t":"yolo_coordinates.txt"},
            end_false_key=end_false_key,                 # ★ 必須補這行
            reset_frame_counter=True)                    #   才不會把 True 傳到 end_false_key

    # 顯示與同步                                                                       # 共用顯示
    _qt_show(label, frame, fps)                                                               # 顯示
    barrier.wait()                                                                            # 同步
    return start_time, frame_count, fps, out, frame_count_for_detect, original_out, save_sig, txt_file  # 回傳

def benchpress_body_loop(i, frame, label, save_sig, folder,                                   # 人體視角
                         start_time, frame_count, fps, out, model, txt_file,                  # I/O 與模型（順序不變）
                         frame_count_for_detect, skeleton_connections, barrier,               # 幀計數 / 柵欄
                         shared_state, shared_lock):                                          # 共享狀態
    # ---- 常用 key --------------------------------------------------------------------
    cam_rec_key   = f"rec_cam{i}"                                                             # 是否在錄
    cam_seg_key   = f"seg_cam{i}"                                                             # 段號
    end_false_key = f"rec_end_false_cam{i}"                                                   # False 緩衝
    tmp_paths_key = f"tmp_paths_cam{i}"                                                       # 暫存路徑
    hit_key       = f"body_hit_cnt_cam{i}"                                                    # 命中
    miss_key      = f"body_miss_cnt_cam{i}"                                                   # 未中
    orig_wr_key   = f"original_out_cam{i}"                                                    # 原始 writer 句柄

    # ---- FPS 更新 --------------------------------------------------------------------
    start_time, frame_count, fps = _update_fps(start_time, frame_count, fps)                  # 刷新 FPS

    # ---- 狀態讀取 --------------------------------------------------------------------
    gate_ui   = _shared_get(shared_state, shared_lock, "recording_sig", False)                # UI Gate
    is_rec    = _shared_get(shared_state, shared_lock, cam_rec_key, False)                    # 是否在錄
    seg_no    = _shared_get(shared_state, shared_lock, cam_seg_key, 0)                        # 段號
    end_false_cnt = _shared_get(shared_state, shared_lock, end_false_key, 0)                  # False 緩衝
    hit_cnt   = _shared_get(shared_state, shared_lock, hit_key, 0)                            # 命中幀
    miss_cnt  = _shared_get(shared_state, shared_lock, miss_key, 0)                           # 未中幀
    original_out = _shared_get(shared_state, shared_lock, orig_wr_key, None)                  # 取原始 writer

    # ---- UI 未啟動 → 統一收尾（早退） -------------------------------------------------
    early, (out, _unused_original, txt_file), (save_sig, frame_count_for_detect) =\
        _idle_if_ui_off(gate_ui, label, frame, fps, barrier,
                        shared_state, shared_lock, cam_rec_key,
                        (out, None, txt_file), (save_sig, frame_count_for_detect))
    if early:                                                                                 # 若早退
        return start_time, frame_count, fps, out, frame_count_for_detect, save_sig, txt_file  # 回傳數量不變

    # ---- 保留原始畫面（給 original_vision2.mp4） ---------------------------------------
    try:
        original_frame = frame.copy()                                                         # 原始畫面
    except Exception:
        original_frame = frame                                                               # 退化保護

    # ---- YOLO 推論 + 疊圖 --------------------------------------------------------------
    try:
        results = list(model(source=frame, stream=True, verbose=False))                       # 推論
    except Exception as e:
        results = []                                                                          # 失敗視為無偵測
        print(f"[benchpress_body_loop] model error: {e}")                                     # 記錄

    frame_count_for_detect += 1                                                               # 偵測幀+1
    body_now, rows = False, []                                                                # 當幀旗標/輸出列
    if results and getattr(results[0], "keypoints", None) is not None:                        # 有點集
        kpts = results[0].keypoints                                                           # 取點
        xy = kpts.xy                                                                          # xy
        if hasattr(xy, "detach"): xy = xy.detach().cpu().numpy()                              # tensor→np
        first = xy[0] if len(xy) > 0 else None                                                # 第一人
        if first is not None:                                                                 # 有人體
            body_now = True                                                                   # 標記
            if not skeleton_connections:                                                      # 預設連線
                skeleton_connections = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(4,6),(5,7)]      # 簡化
            import cv2                                                                        # 僅此處用
            pts = []                                                                          # 點集
            for idx, (xk, yk) in enumerate(first[:, :2].astype(int)):                         # 逐點
                valid = not (xk == 0 and yk == 0)                                             # 有效點
                pts.append((xk, yk) if valid else None)                                       # 收集
                if valid: cv2.circle(frame, (xk, yk), 5, (0,255,0), cv2.FILLED)               # 畫點
                rows.append(f"{frame_count_for_detect},{idx},{xk},{yk}")                      # 記錄
            for a, b in skeleton_connections:                                                 # 逐線
                if a < len(pts) and b < len(pts) and pts[a] and pts[b]:                       # 檢查
                    cv2.line(frame, pts[a], pts[b], (0,255,255), 2)                           # 畫線

    # ---- 人體 gate 鎖存（20 幀緩衝） -----------------------------------------------------
    hit_cnt, miss_cnt, latched_now = _latch_by_buffer(hit_cnt, miss_cnt, body_now, BODY_BUF_FRAMES)  # 緩衝
    _shared_set_many(shared_state, shared_lock, {hit_key: hit_cnt, miss_key: miss_cnt})       # 回寫
    _shared_set_many(shared_state, shared_lock, {"body_detected": latched_now})               # 更新 gate

    # ---- 三 Gate 決策 -------------------------------------------------------------------
    gate_bar = _shared_get(shared_state, shared_lock, "bar_y_changed", False)                 # 槓 gate
    should_record = gate_ui and latched_now and gate_bar                                      # 三者同時

    # ---- 開段（同時建立疊圖 out 與原始 original_out） -----------------------------------
    opened, seg_no, out_new, original_out_new, txt_new = _segment_start_if_needed(            # 取新 I/O
        should_record, is_rec, folder, i, seg_no, frame,
        need_original=True, need_txt=True, txt_suffix="body",                                 # 要原始+txt
        shared_state=shared_state, shared_lock=shared_lock,
        cam_seg_key=cam_seg_key, cam_rec_key=cam_rec_key,
        end_false_key=end_false_key, tmp_paths_key=tmp_paths_key)

    if opened:                                                                                # 剛開段
        out, txt_file = out_new, txt_new                                                      # 接手 I/O
        if original_out_new is not None:                                                      # 記住原始 writer
            _shared_set_many(shared_state, shared_lock, {orig_wr_key: original_out_new})      # 存共享
            original_out = original_out_new                                                   # 更新本地

    # ---- 寫入 / 關段 --------------------------------------------------------------------
    if should_record:                                                                         # 錄影中
        if original_out is not None:                                                          # 先寫原始
            try: original_out.write(original_frame)                                           # 原始畫面
            except Exception as e: print(f"[benchpress_body_loop] original write err: {e}")   # 保護
        if out is not None:                                                                   # 再寫疊圖
            out.write(frame)                                                                  # 疊圖畫面
        if txt_file is not None:                                                              # 寫關鍵點
            if body_now and rows: txt_file.write("\n".join(rows) + "\n")                      # 批次
            else: txt_file.write(f"{frame_count_for_detect},no detection\n")                  # 無偵測
        _shared_set_many(shared_state, shared_lock, {end_false_key: 0})                       # 清緩衝
    else:                                                                                     # Gate False
        ended, end_false_cnt, _ = _segment_end_if_needed(
            should_record, is_rec, end_false_cnt, END_GRACE_FRAMES,
            out, original_out, txt_file,                                                      # 連同原始 writer
            shared_state, shared_lock, tmp_paths_key, cam_rec_key,
            folder, i, seg_no,
            mapping={"o":"original_vision2.mp4","v":"vision2.mp4","t":"yolo_body_keypoints.txt"},
            end_false_key=end_false_key)                                                      # 結束判定
        if ended:                                                                             # 段落已關
            out, txt_file = None, None                                                        # 釋放本地
            _shared_set_many(shared_state, shared_lock, {orig_wr_key: None})                  # 清共享
            original_out = None                                                               # 釋放原始

    # ---- 顯示與同步 ---------------------------------------------------------------------
    _qt_show(label, frame, fps)                                                               # 顯示
    barrier.wait()                                                                            # 同步
    return start_time, frame_count, fps, out, frame_count_for_detect, save_sig, txt_file      # 回傳（7）

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
    # ---- 三 Gate 讀取 --------------------------------------------------------------
    gate_ui   = _shared_get(shared_state, shared_lock, "recording_sig", False)              # UI Gate
    gate_body = _shared_get(shared_state, shared_lock, "body_detected", False)              # 人體 Gate
    gate_bar  = _shared_get(shared_state, shared_lock, "bar_y_changed", False)              # 槓 Gate
    should_record = gate_ui and gate_body and gate_bar                                       # 三 Gate 決策

    # ---- 錄影狀態讀取 --------------------------------------------------------------
    is_rec        = _shared_get(shared_state, shared_lock, cam_rec_key, False)              # 是否在錄
    seg_no        = _shared_get(shared_state, shared_lock, cam_seg_key, 0)                  # 段號
    end_false_cnt = _shared_get(shared_state, shared_lock, end_false_key, 0)                # False 緩衝計數

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
            mapping={"o": "original_vision3.mp4", "v": "vision3.mp4"},                       # Head 檔名規則
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
