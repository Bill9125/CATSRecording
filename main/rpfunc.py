from PyQt5 import QtCore, QtGui, QtWidgets
import os, glob, sys, time
import cv2, threading
from PyQt5.QtGui import QPainter, QPen
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import json
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject

class GraphUpdater(QObject):
    update_signal = pyqtSignal()

    def __init__(self, canvas):
        super().__init__()
        self.canvas = canvas
        self.update_signal.connect(self.canvas.draw)

    def update(self):
        self.update_signal.emit()

class LineLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)                                         # 呼叫父類初始化
        self.vertical_line_x = 0                                         # 垂直線 X 位置
        self.horizontal_line_y = 0                                       # 水平線 Y 位置

    def set_vertical_line(self, value):
        self.vertical_line_x = value                                     # ✅ 應該更新 X（原本寫成 horizontal）
        self.update()                                                    # 重新觸發繪製

    def set_horizontal_line(self, value):
        self.horizontal_line_y = value                                   # ✅ 應該更新 Y（原本寫成 vertical）
        self.update()                                                    # 重新觸發繪製

    def paintEvent(self, event):
        super().paintEvent(event)  # 保持 QLabel 原本的行為

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QtCore.Qt.red, 3, QtCore.Qt.SolidLine)  # 設定紅色 3px 的線條
        painter.setPen(pen)

        # 畫垂直線
        painter.drawLine(self.vertical_line_x, 0, self.vertical_line_x, self.height())

        # 畫水平線
        painter.drawLine(0, self.horizontal_line_y, self.width(), self.horizontal_line_y)

        painter.end()

        
class MyThread(threading.Thread):
    def __init__(self, caps, index, Play_btn, icons, fast_forward_combobox,
                    Frameslider, framenumber, Vision_labels, qpixmaps, barrier):
        threading.Thread.__init__(self, daemon=True)
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()
        self._stop_event.clear()
        self.cap = caps[index]
        self.index = index
        self.Play_btn = Play_btn
        self.icons = icons
        self.fast_forward_combobox = fast_forward_combobox
        self.Frameslider = Frameslider
        self.framenumber = framenumber
        self.Vision_label = Vision_labels[self.index]
        self.qpixmap = qpixmaps[self.index]
        self.barrier = barrier
        self.is_pause = False
        self.is_slide_end = False
        self.is_slide_start = False
        
    def run(self):
        self.Frameslider.setMaximum(int(self.framenumber))
        start_time = time.time()

        while not self._stop_event.is_set():
            speed_rate = self.fast_forward_combobox.currentText()
            spf = 1 / 30

            # 迴圈暫停條件
            # if self.is_pause:
            #     continue
            self._pause_event.wait()
                
            if self.is_slide_start:
                if self.is_slide_end:
                    val = self.Frameslider.value()
                    current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if val > current_frame:
                        for _ in range(val - current_frame):
                            self.cap.grab()
                    elif val < current_frame:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, val)
                    print(f'{self.index} cap is set.')
                    self.is_slide_end = False
                    self.is_slide_start = False
                self.barrier.wait() 
                continue
                
            # 迴圈終止條件
            if self.Frameslider.value() >= self.framenumber:
                break

            # 等待所有 thread 完成同步
            self.barrier.wait()

            if (time.time() - start_time) >= (spf / float(speed_rate)):
                # 讓第 0 個 threading 處理 slider
                if self.index == 0:
                    val = self.Frameslider.value()
                    self.Frameslider.setValue(val + 1)
                _ , frame = self.cap.read()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                self.qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
                scale_qpixmap = self.qpixmap.scaled(self.Vision_label.width(), self.Vision_label.height(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
                self.Vision_label.setPixmap(scale_qpixmap)
                # 更新時間
                start_time = time.time()
                
        print('break')
        qpixmap = QtGui.QPixmap()
        self.Vision_label.setPixmap(qpixmap)
        self.cap.release()

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def stop(self):
        self._stop_event.set()

class Thread_data(threading.Thread):
    def __init__(self, index, gragh, data, barrier, fast_forward_combobox, Frameslider, framenumber):
        threading.Thread.__init__(self, daemon=True)
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()
        self._stop_event.clear()
        
        self.index = index
        self.ax = gragh['axes'][index]
        self.canvas = gragh['canvas']
        self.data = data
        self.fast_forward_combobox = fast_forward_combobox
        self.Frameslider = Frameslider
        self.framenumber = framenumber
        self.barrier = barrier
        self.is_pause = False
        self.is_slide_end = False
        self.is_slide_start = False
        
        # ✅ 確保 `y_data` 格式正確
        self.x_data = self.data['frames']
        self.y_data = self.data['values']

        if isinstance(self.y_data[0], (list, tuple)) and len(self.y_data[0]) == 2:
            # ✅ 如果 `y_data` 是二維 (e.g., [(val1, val2), (val3, val4), ...])
            self.right_values = [v[0] for v in self.y_data]  # 右側數據
            self.left_values = [v[1] for v in self.y_data]   # 左側數據
            self.line1, = self.ax.plot([], [], color="blue")
            self.line2, = self.ax.plot([], [], color="red")
            self.is_2d = True
        else:
            # ✅ 如果 `y_data` 是一維 (e.g., [val1, val2, val3, ...])
            self.line, = self.ax.plot([], [], color="red")
            self.is_2d = False

        # ✅ 設定軸範圍
        self.ax.set_xlim(min(self.x_data), max(self.x_data))
        self.ax.set_ylim(self.data['y_min'], self.data['y_max'])
        self.ax.set_ylabel(f"{self.data['y_label']}")
        self.ax.legend()

    def run(self):
        start_time = time.time()
        while not self._stop_event.is_set():
            speed_rate = self.fast_forward_combobox.currentText()
            spf = 1 / 30
            self._pause_event.wait()

            if self.is_slide_start:
                if self.is_slide_end:
                    # ✅ 清除舊數據
                    if self.is_2d:
                        self.line1.set_data([], [])
                        self.line2.set_data([], [])
                    else:
                        self.line.set_data([], [])

                    self.is_slide_end = False
                    self.is_slide_start = False
                continue

            if self.Frameslider.value() >= self.framenumber:
                break

            self.barrier.wait()
            val = self.Frameslider.value()

            if self.is_2d:
                self.line1.set_data(self.x_data[:val], self.right_values[:val])
                self.line2.set_data(self.x_data[:val], self.left_values[:val])
            else:
                self.line.set_data(self.x_data[:val], self.y_data[:val])
            if (time.time() - start_time) >= (spf / float(speed_rate)):
                if self.index == 0:
                    self.canvas.draw()
                    start_time = time.time()

    def pause(self):
        self.is_pause = True
        self._pause_event.clear()

    def resume(self):
        self.is_pause = False
        self._pause_event.set()

class Replaybackend():
    def __init__(self):
        super(Replaybackend, self).__init__()
        # init for replay
        self.firstclicked_D = True
        self.firstclicked_B = True
        self.firstclicked_S = True
        self.data_path = {'Deadlift': ['Bar_Position.json', 'Hip_Angle.json', 
                                       'Knee_Angle.json', 'Knee_to_Hip.json', 'Score.json'],
                          'Benchpress' : ['Bar_Position.json', 'Armpit_Angle.json', 
                                          'Shoulder_Angle.json', 'Elbow_Angle.json'],
                           'Squat': ['Bar_Position.json', 'Hip_Angle.json', 
                                       'Knee_Angle.json', 'Knee_to_Hip.json', 'Score.json']}
        self.folders = {}
        self.threads = []
        self.rp_Vision_labels = []
        self.rp_qpixmaps = []
        self.videos = []
        self.caps = []
        self.pred_result = []
        self.info_data = []
        self.pred_data = []
        self.currentsport = ''
        self.ocv = True
        self.index = 0
        self.is_pause = False
        self.exited = False
        self.is_stop = True

    def Deadlift_btn_pressed(
        self, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn, icons,
        Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout,
        head_label, bottom_labels, graph, table
        ):
        self.table = table
        self.currentsport = 'Deadlift'
        self.rp_Vision_labels = head_label + bottom_labels
        self.data_graph = graph
        self.rp_btn_press(
            self.currentsport, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn, icons,
            Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout
            )
        
    def Benchpress_btn_pressed(
        self, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn, icons,
        Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout,
        head_label, bottom_labels, V_sliders, H_sliders, graph, table
        ):
        self.currentsport = 'Benchpress'
        self.rp_Vision_labels = [head_label] + bottom_labels
        self.data_graph = graph
        self.V_sliders = V_sliders
        self.H_sliders = H_sliders
        self.rp_btn_press(
            self.currentsport, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn, icons,
            Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout
            )
        
    def Squat_btn_pressed(
        self, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn, icons,
        Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout,
        head_label, bottom_labels, data_labels
        ):
        self.currentsport = 'Squat'
        self.rp_Vision_labels = head_label + bottom_labels
        self.data_labels = data_labels
        self.rp_btn_press(
            self.currentsport, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn, icons,
            Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout,
            )
        
    def rp_btn_press(                                                           # 播放區共用的按鍵初始化與狀態設定
        self, sport, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn, icons,  # sport 與三個 sport 切換按鈕與圖示
        Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab,    # 停止鍵、時間軸、倍速、資料夾選單、分頁容器
        play_layout,                                                             # 追加參數：呼叫端已傳入的 play_layout，這裡先不使用
        ):    
                
        if sport == 'Deadlift':
            folderPath = 'C:/Users/92A27/MOCAP/recordings'
            self.folders[sport] = folderPath
            Deadlift_btn.setStyleSheet("font-size:18px;background-color: #888888")
            Benchpress_btn.setStyleSheet("font-size:18px;background-color: #666666")
            Squat_btn.setStyleSheet("font-size:18px;background-color: #666666")
        
        elif sport == 'Benchpress':
            folderPath = 'C:/Users/92A27/benchpress/recordings'
            self.folders[sport] = folderPath
            Benchpress_btn.setStyleSheet("font-size:18px;background-color: #888888")
            Squat_btn.setStyleSheet("font-size:18px;background-color: #666666")
            Deadlift_btn.setStyleSheet("font-size:18px;background-color: #666666")

        elif sport == 'Squat':
            folderPath = 'C:/Users/92A27/barbell_squat/recordings'               # 指定 Squat 錄影資料夾
            self.folders[sport] = folderPath                                     # 記住資料夾
            Squat_btn.setStyleSheet("font-size:18px;background-color:#888888")   # ✅ 選中的應該是 Squat
            Benchpress_btn.setStyleSheet("font-size:18px;background-color:#666666") # 其他置灰
            Deadlift_btn.setStyleSheet("font-size:18px;background-color:#666666")   # 其他置灰


        File_comboBox.clear()
        self.all_items = os.listdir(self.folders[sport])
        # 這裡combobox有變動
        for folder in self.all_items[::-1]:
            File_comboBox.addItems([folder])
            
        Play_btn.setEnabled(True)
        Stop_btn.setEnabled(True)
        Frameslider.setEnabled(True)
        fast_forward_combobox.setEnabled(True)


    # 讀取combobox內的資料夾
    def File_combobox_TextChanged(self, file_comboBox, play_btn, icons, Frameslider):
        videofolder = file_comboBox.currentText()                             # 目前選取的資料夾
        root_dir = self.folders.get(self.currentsport, "")                             # 取得目前運動類型對應的根資料夾
        self.folder = os.path.join(root_dir, videofolder) if videofolder else None     # 設定目前選取資料夾的絕對路徑
        folder = self.folders[self.currentsport]                              # 對應運動類別的根資料夾
        all_videos = glob.glob(f'{folder}/{videofolder}/*.avi')               # 把資料夾下所有 avi 撈出
        self.datas = []                                                       # 清空資料曲線

        # --- 依運動類型定義「優先片名」清單（依序嘗試） ---
        # 目標：最後挑出 3 支（頭 + 兩個底部）
        if self.currentsport == 'Squat':
            # 可能的命名：vision2/3/4/5、或 original_vision*、或 *drawed 版本
            desired_groups = [
                ('original_vision2.avi', 'vision3.avi', 'vision6.avi'),  # 再嘗試原始
                ('vision2.avi', 'vision3.avi', 'vision6.avi'),    # 先嘗試有疊圖
            ]
            0
        elif self.currentsport == 'Deadlift':
            desired_groups = [
                ('vision1_drawed.avi', 'vision2.avi', 'vision3.avi'),
                ('vision1.avi', 'vision2.avi', 'vision3.avi'),
            ]
        else:  # Benchpress
            desired_groups = [
                ('original_vision1.avi', 'original_vision2.avi', 'vision3.avi'),
                ('vision1.avi', 'vision2.avi', 'vision3.avi'),
            ]

        # --- 依優先順序取出最貼近的一組 ---
        picked = []
        base_names = {os.path.basename(v): v for v in all_videos}             # 映射檔名→完整路徑
        for group in desired_groups:                                          # 按序嘗試
            candidate = [base_names.get(name) for name in group if name in base_names]  # 取到就加
            if len(candidate) == 3:                                           # 找到完整三支
                picked = candidate
                break
            if not picked and len(candidate) >= 1:                            # 先記下至少一支，避免全軍覆沒
                picked = candidate

        self.videos = picked                                                  # 實際使用清單
        self.info_data = []                                                   # Squat 目前不載 JSON（你的原碼註解掉）
        self.pred_data = []                                                   # 同上

        # --- 重置 pixmap 容器，避免累積 ---
        self.rp_qpixmaps = []                                                # ✅ 先清空
        for _ in range(len(self.videos)):                                     # 依影片數建立空 pixmap
            self.rp_qpixmaps.append(QtGui.QPixmap())

        # --- 統一進入 stop 狀態，會觸發 showprevision() 嘗試顯示第一張 ---
        self.stop(Frameslider, play_btn, icons)                                # 停止→預覽

        # --- 偵錯訊息（可留可去） ---
        if not self.videos:
            print(f"[Replay][{self.currentsport}] 於資料夾 {videofolder} 找不到可用影片")  # 幫助你查錯

    
    def play_btn_clicked(self, fast_forward_combobox, Play_btn, icons, Frameslider):             # 播放鍵點擊處理
        import os, cv2, threading                                                                 # 需求模組
        self.index += 1                                                                           # 切換播放/暫停狀態計數

        # ======= 進入「播放」狀態 =======
        if self.index % 2 == 1:                                                                   # 奇數次：播放
            fast_forward_combobox.setEnabled(False)                                               # 播放中禁用倍率調整
            Frameslider.setEnabled(True)                                                          # 啟用拖動條
            Play_btn.setIcon(icons[0])                                                            # 換成「暫停」圖示

            # --- 來源驗證：過濾可用影片 ---
            valid_videos = []                                                                     # 可用影片清單
            f_num = []                                                                            # 每支片的總幀數
            for video in getattr(self, "videos", []):                                             # 逐一檢查 self.videos
                if not video or not os.path.exists(video):                                        # 路徑不存在
                    continue                                                                      # 略過
                cap = cv2.VideoCapture(video)                                                     # 嘗試開檔
                if cap.isOpened():                                                                # 成功開檔
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))                               # 讀總幀數
                    if frames > 0:                                                                # 幀數有效
                        valid_videos.append(video)                                                # 收錄可用影片
                        f_num.append(frames)                                                      # 收錄幀數
                    cap.release()                                                                 # 關檔
                else:
                    cap.release()                                                                 # 開檔失敗也關
                    continue                                                                      # 略過

            if not valid_videos:                                                                  # 沒有任何可播放影片
                # 這裡可視需求彈警示窗或印 log
                print("[Replay] 無可用影片：請先選擇正確的檔案。")                                   # 提示
                # 回復 UI 狀態
                fast_forward_combobox.setEnabled(True)                                            # 允許調倍率
                Frameslider.setEnabled(False)                                                     # 關閉拖動條
                Play_btn.setIcon(icons[1])                                                        # 換回「播放」圖示
                self.index -= 1                                                                   # 還原狀態
                return                                                                            # 中止

            framenumber = min(f_num)                                                              # 同步播放上限（取最短片）

            # --- stop 後的重新播放（重建 threads） ---
            if getattr(self, "is_stop", False):                                                   # 若之前是 stop
                print('threads created')                                                          # log
                self.is_stop = False                                                              # 清除 stop 狀態

                # 清理舊 threads / caps
                if getattr(self, "threads", None):                                                # 有舊執行緒
                    self.del_mythreads()                                                          # 自訂清理
                if getattr(self, "caps", None) is not None:                                       # 有舊 cap
                    self.caps.clear()                                                             # 清空
                else:
                    self.caps = []                                                                # 初始化

                # 建立播放 barrier（以可用影片數量為準）
                self.barrier_play = threading.Barrier(len(valid_videos))                          # 同步點（影片數）

                # 若有資料曲線要播放，只有在有資料時才建立 barrier
                info_count = len(getattr(self, "info_data", []))                                  # 資料筆數
                self.barrier_data = (threading.Barrier(info_count) if info_count > 0 else None)   # 無資料則 None

                # 重建播放 threads（使用 valid_videos）
                self.threads = []                                                                 # 重建容器
                self.videos = valid_videos                                                        # 以有效片覆寫
                self.caps = []                                                                    # 對應 cap 清單
                for i, video in enumerate(self.videos):                                           # 逐支影片
                    cap = cv2.VideoCapture(video)                                                 # 重新開檔
                    self.caps.append(cap)                                                         # 收 cap
                    thread_play = MyThread(                                                       # 建立播放執行緒
                        self.caps, i, Play_btn, icons, fast_forward_combobox,                     # 參數同原本
                        Frameslider, framenumber, self.rp_Vision_labels,                          # 參數同原本
                        self.rp_qpixmaps, self.barrier_play)                                      # 參數同原本
                    thread_play.start()                                                           # 啟動
                    self.threads.append(thread_play)                                              # 收執行緒

                # 建立資料曲線 threads（只有在 self.datas / info_data 有內容時）
                if getattr(self, "datas", False) and info_count > 0:                              # 有資料才跑
                    for i, data in enumerate(self.info_data):                                     # 逐筆資料
                        data_thread = Thread_data(                                                # 建立資料執行緒
                            i, self.data_graph, data, self.barrier_data,                          # 圖表/資料/同步點
                            fast_forward_combobox, Frameslider, framenumber                       # 控制/同步幀
                        )
                        data_thread.start()                                                       # 啟動
                        self.threads.append(data_thread)                                          # 收執行緒

            # --- pause 後繼續 ---
            else:
                print('resume')                                                                   # log
                for thread in getattr(self, "threads", []):                                       # 逐一喚醒
                    try:
                        thread.resume()                                                           # 喚醒執行緒
                    except Exception as e:
                        print(f"[Replay] resume 失敗: {e}")                                       # 失敗記錄

        # ======= 進入「暫停」狀態 =======
        else:                                                                                     # 偶數次：暫停
            self.pause_event(fast_forward_combobox, Play_btn, icons)                              # 呼叫暫停邏輯


    def pause_event(self, fast_forward_combobox, Play_btn, icons):
        fast_forward_combobox.setEnabled(True)
        Play_btn.setIcon(icons[1])
        for thread in self.threads:
            thread.pause()
        
    # threads 全部刪除，重新播放
    def del_mythreads(self):
        if self.threads:
            for thread in self.threads:
                thread._stop_event.set()
            self.threads.clear()

    def tab_changed(self):
        self.del_mythreads()
        self.is_stop = True
        self.index = 0
    
    def slider_released(self):
        for thread in self.threads:
            thread.is_slide_end = True

    def slider_Pressed(self):
        for thread in self.threads:
            thread.is_slide_start = True

    def sliding(self, Frameslider, TimeCount_LineEdit):
        # 控制秒數
        fps = 30
        val = Frameslider.value()
        sec = val / fps
        minute = "%02d" % int(sec / 60)
        second = "%02d" % int(sec % 60)
        TimeCount_LineEdit.setText(f'{minute}:{second}')
        
    def closeEvent(self, event):
        self.del_mythreads()
        event.accept()
        
    def showprevision(self):
        
        if not hasattr(self, "data_graph") or self.data_graph.get("axes") is None:
            print("⚠️ data_graph 尚未初始化，跳過繪圖流程")
            return

        if self.videos:
            for i, video in enumerate(self.videos):
                temp_cap = cv2.VideoCapture(video)
                if self.ocv:
                    _ , frame = temp_cap.read()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = QtGui.QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QtGui.QImage.Format_RGB888)
                    self.rp_qpixmaps[i] = QtGui.QPixmap.fromImage(image)
                    scaled_pixmap = self.rp_qpixmaps[i].scaled(self.rp_Vision_labels[i].size(), QtCore.Qt.IgnoreAspectRatio)
                    self.rp_Vision_labels[i].setPixmap(scaled_pixmap)
        if self.datas:
            for i, data in enumerate(self.info_data):
                x_data = data['frames']
                y_data = data['values']
                min_length = min(len(x_data), len(y_data))
                x_data = x_data[:min_length]
                y_data = y_data[:min_length]
                y_min = data['y_min']
                y_max = data['y_max']
                
                self.data_graph['axes'][i].clear()
                self.data_graph['axes'][i].set_ylim(y_min, y_max)
                self.data_graph['axes'][i].plot(x_data, y_data, label = f"{data['title']}")
                self.data_graph['axes'][i].set_ylabel(f"{data['y_label']}")
                self.data_graph['axes'][i].legend()
                
            self.data_graph['axes'][-1].set_xlabel('frames')
            self.data_graph['canvas'].draw()
            self.data_graph['graphicscene'].addWidget(self.data_graph['canvas'])
            confs = []
            for NoSet, info in self.pred_data['results'].items():
                score = info[0]
                item = QtWidgets.QTableWidgetItem(f"{str(round(float(score)*100, 1))}")
                # Set font properties (e.g., bold, size 12)
                font = QtGui.QFont("Arial", 24, QtGui.QFont.Bold)
                item.setFont(font)
                # Set the alignment (e.g., center)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(0, int(NoSet), item)
                temp = []
                for conf in info[1]:
                    temp.append(round(conf[1]*100))
                confs = confs + temp
            for i, panel in enumerate(self.conf_panels):
                label = QtWidgets.QLabel(f"{str(confs[i])}%")
                label.setStyleSheet("font-size:20px; color: #070807; border: none;")
                # Allow the label to resize automatically
                label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                font = QtGui.QFont("Arial", 24, QtGui.QFont.Bold)  # Set font to Arial, size 14, bold
                label.setFont(font)
                label.setAlignment(QtCore.Qt.AlignCenter)  # Center-align the text

                panel_layout = QtWidgets.QVBoxLayout()
                panel_layout.addWidget(label)
                panel.setLayout(panel_layout)
        else:
            for ax in self.data_graph['axes']:
                ax.clear()
            self.data_graph['canvas'].draw()
                
    def creat_vision_labels_pixmaps(self, labelsize, parentlayout, sublayout, sport, num, type ='rc'):
        vertical_sliders = []
        horizontal_sliders = []
        Vision_labels = []
        qpixmaps = []
        if sport == 'Deadlift':
            for _ in range(num):
                qpixmap = QtGui.QPixmap()
                qpixmaps.append(qpixmap)
                Vision_label = QtWidgets.QLabel(parentlayout)
                Vision_label.setFrameShape(QtWidgets.QFrame.Panel)
                Vision_label.setMinimumSize(labelsize[0], labelsize[1])
                Vision_label.setMaximumSize(labelsize[0], labelsize[1])
                Vision_label.setPixmap(qpixmap)
                Vision_label.setText('')
                sublayout.addWidget(Vision_label)
                sublayout.setAlignment(Vision_label, QtCore.Qt.AlignCenter)
                Vision_labels.append(Vision_label)
            return Vision_labels, qpixmaps

        if sport == 'Squat':
            for _ in range(num):
                qpixmap = QtGui.QPixmap()
                qpixmaps.append(qpixmap)
                Vision_label = QtWidgets.QLabel(parentlayout)
                Vision_label.setFrameShape(QtWidgets.QFrame.Panel)
                Vision_label.setMinimumSize(labelsize[0], labelsize[1])
                Vision_label.setMaximumSize(labelsize[0], labelsize[1])
                Vision_label.setPixmap(qpixmap)
                Vision_label.setText('')
                sublayout.addWidget(Vision_label)
                sublayout.setAlignment(Vision_label, QtCore.Qt.AlignCenter)
                Vision_labels.append(Vision_label)
            return Vision_labels, qpixmaps
        
        if sport == 'Benchpress':
            if type == 'rc':
                for _ in range(num):
                    qpixmap = QtGui.QPixmap()
                    qpixmaps.append(qpixmap)
                    Vision_label = QtWidgets.QLabel(parentlayout)
                    Vision_label.setFrameShape(QtWidgets.QFrame.Panel)
                    Vision_label.setMinimumSize(labelsize[0], labelsize[1])
                    Vision_label.setMaximumSize(labelsize[0], labelsize[1])
                    Vision_label.setPixmap(qpixmap)
                    Vision_label.setText('')
                    sublayout.addWidget(Vision_label)
                    sublayout.setAlignment(Vision_label, QtCore.Qt.AlignCenter)
                    Vision_labels.append(Vision_label)
                return Vision_labels, qpixmaps
            if type == 'rp':   
                if num == 1:
                    vertical_slider = QtWidgets.QSlider(orientation = QtCore.Qt.Vertical, parent = parentlayout)
                    horizontal_slider = QtWidgets.QSlider(orientation = QtCore.Qt.Horizontal, parent = parentlayout)
                    qpixmap = QtGui.QPixmap()
                    qpixmaps.append(qpixmap)
                    Vision_label = LineLabel(parentlayout)
                    Vision_label.setFrameShape(QtWidgets.QFrame.Panel)
                    Vision_label.setMinimumSize(labelsize[0], labelsize[1])
                    Vision_label.setMaximumSize(labelsize[0], labelsize[1])
                    Vision_label.setPixmap(qpixmap)
                    sublayout.addWidget(Vision_label, 0, 0)
                    sublayout.addWidget(vertical_slider, 0, 1)
                    horizontal_slider.setFixedWidth(labelsize[0])
                    horizontal_slider.setValue(0)
                    horizontal_slider.setMaximum(labelsize[0])
                    horizontal_slider.valueChanged.connect(Vision_label.set_horizontal_line)
                    vertical_slider.setFixedHeight(labelsize[1])
                    vertical_slider.setMaximum(labelsize[1])
                    vertical_slider.setInvertedAppearance(True)
                    vertical_slider.setValue(0)
                    vertical_slider.valueChanged.connect(Vision_label.set_vertical_line)
                    sublayout.addWidget(horizontal_slider, 1, 0)
                    Vision_labels.append(Vision_label)
                    return Vision_label, vertical_slider, horizontal_slider
                
                if  num == 2:
                    for _ in range(num):
                        # ✅ 創建新元件，避免重複使用舊的
                        qpixmap = QtGui.QPixmap()
                        Vision_label = LineLabel(parentlayout)
                        Vision_label.setMinimumSize(labelsize[0], labelsize[1])
                        Vision_label.setMaximumSize(labelsize[0], labelsize[1])
                        Vision_label.setPixmap(qpixmap)

                        vertical_slider = QtWidgets.QSlider(QtCore.Qt.Vertical, parent = parentlayout)
                        horizontal_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, parent = parentlayout)

                        vertical_slider.setFixedHeight(labelsize[1])  # 限制垂直 Slider 高度
                        vertical_slider.setMaximum(labelsize[1])
                        vertical_slider.setInvertedAppearance(True)
                        vertical_slider.setValue(0)
                        vertical_slider.valueChanged.connect(Vision_label.set_vertical_line)
                        horizontal_slider.setFixedWidth(labelsize[0])  # 限制水平 Slider 寬度
                        horizontal_slider.setMaximum(labelsize[0])
                        horizontal_slider.setValue(0)
                        horizontal_slider.valueChanged.connect(Vision_label.set_horizontal_line)

                        # ✅ 建立 GridLayout
                        vis_layout = QtWidgets.QGridLayout()
                        vis_layout.addWidget(Vision_label, 0, 0)
                        vis_layout.addWidget(vertical_slider, 0, 1)
                        vis_layout.addWidget(horizontal_slider, 1, 0, 1, 2)

                        # ✅ 包裝 GridLayout 進 QWidget，才能加入 HLayout
                        temp_widget = QtWidgets.QWidget()
                        temp_widget.setLayout(vis_layout)
                        sublayout.addWidget(temp_widget, alignment=QtCore.Qt.AlignCenter)  # 讓 Widget 置中
                        Vision_labels.append(Vision_label)
                        vertical_sliders.append(vertical_slider)
                        horizontal_sliders.append(horizontal_slider)
                    return Vision_labels, vertical_sliders, horizontal_sliders
        
    
    def creat_graphic(self, parentlayout, sublayout, size, num):
        figure = Figure(figsize=size)
        canvas = FigureCanvas(figure)
        axes = figure.subplots(num, 1, sharex=True)
        with open(f'./config/Deadlift_data/Score.json', mode='r', encoding='utf-8') as file:
            data = json.load(file)
            for NoSet, info in data['results'].items():
                pass
        # 創建 QGraphicsView 和 QGraphicsScene
        graphicview = QtWidgets.QGraphicsView(parentlayout)
        graphicscene = QtWidgets.QGraphicsScene(parentlayout)

        # **建立外部表格**
        table = QtWidgets.QTableWidget(2, int(NoSet)+1)
        table.setVerticalHeaderLabels(["Score", "Confidence"])
        table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter)
        table.verticalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter)
        table.horizontalHeader().hide()
        table.verticalHeader().hide()
        for row in range(table.rowCount()):
            for column in range(table.columnCount()):
                item = table.item(row, column)
                if item:
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
         # **將第 2 列的每一個儲存格內加入 QHBoxLayout 並分成 4 小區塊**
        self.conf_panels = []
        for col in range(int(NoSet)+1):  # 遍歷 A, B, C 欄
            if col % 2 ==0:
                font = 'background-color: #eaf0e9; border: 1px solid black;'
            else:
                font = 'background-color: #d0d4be; border: 1px solid black;'
            cell_widget = QtWidgets.QWidget()  # 創建 QWidget 作為容器
            layout_inside = QtWidgets.QHBoxLayout(cell_widget)  # 創建 QHBoxLayout
            layout_inside.setContentsMargins(0, 0, 0, 0)  # 移除邊距
            layout_inside.setSpacing(5)  # 設定間距

            # **建立 4 個 Panel**
            for i in range(4):
                panel = QtWidgets.QFrame()
                panel.setStyleSheet(font)
                panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                self.conf_panels.append(panel)
                layout_inside.addWidget(panel)  # 加入 Layout
            table.setCellWidget(1, col, cell_widget)  # **將 QWidget 設為 CellWidget**

        # **外部表格自適應大小**
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table.setFixedSize(2500, 200)  # **設定表格大小**
        
        # **將圖表和表格加入場景**
        canvas_proxy = graphicscene.addWidget(canvas)
        table_proxy = graphicscene.addWidget(table)

        # **確保場景大小與 `graphicview` 一致**
        graphicview.setScene(graphicscene) 

        # **讓表格置中**
        scene_width = graphicview.sceneRect().width()
        table_x = (scene_width - table.width()) / 2
        table_proxy.setPos(table_x, 10)  # **表格放在上方**
        canvas_proxy.setPos(10, table.height() + 20)  # **圖表放在表格下方**

        # 設定 Layout
        sublayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, graphicview)
        sublayout.setFormAlignment(QtCore.Qt.AlignCenter)

        return graphicview, graphicscene, canvas, axes, table



    def stop(self, Frameslider, Play_btn, icons):
        print('stop')
        Frameslider.setEnabled(False)
        self.del_mythreads()
        self.is_stop = True
        self.index = 0
        Play_btn.setIcon(icons[1])
        Frameslider.setSliderPosition(0)
        self.showprevision()
            
    def slider_changed(self, Frameslider, Play_btn, icons):
        val = Frameslider.value()
        if val >= Frameslider.maximum():
            self.stop(Frameslider, Play_btn, icons)
            
    def search_text_changed(self, comboBox, filter_text):
        comboBox.clear()                                                       # 清空重建
        text = (filter_text or "").lower()                                    # 轉小寫並避免 None
        filtered = [item for item in self.all_items if text in item.lower()]  # 兩邊都小寫
        comboBox.addItems(filtered)                                           # 加回符合項

        
    # 遍歷 layout，清空所有子佈局和控件
    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()  # 刪除控件
            else:
                sub_layout = item.layout()
                if sub_layout:
                    self.clear_layout(sub_layout)  # 遞迴刪除子佈局
        layout.update()  # 更新佈局，確保視圖刷新


    def data_produce_btn_clicked_rp(self, sport):                                  # Replay 的 data produce 主流程
        # === 檢查目前是否已選到有效資料夾（沿用 ui.on_data_produce_clicked 的保護） ===   # 說明
        folder = getattr(self, "folder", None)                                      # 取目前儲存的資料夾路徑
        if not folder or not os.path.isdir(folder):                                 # 檢查不存在或不是資料夾
            QtWidgets.QMessageBox.warning(                                          # 跳提示視窗
                None, "注意", "請先在下拉選單選擇一個有效的資料夾！"                       # 與原 UI 一致的訊息
            )                                                                       
            return                                                                  # 中止流程

        # === 以下維持你原本的最小指令流程（僅把 self.folder 換成本地變數 folder 使用） === # 說明
        if sport == 'Deadlift':                                                     # Deadlift 流程
            os.system(f'python ./tools/Deadlift_tool/interpolate.py {folder}')      # 槓端與骨架內插
            os.system(f'python ./tools/Benchpress_tool/bar_data_produce.py {folder} --out ./config --sport deadlift')  # bar
            os.system(f'python ./tools/Deadlift_tool/data_produce.py {folder} --out ./config')                         # angle
            os.system(f'python ./tools/Deadlift_tool/data_split.py {folder}')       # split
            os.system(f'python ./tools/Deadlift_tool/predict.py {folder} --out ./config')                               # predict

        if sport == 'Benchpress':                                                   # Benchpress 流程
            os.system(f'python ./tools/Benchpress_tool/interpolate.py {folder}')      # 槓端與骨架內插
            os.system(f'python ./tools/Benchpress_tool/bar_data_produce.py {folder} --out ./config --sport benchpress')   # bar（保留你原設定）
            # os.system(f'python ./tools/Benchpress_tool/step0_hampel_bar.py {self.folder}')
            # os.system(f'python ./tools/Benchpress_tool/step0_hampel_yolo_ske_rear.py {self.folder}')
            # os.system(f'python ./tools/Benchpress_tool/step0_hampel_yolo_ske_top.py {self.folder} ')
            # os.system(f'python ./tools/Benchpress_tool/step1_interpolate_bar.py {self.folder}')
            # os.system(f'python ./tools/Benchpress_tool/step2_interpolate_yolo_ske.py {self.folder}')
            # os.system(f'python ./tools/Benchpress_tool/step3_autocutting_0801.py {self.folder}')
            # os.system(f'python ./tools/Benchpress_tool/step5_calculate_angle_new_feature_test.py {self.folder}')
            # os.system(f'python ./tools/Benchpress_tool/step6_cut.py {self.folder} ')
            # os.system(f'python ./tools/Benchpress_tool/step7_length_100.py {self.folder}')
            # os.system(f'python ./tools/Benchpress_tool/step8_normalize.py {self.folder}')

        if sport == 'Squat':                                                        # Squat 流程
            pass                                                                    # 目前無動作（保留）

        os.system(f'python ./tools/trajectory.py {folder}')                         # 後製軌跡影片
        print('執行完成')                                                            # 完成訊息