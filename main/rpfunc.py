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
        self.vertical_line_x = 0                                         # 第1條垂直線 X
        self.vertical_line_x2 = None                                     # 第2條垂直線 X（None 表示不畫）
        self.horizontal_line_y = 0                                       # 水平線 Y

    def set_vertical_line(self, value: int):
        self.vertical_line_x = int(value)                                # 更新第1條垂直線 X
        self.update()                                                    # 重新觸發繪製

    def set_vertical_line2(self, value: int):
        self.vertical_line_x2 = int(value)                               # 更新第2條垂直線 X
        self.update()                                                    # 重新觸發繪製

    def set_horizontal_line(self, value: int):
        self.horizontal_line_y = int(value)                              # 更新水平線 Y
        self.update()                                                    # 重新觸發繪製

    def paintEvent(self, event):
        super().paintEvent(event)                                        # 保持 QLabel 原本的行為
        painter = QPainter(self)                                         # 建立畫筆
        painter.setRenderHint(QPainter.Antialiasing)                     # 抗鋸齒
        pen = QPen(QtCore.Qt.red, 3, QtCore.Qt.SolidLine)                # 紅色 3px
        painter.setPen(pen)                                              # 套用畫筆

        # 垂直線（第1條）
        painter.drawLine(self.vertical_line_x, 0, self.vertical_line_x, self.height())  # 畫第1條垂直線

        # 垂直線（第2條，如有設定）
        if self.vertical_line_x2 is not None:                            # 有設定才畫
            painter.drawLine(self.vertical_line_x2, 0, self.vertical_line_x2, self.height())  # 畫第2條垂直線

        # 水平線
        painter.drawLine(0, self.horizontal_line_y, self.width(), self.horizontal_line_y)      # 畫水平線
        painter.end()                                                     # 結束畫圖

        
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

# class Thread_data(threading.Thread):
#     def __init__(self, index, gragh, data, barrier, fast_forward_combobox, Frameslider, framenumber):
class Thread_data(threading.Thread):
    def __init__(self, index, gragh, data, barrier, fast_forward_combobox, Frameslider, framenumber, sport):

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

        # ✅ Benchpress 的 Bar_Position 顯示「起槓位置」(y=300~350)
        self.sport = sport  # 記住運動類型  #
        if (str(self.sport).lower() == 'benchpress' 
            and str(self.data.get('title', '')).lower() == 'bar_position'):
            self.ax.axhspan(280, 330, alpha=0.18, color='orange', zorder=0)  # 起槓帶狀區  #
            # 用 y 軸混合座標：x 用座標系(0~1)、y 用資料座標
            self.ax.text(0.98, 280, "Top Position", transform=self.ax.get_yaxis_transform(),
                         va='bottom', ha='right', fontsize=20, color="#ff9a3c")  # 文字標註  #


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
                          'Benchpress' : ['Bar_Position.json', 'right_elbow_torsor_angle_top.json', 
                                          'left_elbow_torsor_angle_top.json'],
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
    # 讀取combobox內的資料夾
    def File_combobox_TextChanged(self, file_comboBox, play_btn, icons, Frameslider):
        selected_subdir = file_comboBox.currentText()                                 # 目前選到的子資料夾名稱（純字串）
        sport_root = self.folders.get(self.currentsport, "")                          # 依當前運動類型取得根資料夾
        self.folder = os.path.join(sport_root, selected_subdir) if selected_subdir else None  # 把完整路徑存到狀態 self.folder
        all_videos = glob.glob(f'{sport_root}/{selected_subdir}/*.avi')               # 掃描該子資料夾下的 .avi
        self.datas = []                                                                # 清空資料曲線

        # 依運動類型定義候選影片組（會按序挑到完整的三支）
        if self.currentsport == 'Squat':
            desired_groups = [
                ('original_vision2.avi', 'vision3.avi', 'vision6.avi'),                # 優先原始命名
                ('vision2.avi', 'vision3.avi', 'vision6.avi'),                         # 次選一般命名
            ]                                                                          # ← 你原本多打一個孤立的 0，已移除
        elif self.currentsport == 'Deadlift':
            desired_groups = [
                ('vision1_drawed.avi', 'vision2.avi', 'vision3.avi'),
            ('vision1.avi', 'vision2.avi', 'vision3.avi'),
            ]
        else:  # Benchpress
            desired_groups = [
                ('vision1_drawed.avi', 'original_vision2.avi', 'vision3.avi'),
                ('original_vision1.avi', 'original_vision2.avi', 'vision3.avi'),
            ]

        picked = []                                                                    # 最終選用的影片清單
        base_map = {os.path.basename(p): p for p in all_videos}                        # 檔名 → 完整路徑對照表
        for group in desired_groups:                                                   # 依優先順序嘗試
            candidate = [base_map[name] for name in group if name in base_map]         # 該組有的就收
            if len(candidate) == 3:                                                    # 三支都齊就用這組
                picked = candidate
                break
            if not picked and len(candidate) >= 1:                                     # 先記下至少有一支的組（避免全沒）
                picked = candidate

        self.videos = picked                                                           # 儲存實際要用的影片清單
        self.info_data = []                                                            # （維持原設計）先清空
        self.pred_data = []                                                            # （維持原設計）先清空

        # 重置 pixmap 容器，避免殘留
        self.rp_qpixmaps = [QtGui.QPixmap() for _ in range(len(self.videos))]          # 為每支影片準備空 pixmap

        # 切回 stop 狀態並顯示預覽第一幀
        self.stop(Frameslider, play_btn, icons)                                        # 停止 → 預覽

        # 除錯訊息
        if not self.videos:                                                            # 找不到可用影片時提示
            print(f"[Replay][{self.currentsport}] 於資料夾 {selected_subdir} 找不到可用影片")

    
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
                        data_thread = Thread_data(
                            i, self.data_graph, data, self.barrier_data,
                            fast_forward_combobox, Frameslider, framenumber,
                            self.currentsport  # ✅ 傳入當前運動別（Benchpress/Deadlift/Squat）
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
        
    def showprevision(self):                                                       # 預覽＋繪圖（依運動別限制張數）
        # === 安全檢查 ===
        if not hasattr(self, "data_graph") or self.data_graph.get("axes") is None: # 無圖表物件
            print("⚠️ data_graph 尚未初始化，跳過繪圖流程")                          # 訊息
            return                                                                 # 結束

        # === 讀資料 ===
        sport  = getattr(self, "currentsport", "") or ""                           # 運動別
        folder = getattr(self, "folder", None)                                     # 選取資料夾
        max_charts = self._plot_count_for(sport)                                   # 本次最大圖數
        self.info_data, self.pred_data = self._load_info_from_folder(sport, folder)# 讀入資料
        self.datas = bool(self.info_data)                                          # 有資料才畫

        # === 顯示影片第一幀（略，維持原本流程） ===
        if getattr(self, "videos", None):                                          # 有影片時
            for i, video in enumerate(self.videos):                                # 逐檔
                temp_cap = cv2.VideoCapture(video)                                 # 開檔
                if self.ocv and temp_cap.isOpened():                               # 可讀
                    _, frame = temp_cap.read()                                     # 讀第一幀
                    if frame is not None:                                          # 有畫面
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)         # BGR→RGB
                        image = QtGui.QImage(frame_rgb.data, frame_rgb.shape[1],
                                            frame_rgb.shape[0], QtGui.QImage.Format_RGB888)  # QImage
                        self.rp_qpixmaps[i] = QtGui.QPixmap.fromImage(image)       # 建 Pixmap
                        scaled = self.rp_qpixmaps[i].scaled(self.rp_Vision_labels[i].size(),
                                                            QtCore.Qt.IgnoreAspectRatio)      # 縮放
                        self.rp_Vision_labels[i].setPixmap(scaled)                 # 顯示
                temp_cap.release()                                                 # 關檔

        # === 畫資料曲線或清空 ===
        axes = self.data_graph['axes']                                             # 取得子圖列
        for ax in axes:                                                            # 先清空全部
            ax.clear()                                                             # 清空

        if self.datas:                                                             # 有資料
            # 只畫到 max_charts 或 axes 長度的較小者
            n_plot = min(max_charts, len(axes), len(self.info_data))               # 本次實際繪製數
            for i, data in enumerate(self.info_data[:n_plot]):                     # 逐條畫
                x_data = data['frames']                                            # X
                y_data = data['values']                                            # Y
                min_len = min(len(x_data), len(y_data))                            # 對齊
                x_data = x_data[:min_len]                                          # 截斷
                y_data = y_data[:min_len]                                          # 截斷

                ax = axes[i]                                                       # 目標子圖
                ax.set_ylim(data['y_min'], data['y_max'])                          # 設 y 範圍
                if isinstance(y_data[0], (list, tuple)) and len(y_data[0]) == 2:   # 雙路資料
                    r_vals = [v[0] for v in y_data]                                # 右側
                    l_vals = [v[1] for v in y_data]                                # 左側
                    ax.plot(x_data, r_vals, label=f"{data['title']}-R")            # 畫線 R
                    ax.plot(x_data, l_vals, label=f"{data['title']}-L")            # 畫線 L
                else:                                                              # 一維
                    ax.plot(x_data, y_data, label=f"{data['title']}")              # 畫線
                ax.set_ylabel(f"{data['y_label']}")                                # y 標籤
                ax.legend()                                                        # 圖例

                # ✅ Benchpress 的 Bar_Position 顯示「起槓位置」(y=300~350)
                if (str(self.currentsport).lower() == 'benchpress'
                    and str(data.get('title', '')).lower() == 'bar_position'):
                    ax.axhspan(280, 330, alpha=0.18, color='orange', zorder=0)   # 起槓帶狀區  #
                    ax.text(0.98, 280, "Top Position", transform=ax.get_yaxis_transform(),
                            va='bottom', ha='right', fontsize=20, color="#ff9a3c")                 # 文字標註  #


            # 清理多餘子圖（若 axes 比 n_plot 多）
            for j in range(n_plot, len(axes)):                                     # 其餘子圖
                axes[j].clear()                                                    # 清空
                axes[j].set_visible(False)                                         # 暫時隱藏

            # 讓用到的子圖可見並補上 x 標籤
            for j in range(n_plot):                                                # 已繪製區塊
                axes[j].set_visible(True)                                          # 顯示
            if n_plot > 0:                                                         # 有圖才加 x 標
                axes[n_plot-1].set_xlabel('frames')                                # 最底下一張加 x 標

            self.data_graph['canvas'].draw()                                       # 重繪
            self.data_graph['graphicscene'].addWidget(self.data_graph['canvas'])   # 掛回場景

            # === 若有 Score.json 的 results，就更新表格/面板 ===
            if isinstance(self.pred_data, dict) and 'results' in self.pred_data:   # 有預測
                confs = []                                                         # 累積 conf%
                for NoSet, info in self.pred_data['results'].items():              # 逐 set
                    score = info[0]                                                # 取 score
                    item = QtWidgets.QTableWidgetItem(f"{str(round(float(score)*100, 1))}")  # 表格 cell
                    font = QtGui.QFont("Arial", 24, QtGui.QFont.Bold)              # 字體
                    item.setFont(font)                                             # 套字體
                    item.setTextAlignment(QtCore.Qt.AlignCenter)                   # 置中
                    self.table.setItem(0, int(NoSet), item)                        # 放入第 0 列
                    temp = [round(c[1]*100) for c in info[1]]                      # 類別置信度%
                    confs.extend(temp)                                             # 收集
                for i, panel in enumerate(getattr(self, 'conf_panels', [])):       # 逐 panel
                    lbl = QtWidgets.QLabel(f"{str(confs[i])}%") if i < len(confs) else QtWidgets.QLabel("")  # 有值才顯示
                    lbl.setStyleSheet("font-size:20px; color:#070807; border:none;")# 樣式
                    lbl.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)      # 自適應
                    font = QtGui.QFont("Arial", 24, QtGui.QFont.Bold)              # 字體
                    lbl.setFont(font)                                              # 套字體
                    lbl.setAlignment(QtCore.Qt.AlignCenter)                        # 置中
                    lay = QtWidgets.QVBoxLayout()                                  # 佈局
                    lay.addWidget(lbl)                                             # 加入
                    panel.setLayout(lay)                                           # 掛上
        else:                                                                      # 沒資料
            for ax in axes:                                                        # 全部子圖
                ax.clear()                                                         # 清空
                ax.set_visible(True)                                               # 顯示（避免下次仍隱藏）
            self.data_graph['canvas'].draw()                                       # 重繪空畫布


    def _load_info_from_folder(self, sport, folder):                              # 僅從選到的資料夾載入 JSON（含上限）
        """
        只在 `folder` 內尋找 JSON，不回退；依運動別限制曲線數量（Benchpress=3，其它=4）。      # 行為說明
        """
        import os, json, numpy as np                                              # 近域匯入
        if not sport or not folder or not os.path.isdir(folder):                  # 基本健檢
            return [], {}                                                         # 無效就回空

        max_charts = self._plot_count_for(sport)                                  # 本次允許的曲線上限
        wanted_names = list(self.data_path.get(sport, []))                        # 既有宣告清單
        if not wanted_names:                                                      # 若未宣告
            wanted_names = [f for f in os.listdir(folder)                         # 掃描資料夾
                            if f.lower().endswith('.json')]                      # 只收 json

        info_data, pred_data, seen_titles = [], {}, set()                         # 容器
        for name in wanted_names:                                                 # 逐檔名
            if len(info_data) >= max_charts:                                      # 已達上限
                break                                                             # 停止載入
            file_path = os.path.join(folder, name)                                # 組路徑
            if not (name.lower().endswith('.json') and os.path.isfile(file_path)):# 檔案檢查
                continue                                                          # 跳過

            try:
                with open(file_path, 'r', encoding='utf-8') as f:                 # 開檔
                    j = json.load(f)                                              # 解析
            except Exception as e:
                print(f"⚠️ 讀檔失敗：{file_path} -> {e}")                            # 記錄錯誤
                continue                                                          # 跳過

            base_title = os.path.splitext(os.path.basename(name))[0]              # 取標題
            if base_title in seen_titles:                                         # 去重
                continue                                                          # 跳過
            seen_titles.add(base_title)                                           # 登記

            if base_title.lower() == 'score' and isinstance(j, dict) and 'results' in j:  # Score 特案
                pred_data = j                                                     # 存評分資料
                continue                                                          # 不當曲線畫

            frames = j.get('frames', None) if isinstance(j, dict) else None       # 標準 x 欄
            values = j.get('values', None) if isinstance(j, dict) else None       # 標準 y 欄

            if frames is None or values is None:                                  # 缺欄位→嘗試自動抽取
                candidate_vals = None                                             # 候選 y
                if isinstance(j, dict):                                           # 掃 dict
                    for k, v in j.items():                                        # 每鍵
                        if k.lower() in ('frames','frame','index','idx'):         # 排除 x 類鍵
                            continue                                              # 繼續
                        if isinstance(v, list) and len(v) > 1:                    # 有長度
                            if isinstance(v[0], (int,float)) or (isinstance(v[0], list) and len(v[0]) in (2,)):  # 1D 或 pair
                                candidate_vals = v                                 # 接受為 y
                                break                                              # 停止找
                if candidate_vals is None and isinstance(j, list) and j and isinstance(j[0], dict):  # list[dict] 形
                    xs, ys = [], []                                               # 暫存
                    for row in j:                                                 # 逐筆
                        if 'value' in row:                                        # 有 y
                            ys.append(row['value'])                                # 收 y
                            xs.append(row.get('frame', len(xs)+1))                 # 無 frame 用流水號
                    if ys:                                                         # 有資料
                        frames, values = xs, ys                                    # 指派

            if values is None or frames is None:                                   # 還是沒有
                continue                                                           # 跳過

            # 計算 y 範圍（容錯）
            try:
                arr = np.array(values, dtype=float)                                # 轉陣列
                y_min = float(np.nanmin(arr))                                      # 最小
                y_max = float(np.nanmax(arr))                                      # 最大
            except Exception:                                                      # 混雜非數值
                flatted = []                                                       # 扁平收集
                if isinstance(values, list) and values and isinstance(values[0], (list,tuple)) and len(values[0]) == 2:  # pair
                    for v0, v1 in values:
                        try: flatted += [float(v0), float(v1)]
                        except: pass
                else:
                    for v in values:
                        try: flatted.append(float(v))
                        except: pass
                if not flatted:                                                    # 無可用
                    continue                                                       # 跳過
                y_min, y_max = min(flatted), max(flatted)                          # 重新取得

            yr  = y_max - y_min if y_max > y_min else 1.0                          # 範圍
            pad = max(yr * 0.05, 1e-3)                                             # 邊界
            y_lo, y_hi = y_min - pad, y_max + pad                                  # 帶 padding

            min_len = min(len(frames), len(values))                                # 對齊長度
            if min_len <= 0:                                                       # 無資料
                continue                                                           # 跳過

            y_label = j.get('y_label', base_title) if isinstance(j, dict) else base_title  # y 標籤
            info_data.append({                                                     # 累積曲線
                'title': base_title,                                               # 標題
                'frames': list(frames)[:min_len],                                  # X 資料（裁齊）
                'values': list(values)[:min_len],                                  # Y 資料（裁齊）
                'y_min': y_lo,                                                     # y 下界
                'y_max': y_hi,                                                     # y 上界
                'y_label': y_label                                                 # y 標籤
            })                                                                     # 追加一條

        return info_data, pred_data                                                # 回傳


    def _plot_count_for(self, sport):                                             # 傳回該運動欲顯示的圖數
        return 3 if str(sport).lower() == 'benchpress' else 4                     # Benchpress=3, 其他=4


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
                    qpixmap = QtGui.QPixmap()                                                                 # 建立空白 QPixmap
                    qpixmaps.append(qpixmap)                                                                  # 收集 pixmap
                    Vision_label = QtWidgets.QLabel(parentlayout)                                             # 影像顯示 QLabel
                    Vision_label.setFrameShape(QtWidgets.QFrame.Panel)                                        # 外框樣式
                    Vision_label.setMinimumSize(labelsize[0], labelsize[1])                                   # 固定大小（寬, 高）
                    Vision_label.setMaximumSize(labelsize[0], labelsize[1])                                   # 固定大小（寬, 高）
                    Vision_label.setPixmap(qpixmap)                                                           # 指定 pixmap
                    Vision_label.setText('')                                                                  # 清空文字
                    sublayout.addWidget(Vision_label)                                                         # 佈局加入
                    sublayout.setAlignment(Vision_label, QtCore.Qt.AlignCenter)                               # 置中
                    Vision_labels.append(Vision_label)                                                        # 收集 label
                return Vision_labels, qpixmaps                                                                # 回傳

            if type == 'rp':   
                if num == 1:
                    vertical_slider = QtWidgets.QSlider(orientation=QtCore.Qt.Vertical, parent=parentlayout)  # 垂直 slider（沿 Y 方向擺放）  #
                    horizontal_slider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal, parent=parentlayout)  # 水平 slider（沿 X 方向擺放）  #
                    qpixmap = QtGui.QPixmap()                                                                 # 建立空白 QPixmap  #
                    qpixmaps.append(qpixmap)                                                                  # 收集 pixmap  #

                    Vision_label = LineLabel(parentlayout)                                                    # 自訂 LineLabel：能畫多條垂直線＋水平線  #
                    Vision_label.setFrameShape(QtWidgets.QFrame.Panel)                                        # 外框樣式  #
                    Vision_label.setMinimumSize(labelsize[0], labelsize[1])                                   # 固定大小（寬, 高）  #
                    Vision_label.setMaximumSize(labelsize[0], labelsize[1])                                   # 固定大小（寬, 高）  #
                    Vision_label.setPixmap(qpixmap)                                                           # 指定 pixmap  #

                    sublayout.addWidget(Vision_label, 0, 0)                                                   # 影像放左上格  #
                    sublayout.addWidget(vertical_slider, 0, 1)                                                # 垂直 slider 放影像右側  #

                    horizontal_slider.setFixedWidth(labelsize[0])                                             # 水平 slider 寬度=影像寬  #
                    horizontal_slider.setValue(0)                                                             # 初值 0  #
                    horizontal_slider.setMaximum(labelsize[0])                                                # 最大值=影像寬（對應 X）→ 控制「第1條垂直線 X」  #

                    vertical_slider.setFixedHeight(labelsize[1])                                              # 垂直 slider 高度=影像高  #
                    vertical_slider.setMaximum(labelsize[1])                                                  # 最大值=影像高（對應 Y）→ 控制水平線 Y  #
                    vertical_slider.setInvertedAppearance(True)                                               # 由上往下數值增  #

                    # === 控制對應 ===
                    horizontal_slider.valueChanged.connect(Vision_label.set_vertical_line)                    # 水平 slider1 → 垂直線1（X）  #
                    vertical_slider.setValue(0)                                                               # 初值 0  #
                    vertical_slider.valueChanged.connect(Vision_label.set_horizontal_line)                    # 垂直 slider → 水平線（Y）  #

                    sublayout.addWidget(horizontal_slider, 1, 0)                                              # 水平 slider1 放影像下方  #

                    # ---------- ★ 新增：第二個水平 slider，控制第2條垂直線 ----------
                    horizontal_slider2 = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal, parent=parentlayout)  # ★ 第二條垂直線用的 slider  #
                    horizontal_slider2.setFixedWidth(labelsize[0])                                           # ★ 寬度與影像一致  #
                    horizontal_slider2.setValue(0)                                                           # ★ 初值 0  #
                    horizontal_slider2.setMaximum(labelsize[0])                                              # ★ 最大值=影像寬  #
                    horizontal_slider2.valueChanged.connect(Vision_label.set_vertical_line2)                 # ★ slider2 → 垂直線2（X）  #
                    sublayout.addWidget(horizontal_slider2, 2, 0)                                            # ★ 放在第一個 slider 的下方  #
                    # -------------------------------------------------------------

                    Vision_labels.append(Vision_label)                                                        # 收集 label  #
                    return Vision_label, vertical_slider, horizontal_slider                                   # 回傳（保持舊介面，避免動到其它呼叫點）  #


                        
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
                        vertical_slider.valueChanged.connect(Vision_label.set_horizontal_line)
                        horizontal_slider.setFixedWidth(labelsize[0])  # 限制水平 Slider 寬度
                        horizontal_slider.setMaximum(labelsize[0])
                        horizontal_slider.setValue(0)
                        horizontal_slider.valueChanged.connect(Vision_label.set_vertical_line)

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
            os.system(f'python ./tools/Benchpress_tool/offline_benchpress_head.py "{folder}"')                       # 接著跑頭部/槓端流程           
            os.system(f'python ./tools/Benchpress_tool/interpolate.py "{folder}"')                                 # 若要做骨架/槓端內插再開

            os.system(f'python ./tools/Benchpress_tool/step0_hampel_bar.py "{folder}"')                              # 先做 Hampel 濾波
            os.system(f'python ./tools/Benchpress_tool/bar_data_produce.py {folder} --out ./config --sport benchpress')  # bar         
               
            os.system(f'python ./tools/Benchpress_tool/step0_hampel_yolo_ske_rear.py {folder}')
            os.system(f'python ./tools/Benchpress_tool/step0_hampel_yolo_ske_top.py {folder} ')
            os.system(f'python ./tools/Benchpress_tool/step1_interpolate_bar.py {folder}')
            os.system(f'python ./tools/Benchpress_tool/step2_interpolate_yolo_ske.py {folder}')


            os.system(f'python ./tools/Benchpress_tool/torsor_angle_produce.py {folder}')
            os.system(f'python ./tools/Benchpress_tool/step3_autocutting_0801.py {folder}')
            os.system(f'python ./tools/Benchpress_tool/step5_calculate_angle_new_feature_test.py {folder}')
            os.system(f'python ./tools/Benchpress_tool/step6_cut.py {folder} ')
            os.system(f'python ./tools/Benchpress_tool/step7_length_100.py {folder}')
            os.system(f'python ./tools/Benchpress_tool/step8_normalize.py {folder}')

        if sport == 'Squat':                                                        # Squat 流程
            pass                                                                    # 目前無動作（保留）

        os.system(f'python ./tools/trajectory.py {folder}')                         # 後製軌跡影片
        print('執行完成')                                                            # 完成訊息