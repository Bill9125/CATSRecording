from PyQt5 import QtCore, QtGui, QtWidgets
import os, glob, sys, time
import cv2, threading
from PyQt5.QtGui import QPainter, QPen
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import json
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject
import matplotlib.ticker as ticker

from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot, QThread  # ✅ 加上QThread/pyqtSlot  # 匯入QThread與pyqtSlot供背景執行用

class BusyOverlay(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, text="資料處理中，請稍候…"):
        super().__init__(parent)                                                                 # 蓋在父視窗上
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)                         # 攔截滑鼠以達到鎖定
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool)                      # 無邊框、置於父視窗上層
        self.setStyleSheet("background: rgba(0,0,0,120);")                                       # 半透明黑遮罩
        self.setGeometry(parent.rect())                                                          # 尺寸覆蓋整個父視窗

        lay = QtWidgets.QVBoxLayout(self)                                                        # 垂直置中容器
        lay.setAlignment(QtCore.Qt.AlignCenter)                                                  # 內容置中
        box = QtWidgets.QFrame(self)                                                             # 內部白色卡片
        box.setStyleSheet("background: #222; color:#fff; border-radius:12px; padding:24px;")     # 卡片樣式
        v = QtWidgets.QVBoxLayout(box)                                                           # 卡片內排版
        lbl = QtWidgets.QLabel(text, box)                                                        # 文字
        lbl.setAlignment(QtCore.Qt.AlignCenter)                                                  # 置中
        bar = QtWidgets.QProgressBar(box)                                                        # 進度條
        bar.setRange(0, 0)                                                                       # 不定進度（馬拉松條）
        v.addWidget(lbl)                                                                         # 放入卡片
        v.addWidget(bar)                                                                         # 放入卡片
        lay.addWidget(box, 0, QtCore.Qt.AlignCenter)                                             # 卡片置中於遮罩

    def showEvent(self, e):
        self.setGeometry(self.parent().rect())                                                   # 顯示時再對齊尺寸
        super().showEvent(e)                                                                     # 呼叫父類

    def eventFilter(self, obj, event):
        if obj is self.parent() and event.type() == QtCore.QEvent.Resize:                        # 父視窗調整大小
            self.setGeometry(self.parent().rect())                                               # 跟著更新遮罩大小
        return super().eventFilter(obj, event)                                                   # 交回預設行為


class PendingDialog(QtWidgets.QDialog):  # ✅ 等待中的彈窗  # 提供處理中提示的模態對話框
    def __init__(self, parent=None, text="資料處理中，請稍候…"):  # 建構子  # 設定提示文字
        super().__init__(parent)  # 呼叫父類  # 初始化QDialog
        self.setWindowTitle("處理中")  # 視窗標題  # 設定標題
        self.setModal(True)  # 模態  # 阻擋其他操作
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)  # 不可手動關閉  # 避免誤關
        lay = QtWidgets.QVBoxLayout(self)  # 直向排版  # 安排子元件
        self.label = QtWidgets.QLabel(text, self)  # 提示標籤  # 顯示訊息
        self.label.setAlignment(QtCore.Qt.AlignCenter)  # 置中  # 美觀
        self.bar = QtWidgets.QProgressBar(self)  # 進度條  # 表示進行中
        self.bar.setRange(0, 0)  # 不定進度  # 轉圈效果
        lay.addWidget(self.label)  # 加入版面  # 顯示文字
        lay.addWidget(self.bar)  # 加入版面  # 顯示進度
        

class DataProduceWorker(QObject):  # ✅ 背景工作者  # 真正執行長任務的物件
    finished = pyqtSignal()  # 完成訊號  # 任務結束通知UI
    error = pyqtSignal(str)  # 錯誤訊號  # 任務中斷時通知錯誤
    log = pyqtSignal(str)  # 紀錄訊號  # 回報即時指令

    def __init__(self, sport: str, folder: str):  # 建構子  # 帶入運動別與資料夾
        super().__init__()  # 呼叫父類  # 初始化QObject
        self.sport = sport  # 存運動別  # Deadlift/Benchpress/Squat
        self.folder = folder  # 存資料夾  # 當前選擇目錄

    def _run_cmd(self, cmd: str) -> int:  # 回傳 exit code  # 方便後續記錄但不阻斷
        self.log.emit(cmd)                # 回報正在執行     # 統一從這裡列印
        code = os.system(cmd)             # 同步執行         # 等待完成
        if code != 0:
            self.log.emit(f"[warn] exit={code}: {cmd}")  # 僅記錄警告   # 不拋例外
        return code                       # 回傳碼給上層     # 供統計或日後檢查


    @pyqtSlot()
    def run(self):                                          # 背景主流程       # 由 QThread 啟動
        # 不使用 try/except 把流程整個包住以免提前 finished  # 改為只在最終 emit
        folder = self.folder                                # 當前資料夾       # 簡化變數
        sport  = self.sport                                 # 運動別

        steps = []                                          # 預先組裝所有步驟 # 以便統一迭代
        if sport == 'Deadlift':                             # Deadlift 流程
            steps = [
                f'python ./tools/Deadlift_tool/interpolate.py "{folder}"',
                f'python ./tools/Benchpress_tool/bar_data_produce.py "{folder}" --out ./config --sport deadlift',
                f'python ./tools/Deadlift_tool/data_produce.py "{folder}" --out ./config',
                f'python ./tools/Deadlift_tool/data_split.py "{folder}"',
                f'python ./tools/Deadlift_tool/predict.py "{folder}" --out ./config',
            ]                                               # 依原順序
        elif sport == 'Benchpress':                         # Benchpress 流程
            steps = [
                f'python ./tools/Benchpress_tool/offline_benchpress_head.py "{folder}"',
                f'python ./tools/Benchpress_tool/interpolate.py "{folder}"',
                f'python ./tools/Benchpress_tool/step0_hampel_bar.py "{folder}"',
                f'python ./tools/Benchpress_tool/bar_data_produce.py "{folder}" --out ./config --sport benchpress',
                f'python ./tools/Benchpress_tool/step0_hampel_yolo_ske_rear.py "{folder}"',
                f'python ./tools/Benchpress_tool/step0_hampel_yolo_ske_top.py "{folder}"',
                f'python ./tools/Benchpress_tool/step1_interpolate_bar.py "{folder}"',
                f'python ./tools/Benchpress_tool/step2_interpolate_yolo_ske.py "{folder}"',
                f'python ./tools/Benchpress_tool/torsor_angle_produce.py "{folder}"',
                f'python ./tools/Benchpress_tool/step3_autocutting_0801.py "{folder}"',
                f'python ./tools/Benchpress_tool/step5_calculate_angle_new_feature_test.py "{folder}"',
                f'python ./tools/Benchpress_tool/step6_cut.py "{folder}"',
                f'python ./tools/Benchpress_tool/step7_length_100.py "{folder}"',
                f'python ./tools/Benchpress_tool/step8_normalize.py "{folder}"',
            ]                                               # 依原順序
        elif sport == 'Squat':  # ✅ [修正] 補上 Squat 的執行步驟
                    steps = [
                        f'python ./tools/Deadlift_tool/interpolate.py "{folder}"',  # 執行內插 (共用 Deadlift 工具)
                        f'python ./tools/Benchpress_tool/bar_data_produce.py "{folder}" --out ./config --sport squat',  # 產生 Bar 數據 (參數改為 squat)
                        f'python ./tools/Deadlift_tool/data_produce.py "{folder}" --out ./config',  # 產生角度數據 (共用 Deadlift 工具)
                        f'python ./tools/Deadlift_tool/data_split.py "{folder}"',  # 資料分割 (共用 Deadlift 工具)
                    ]

        for cmd in steps:  # 遍歷所有步驟
            self._run_cmd(cmd)  # 執行指令

        # 軌跡圖
        self._run_cmd(f'python ./tools/trajectory.py "{folder}"')  # 最後執行軌跡繪製

        print('執行完成')  # 印出完成訊息
        self.finished.emit()  # 發送完成訊號，通知 UI 解鎖


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
        # －－ 放在 Replaybackend.__init__ 內初始化一次 －－
        self._overlay = None                               # 記錄遮罩物件  # 初始為 None
        self._locked_parent = None                         # 記錄被鎖的父視窗  # 初始為 None

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
            head_label, bottom_labels, graph  # ✅ [修正] 參數名修正為 graph 以對應傳入物件
            ):
            self.currentsport = 'Squat'  # 設定當前運動為 Squat
            self.rp_Vision_labels = head_label + bottom_labels  # 組合影像標籤
            self.data_graph = graph  # ✅ [修正] 正確指派給 self.data_graph，圖表才能繪製
            self.rp_btn_press(
                self.currentsport, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn, icons,
                Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout,
                )  # 呼叫共用的按鈕處理函式
        
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

        # === 新增：把之後 refresh 需要用到的 UI 參考存起來 ===
        self._file_combo = File_comboBox       # 之後要重建清單並維持選項
        self._play_btn = Play_btn              # 之後要呼叫 TextChanged 時需要
        self._icons = icons                    # 同上
        self._frameslider = Frameslider        # 同上



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
                ('vision1_drawed.avi', 'vision5.avi', 'vision6.avi'),                # 優先原始命名
                ('vision1.avi', 'vision5.avi', 'vision6.avi'),                         # 次選一般命名
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
        
    def showprevision(self):
            # 1. 檢查圖表物件是否初始化
            if not hasattr(self, "data_graph") or self.data_graph.get("axes") is None:  # 若無圖表物件
                return  # 直接結束

            # 2. 讀取資料與路徑
            sport  = getattr(self, "currentsport", "") or ""  # 取得當前運動類型
            folder = getattr(self, "folder", None)  # 取得當前資料夾路徑
            max_charts = self._plot_count_for(sport)  # 取得該運動允許的最大圖表數
            self.info_data, self.pred_data = self._load_info_from_folder(sport, folder)  # 讀取 JSON 資料
            self.datas = bool(self.info_data)  # 標記是否有資料

            # 3. 顯示影片第一幀預覽
            if getattr(self, "videos", None):  # 若有影片列表
                for i, video in enumerate(self.videos):  # 逐一處理影片
                    temp_cap = cv2.VideoCapture(video)  # 開啟影片檔案
                    if self.ocv and temp_cap.isOpened():  # 若 OpenCV 啟用且開啟成功
                        _, frame = temp_cap.read()  # 讀取第一幀
                        if frame is not None:  # 若讀取成功
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR 轉 RGB
                            image = QtGui.QImage(frame_rgb.data, frame_rgb.shape[1],
                                                frame_rgb.shape[0], QtGui.QImage.Format_RGB888)  # 轉換為 QImage
                            self.rp_qpixmaps[i] = QtGui.QPixmap.fromImage(image)  # 建立 QPixmap
                            scaled = self.rp_qpixmaps[i].scaled(self.rp_Vision_labels[i].size(),
                                                                QtCore.Qt.IgnoreAspectRatio)  # 縮放至 Label 大小
                            self.rp_Vision_labels[i].setPixmap(scaled)  # 顯示圖片
                    temp_cap.release()  # 釋放影片資源

            # 4. 準備繪圖：清空舊圖
            axes = self.data_graph['axes']  # 取得所有子圖 (Axes)
            for ax in axes:  # 遍歷所有子圖
                ax.clear()  # 清空內容

            # 5. 開始繪製曲線
            if self.datas:  # 若有資料
                n_plot = min(max_charts, len(axes), len(self.info_data))  # 計算實際要畫幾張圖
                for i, data in enumerate(self.info_data[:n_plot]):  # 逐一處理每一筆資料
                                    # --- 準備資料 ---
                                    x_data = data['frames']  # 取得 X 軸數據 (幀數)
                                    y_data = data['values']  # 取得 Y 軸數據 (數值)
                                    min_len = min(len(x_data), len(y_data))  # 確保長度一致
                                    x_data = x_data[:min_len]  # 裁切 X
                                    y_data = y_data[:min_len]  # 裁切 Y

                                    ax = axes[i]  # 取得對應的子圖
                                    title_lower = str(data.get('title', '')).lower()  # 取得標題並轉小寫

                                    # ==========================================
                                    # ✅ [設定區域] 自定義 Y 軸上下限、刻度 與 字體大小
                                    # ==========================================
                                    
                                    # 1. 設定預設值 (Default Values)
                                    final_y_min = data['y_min']  # 預設最小值
                                    final_y_max = data['y_max']  # 預設最大值
                                    
                                    # ✅ [字體設定] 定義預設字體大小 (您可以依需求微調這裡)
                                    font_label_size = 10  # 預設 Y 軸標題大小
                                    font_tick_size = 8    # 預設 刻度數字大小
                                    font_legend_size = 8  # 預設 圖例文字大小
                                    
                                    # --- 針對不同檔案設定 ---
                                    if 'bar_position' in title_lower:  # 若為槓鈴位置
                                        # Bar_Position (像素座標)
                                        final_y_min, final_y_max = 50, 550  # (您設定的參數)   #squat 50,600; bp 200, 400
                                        ax.yaxis.set_major_locator(ticker.MultipleLocator(50)) # (您設定的參數)

                                        # ✅ [修改] 針對此圖表加大字體
                                        font_label_size = 16
                                        font_tick_size = 14
                                        font_legend_size = 16
                                        
                                    elif 'hip_angle' in title_lower:  # 若為髖關節角度
                                        # Hip_Angle (角度)
                                        final_y_min, final_y_max = 80, 180  # (您設定的參數)
                                        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))  # (您設定的參數)

                                        # ✅ [修改] 針對此圖表加大字體
                                        font_label_size = 16
                                        font_tick_size = 14
                                        font_legend_size = 16

                                    elif 'knee_angle' in title_lower:  # 若為膝關節角度
                                        # Knee_Angle (角度)
                                        final_y_min, final_y_max = 50, 180  # (您設定的參數)
                                        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))  # (您設定的參數)
                                        
                                        # ✅ [修改] 針對此圖表加大字體
                                        font_label_size = 16
                                        font_tick_size = 14
                                        font_legend_size = 16

                                    elif 'knee_to_hip' in title_lower:  # 若為膝髖距離
                                        # Knee_to_Hip (距離或比例)
                                        final_y_min, final_y_max = 0.2, 2.2  # (您設定的參數)
                                        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))  # (您設定的參數)
                                        
                                        # ✅ [修改] 針對此圖表加大字體
                                        font_label_size = 16
                                        font_tick_size = 14
                                        font_legend_size = 16

                                    elif 'right_elbow_torsor_angle_top' in title_lower: 
                                        final_y_min, final_y_max = 30, 100  # (您設定的參數)
                                        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))  # (您設定的參數)
                                        
                                        # ✅ [修改] 針對此圖表加大字體
                                        font_label_size = 16
                                        font_tick_size = 14
                                        font_legend_size = 16

                                    elif 'left_elbow_torsor_angle_top' in title_lower:  
                                        final_y_min, final_y_max = 30, 100  # (您設定的參數)
                                        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))  # (您設定的參數)
                                        
                                        # ✅ [修改] 針對此圖表加大字體
                                        font_label_size = 16
                                        font_tick_size = 14
                                        font_legend_size = 16                                    
                                    
                                    else:  # 其他情況
                                        ax.yaxis.set_major_locator(ticker.AutoLocator())  # 自動刻度
                                        # ✅ [修改] 針對此圖表加大字體
                                        font_label_size = 16
                                        font_tick_size = 14
                                        font_legend_size = 16

                                    # ==========================================
                                    # 6. 套用設定並繪圖
                                    # ==========================================
                                    
                                    # 套用上下限
                                    ax.set_ylim(final_y_min, final_y_max)  # 設定 Y 軸範圍
                                    
                                    # ✅ 套用刻度字體大小
                                    ax.tick_params(axis='both', which='major', labelsize=font_tick_size)
                                    
                                    # 繪製格線 (讓刻度更清楚)
                                    ax.grid(True, linestyle='--', alpha=0.5)  # 顯示虛線網格

                                    # 繪製線條 (區分單線或雙線)
                                    if isinstance(y_data[0], (list, tuple)) and len(y_data[0]) == 2:  # 若為雙線數據 (左右)
                                        r_vals = [v[0] for v in y_data]  # 取右側數據
                                        l_vals = [v[1] for v in y_data]  # 取左側數據
                                        ax.plot(x_data, r_vals, label=f"{data['title']}-R")  # 繪製右線
                                        ax.plot(x_data, l_vals, label=f"{data['title']}-L")  # 繪製左線
                                    else:  # 若為單線數據
                                        ax.plot(x_data, y_data, label=f"{data['title']}")  # 繪製線條
                                    
                                    # ✅ 套用 Y 軸標題與字體大小
                                    ax.set_ylabel(f"{data['y_label']}", fontsize=font_label_size)
                                    
                                    # ✅ 套用 圖例與字體大小
                                    ax.legend(fontsize=font_legend_size)

                                    # # Benchpress 特殊標記 (保持原本邏輯)
                                    # if (str(self.currentsport).lower() == 'benchpress'
                                    #     and 'bar_position' in title_lower):  # 若為臥推且是 Bar Position
                                    #     ax.axhspan(280, 330, alpha=0.18, color='orange', zorder=0)  # 標示 Top Position 區域
                                    #     ax.text(0.98, 280, "Top Position", transform=ax.get_yaxis_transform(),
                                    #             va='bottom', ha='right', fontsize=20, color="#ff9a3c")  # 標示文字

                # 7. 清理多餘的子圖 (隱藏沒用到的)
                for j in range(n_plot, len(axes)):  # 遍歷剩餘的子圖
                    axes[j].clear()  # 清空
                    axes[j].set_visible(False)  # 隱藏

                # 確保用到的子圖是顯示的
                for j in range(n_plot):  # 遍歷使用中的子圖
                    axes[j].set_visible(True)  # 顯示
                
                # 最底下的圖加上 X 軸標籤
                if n_plot > 0:  # 若有圖
                    axes[n_plot-1].set_xlabel('frames')  # 設定 X 軸標籤

                # 8. 更新畫布與場景
                self.data_graph['canvas'].draw()  # 重繪畫布
                self.data_graph['graphicscene'].addWidget(self.data_graph['canvas'])  # 將畫布加入場景

                # 9. 更新分數表格 (如果有的話)
                if isinstance(self.pred_data, dict) and 'results' in self.pred_data:  # 若有預測結果
                    confs = []  # 初始化信心度列表
                    for NoSet, info in self.pred_data['results'].items():  # 遍歷每一組結果
                        score = info[0]  # 取得分數
                        item = QtWidgets.QTableWidgetItem(f"{str(round(float(score)*100, 1))}")  # 建立表格項目
                        font = QtGui.QFont("Arial", 24, QtGui.QFont.Bold)  # 設定字體
                        item.setFont(font)  # 套用字體
                        item.setTextAlignment(QtCore.Qt.AlignCenter)  # 文字置中
                        if hasattr(self, 'table') and self.table:  # 若表格存在
                            # 容錯：檢查 row/col 範圍
                            if int(NoSet) < self.table.columnCount():  # 確保欄位不越界
                                self.table.setItem(0, int(NoSet), item)  # 設定儲存格內容
                        temp = [round(c[1]*100) for c in info[1]]  # 取得信心度
                        confs.extend(temp)  # 加入列表
                    
                    if hasattr(self, 'conf_panels'):  # 若有信心度面板
                        for i, panel in enumerate(getattr(self, 'conf_panels', [])):  # 逐一處理面板
                            # 清空舊 layout
                            if panel.layout():  # 若已有佈局
                                QtWidgets.QWidget().setLayout(panel.layout())  # 透過替換 Widget 清除引用
                            
                            lbl = QtWidgets.QLabel(f"{str(confs[i])}%") if i < len(confs) else QtWidgets.QLabel("")  # 建立標籤
                            lbl.setStyleSheet("font-size:20px; color:#070807; border:none;")  # 設定樣式
                            lbl.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)  # 設定大小策略
                            font = QtGui.QFont("Arial", 24, QtGui.QFont.Bold)  # 設定字體
                            lbl.setFont(font)  # 套用字體
                            lbl.setAlignment(QtCore.Qt.AlignCenter)  # 置中
                            lay = QtWidgets.QVBoxLayout()  # 建立垂直佈局
                            lay.addWidget(lbl)  # 加入標籤
                            panel.setLayout(lay)  # 套用佈局
            else:  # 若無資料
                # 無資料時清空畫面
                for ax in axes:  # 遍歷所有子圖
                    ax.clear()  # 清空
                    ax.set_visible(True)  # 顯示空圖
                self.data_graph['canvas'].draw()  # 重繪畫布


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
            figure = Figure(figsize=size)  # 建立 Matplotlib 圖形物件
            canvas = FigureCanvas(figure)  # 建立畫布
            axes = figure.subplots(num, 1, sharex=True)  # 建立子圖，共用 X 軸
            
            # ✅ [修正] 增加讀取 JSON 的容錯機制
            NoSet = 5  # 設定預設組數，避免讀檔失敗時變數未定義
            try:
                with open(f'./config/Deadlift_data/Score.json', mode='r', encoding='utf-8') as file:  # 嘗試開啟分數檔案
                    data = json.load(file)  # 載入 JSON
                    if 'results' in data:  # 確認有無 results 欄位
                        for NoSet, info in data['results'].items():  # 遍歷取得最後一組的 Key
                            pass  # 僅為了取得最後的 NoSet 值
            except:
                pass  # 若檔案不存在或格式錯誤，直接忽略，使用預設 NoSet

            graphicview = QtWidgets.QGraphicsView(parentlayout)  # 建立圖表視圖
            graphicscene = QtWidgets.QGraphicsScene(parentlayout)  # 建立圖表場景

            table = QtWidgets.QTableWidget(2, int(NoSet)+1)  # 建立表格，根據 NoSet 決定欄數
            table.setVerticalHeaderLabels(["Score", "Confidence"])  # 設定垂直表頭
            table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter)  # 水平置中
            table.verticalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter)  # 垂直置中
            table.horizontalHeader().hide()  # 隱藏水平表頭
            table.verticalHeader().hide()  # 隱藏垂直表頭
            for row in range(table.rowCount()):  # 遍歷列
                for column in range(table.columnCount()):  # 遍歷欄
                    item = table.item(row, column)  # 取得單元格
                    if item:
                        item.setTextAlignment(QtCore.Qt.AlignCenter)  # 設定文字置中
            
            self.conf_panels = []  # 初始化信心度面板列表
            for col in range(int(NoSet)+1):  # 遍歷每一欄
                if col % 2 ==0:
                    font = 'background-color: #eaf0e9; border: 1px solid black;'  # 偶數欄樣式
                else:
                    font = 'background-color: #d0d4be; border: 1px solid black;'  # 奇數欄樣式
                cell_widget = QtWidgets.QWidget()  # 建立儲存格內的 Widget
                layout_inside = QtWidgets.QHBoxLayout(cell_widget)  # 建立內部水平佈局
                layout_inside.setContentsMargins(0, 0, 0, 0)  # 移除邊距
                layout_inside.setSpacing(5)  # 設定間距

                for i in range(4):  # 建立 4 個小區塊
                    panel = QtWidgets.QFrame()  # 建立 Frame
                    panel.setStyleSheet(font)  # 套用樣式
                    panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)  # 設定大小策略
                    self.conf_panels.append(panel)  # 加入列表
                    layout_inside.addWidget(panel)  # 加入佈局
                table.setCellWidget(1, col, cell_widget)  # 將 Widget 設定到表格中

            table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)  # 表格寬度自適應
            table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)  # 表格高度自適應
            table.setFixedSize(2500, 200)  # 固定表格尺寸
            
            canvas_proxy = graphicscene.addWidget(canvas)  # 將畫布加入場景
            table_proxy = graphicscene.addWidget(table)  # 將表格加入場景

            graphicview.setScene(graphicscene)  # 視圖設定場景

            scene_width = graphicview.sceneRect().width()  # 取得場景寬度
            table_x = (scene_width - table.width()) / 2  # 計算表格置中位置
            table_proxy.setPos(table_x, 10)  # 設定表格位置
            canvas_proxy.setPos(10, table.height() + 20)  # 設定畫布位置 (在表格下方)

            sublayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, graphicview)  # 將視圖加入主佈局
            sublayout.setFormAlignment(QtCore.Qt.AlignCenter)  # 設定佈局置中

            return graphicview, graphicscene, canvas, axes, table  # 回傳建立的物件



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

    def _lock_ui(self, parent: QtWidgets.QWidget, text="資料處理中，請稍候…"):
        """顯示半透明遮罩並鎖住 parent（不彈窗）。"""                             # 函式說明
        if parent is None:                                                             # 保險：沒有父就找目前視窗
            parent = QtWidgets.QApplication.activeWindow()                             # 取當前視窗
            if parent is None:
                return                                                                 # 找不到就放棄

        # 若已經有舊遮罩，先清掉（避免重入）
        self._unlock_ui()                                                               # 先嘗試解一次  # 保險

        self._locked_parent = parent                                                   # 記住這次被鎖的 parent
        self._overlay = BusyOverlay(parent, text)                                      # 建立遮罩
        parent.installEventFilter(self._overlay)                                       # 讓遮罩跟隨尺寸
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)                 # 游標改沙漏
        parent.setEnabled(False)                                                       # 真正鎖互動
        self._overlay.show()                                                           # 顯示遮罩
        self._overlay.raise_()                                                         # 置頂（確保在最上層）


    def _unlock_ui(self):
        """關閉遮罩並恢復互動與顏色。"""                                              # 函式說明
        try:
            QtWidgets.QApplication.restoreOverrideCursor()                             # 還原游標
        except Exception:
            pass

        # 正確移除事件過濾器與刪除遮罩
        if getattr(self, "_overlay", None) is not None:
            try:
                if getattr(self, "_locked_parent", None) is not None:
                    self._locked_parent.removeEventFilter(self._overlay)               # 移除 filter（關鍵）
            except Exception:
                pass
            try:
                self._overlay.hide()                                                   # 先隱藏
                self._overlay.setParent(None)                                          # 解除父子，避免殘影
                self._overlay.deleteLater()                                            # 排程刪除
            except Exception:
                pass
            self._overlay = None                                                       # 清引用

        # 把同一個父視窗解鎖（不要用 activeWindow）
        if getattr(self, "_locked_parent", None) is not None:
            try:
                self._locked_parent.setEnabled(True)                                   # 恢復互動（關鍵）
                self._locked_parent.repaint()                                          # 立即重繪一次
            except Exception:
                pass
            self._locked_parent = None                                                 # 清引用

        QtWidgets.QApplication.processEvents()                                         # 沖一下事件，立刻生效



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


    def data_produce_btn_clicked_rp(self, sport):                                                    # 按下資料產出
        folder = getattr(self, "folder", None)                                                       # 目前資料夾
        if not folder or not os.path.isdir(folder):                                                  # 檢查有效性
            QtWidgets.QMessageBox.warning(None, "注意", "請先在下拉選單選擇一個有效的資料夾！")          # 導引
            return                                                                                   # 中止
        if not sport:                                                                                # 檢查運動別
            QtWidgets.QMessageBox.warning(None, "注意", "請先選擇運動類型（Deadlift/Benchpress/Squat）。")
            return                                                                                   # 中止

        parent = QtWidgets.QApplication.activeWindow()                                               # 找父視窗
        if parent is None:                                                                           # 萬一找不到
            parent = QtWidgets.QWidget()                                                             # 建立臨時父
        self._lock_ui(parent, "資料處理中，請稍候…")                                                   # ★ 鎖定整個介面（不彈窗）

        # === 建立背景執行緒與工作者（保持你原本的 DataProduceWorker，不動 interpolate） ===
        self._dp_thread = QThread()                                                                  # 建 QThread
        self._dp_worker = DataProduceWorker(sport, folder)                                           # 建 Worker
        self._dp_worker.moveToThread(self._dp_thread)                                                # 移入執行緒
        self._dp_thread.started.connect(self._dp_worker.run)                                         # 開始即執行
        self._dp_worker.finished.connect(self._dp_thread.quit)                                       # 任務完畢→退出執行緒
        self._dp_worker.finished.connect(self._dp_worker.deleteLater)                                # 釋放 Worker
        self._dp_thread.finished.connect(self._dp_thread.deleteLater)                                # 釋放 Thread

        # ★ 建議改為：只接一次完成事件，統一在 slot 裡做解鎖＋刷新
        # （若想保留原本保險也行，不會出錯）
        self._dp_worker.finished.connect(self._refresh_after_produce)                                    # 完成後刷新當前資料夾 

        self._dp_worker.error.connect(lambda msg: print(f"[data_produce][error] {msg}"))                 # 只記錄錯誤 
        self._dp_worker.log.connect(lambda s: print(f"[data_produce] {s}"))                              # 即時 log 
        self._dp_thread.start()                                            

    def _refresh_after_produce(self):                                                             # 背景任務完成後的刷新  #
        self._unlock_ui()                                                                         # 先確保解鎖  #
        try:
            sport = self.currentsport or ''                                                       # 目前運動別  #
            root = self.folders.get(sport, '')                                                    # 根路徑  #
            current_subdir = os.path.basename(self.folder) if getattr(self, "folder", None) else ''  # 現行子資料夾  #
            if not root or not os.path.isdir(root) or not getattr(self, "_file_combo", None):     # 健檢  #
                return                                                                            # 無法刷新直接離開  #

            items = os.listdir(root)                                                              # 重新列出子資料夾  #
            items_sorted = items[::-1]                                                            # 與原本一致（反向） #
            combo = self._file_combo                                                              # 取回參考  #
            combo.blockSignals(True)                                                              # 重建時先關閉訊號  #
            combo.clear()                                                                         # 清空  #
            combo.addItems(items_sorted)                                                          # 填入  #
            combo.blockSignals(False)                                                             # 重新開啟訊號  #

            if current_subdir in items_sorted:                                                    # 原選項仍存在  #
                idx = items_sorted.index(current_subdir)                                          # 找索引  #
                combo.setCurrentIndex(idx)                                                        # 指回原選項  #
            elif items_sorted:                                                                    # 原選項不在、但還有項目 #
                combo.setCurrentIndex(0)                                                          # 指第一個  #
                current_subdir = combo.currentText()                                              # 更新選擇  #
                self.folder = os.path.join(root, current_subdir)                                  # 更新路徑  #

            # 直接呼叫你既有的邏輯來重載影片/圖表  #
            self.File_combobox_TextChanged(combo, self._play_btn, self._icons, self._frameslider) # 重新載入  #
            self.showprevision()                                                                  # 立即重繪  #
        except Exception as e:
            print(f"[refresh] failed: {e}")        