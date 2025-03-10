from PyQt5 import QtCore, QtGui, QtWidgets
import os, glob, sys, time
import cv2, threading
from PyQt5.QtGui import QPainter, QPen
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import json
import numpy as np

class LineLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vertical_line_x = 0  # 初始垂直線位置
        self.horizontal_line_y = 0  # 初始水平線位置

    def set_vertical_line(self, value):
        """更新垂直線的位置 (X 軸) 並重新繪製"""
        self.horizontal_line_y = value
        self.update()  # 重新觸發 paintEvent()

    def set_horizontal_line(self, value):
        """更新水平線的位置 (Y 軸) 並重新繪製"""
        self.vertical_line_x = value
        self.update()  # 重新觸發 paintEvent()

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
        while not self._stop_event.is_set():
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

            if self.index == 0 and val % 7 == 0:
                self.canvas.draw()
            self.barrier.wait()

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
        self.data_path = {'Deadlift': ['Body_Length.json', 'Hip_Angle.json', 
                                       'Knee_Angle.json', 'Knee_to_Hip.json'],
                          'Benchpress' : ['Bar_Position.json', 'Armpit_Angle.json', 
                                          'Shoulder_Angle.json', 'Elbow_Angle.json']}
        self.folders = {}
        self.threads = []
        self.rp_Vision_labels = []
        self.rp_qpixmaps = []
        self.videos = []
        self.caps = []
        self.currentsport = ''
        self.ocv = True
        self.index = 0
        self.is_pause = False
        self.exited = False
        self.is_stop = True

    def Deadlift_btn_pressed(
        self, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn, icons,
        Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout,
        head_label, bottom_labels, graph
        ):
        self.currentsport = 'Deadlift'
        self.rp_Vision_labels = head_label + bottom_labels
        self.data_graph = graph
        self.rp_btn_press(
            self.currentsport, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn, icons,
            Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout,
            )
        
    def Benchpress_btn_pressed(
        self, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn, icons,
        Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout,
        head_label, bottom_labels, V_sliders, H_sliders, graph):
        self.currentsport = 'Benchpress'
        self.rp_Vision_labels = [head_label] + bottom_labels
        self.data_graph = graph
        self.V_sliders = V_sliders
        self.H_sliders = H_sliders
        self.rp_btn_press(
            self.currentsport, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn, icons,
            Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout,
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
        
    def rp_btn_press(
        self, sport, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn, icons, 
        Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout,
        ):
                
        if sport == 'Deadlift':
            folderPath = self.resource_path('C:/Users/92A27/MOCAP/recordings')
            self.folders[sport] = folderPath
            Deadlift_btn.setStyleSheet("font-size:18px;background-color: #888888")
            Benchpress_btn.setStyleSheet("font-size:18px;background-color: #666666")
            Squat_btn.setStyleSheet("font-size:18px;background-color: #666666")
        
        elif sport == 'Benchpress':
            folderPath = self.resource_path('C:/Users/92A27/benchpress/recordings')
            self.folders[sport] = folderPath
            Benchpress_btn.setStyleSheet("font-size:18px;background-color: #888888")
            Squat_btn.setStyleSheet("font-size:18px;background-color: #666666")
            Deadlift_btn.setStyleSheet("font-size:18px;background-color: #666666")

        elif sport == 'Squat':
            folderPath = self.resource_path(f'C:/Users/92A27/barbell_squart/recordings')
            self.folders[sport] = folderPath
            self.rp_Vision_labels, self.rp_qpixmaps = self.creat_vision_labels_pixmaps([384, 512], rp_tab, play_layout, 5)
            Squat_btn.setStyleSheet("font-size:18px;background-color: #888888")
            Benchpress_btn.setStyleSheet("font-size:18px;background-color: #666666")
            Deadlift_btn.setStyleSheet("font-size:18px;background-color: #666666")

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
        videofolder = file_comboBox.currentText()
        folder = self.folders[self.currentsport]
        videos = glob.glob(f'{folder}/{videofolder}/*.avi')
        self.datas = []
        
        # 臥推有六部avi影片，要抽取三部    
        if self.currentsport == 'Benchpress':
            if len(videos) == 6:
                self.videos = [video for video in videos 
                            if os.path.basename(video) in ('original_vision1.avi', 'vision2.avi', 'original_vision3.avi')
                            ]
                if self.videos:
                    self.videos[1], self.videos[2] = self.videos[2], self.videos[1]
            if len(videos) == 7:
                self.videos = [video for video in videos 
                            if os.path.basename(video) in ('vision1_drawed.avi', 'vision2.avi', 'original_vision3.avi')
                            ]
                if self.videos:
                    self.videos[1], self.videos[2] = self.videos[2], self.videos[1]
                for i in range(len(self.data_path[self.currentsport])):
                    with open(f'../config/{self.currentsport}_data/{self.data_path[self.currentsport][i]}',
                                mode='r', encoding='utf-8') as file:
                        if file:
                            data = json.load(file)
                            self.datas.append(data)
                
        ## 硬舉avi只需要 1, 2, 3 視角
        if self.currentsport == 'Deadlift':
            # 未後製
            if len(videos) == 5:
                self.videos = [video for video in videos
                            if os.path.basename(video) in ('vision1.avi', 'vision2.avi', 'vision3.avi')]
                if self.videos:
                    self.videos = [self.videos[1], self.videos[2], self.videos[0]]
                self.datas = []
            # 已後製
            elif len(videos) == 6:
                self.videos = [video for video in videos
                            if os.path.basename(video) in ('vision1_drawed.avi', 'vision2.avi', 'vision3.avi')]
                if self.videos:
                    self.videos = [self.videos[1], self.videos[2], self.videos[0]]
                # 抓取計算完的檔案
                for i in range(len(self.data_path[self.currentsport])):
                    with open(f'../config/{self.currentsport}_data/{self.data_path[self.currentsport][i]}',
                                mode='r', encoding='utf-8') as file:
                        data = json.load(file)
                        self.datas.append(data)
        
        for _ in range(len(self.videos)):
            pixmap = QtGui.QPixmap()
            self.rp_qpixmaps.append(pixmap)
        self.stop(Frameslider, play_btn, icons)
    
    def play_btn_clicked(self, fast_forward_combobox, Play_btn, icons, Frameslider):
        self.index += 1
        # play
        if self.index % 2 == 1: 
            fast_forward_combobox.setEnabled(False)
            Frameslider.setEnabled(True)
            Play_btn.setIcon(icons[0])
            f_num = []
            for video in self.videos:
                cap = cv2.VideoCapture(video)
                if cap.get(cv2.CAP_PROP_FRAME_COUNT) != 0:
                    f_num.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            framenumber = min(f_num)

            # stop 後播放
            if self.is_stop:
                print('threads created')
                self.is_stop = False
                if self.threads:
                    self.del_mythreads()
                if self.caps:
                    self.caps.clear()
                self.barrier_play = threading.Barrier(len(self.videos))
                self.barrier_data = threading.Barrier(len(self.datas))
                for i, video in enumerate(self.videos):
                    cap = cv2.VideoCapture(video)
                    self.caps.append(cap)
                    thread_play = MyThread(self.caps, i, Play_btn, icons, fast_forward_combobox,
                                            Frameslider, framenumber, self.rp_Vision_labels,
                                            self.rp_qpixmaps, self.barrier_play)
                    thread_play.start()
                    self.threads.append(thread_play)
                    
                if self.datas:
                    for i, data in enumerate(self.datas):
                        data_thread = Thread_data(i, self.data_graph, data, self.barrier_data, fast_forward_combobox, Frameslider, framenumber)
                        data_thread.start()
                        self.threads.append(data_thread)
                    
            # pause 後繼續播放
            else:
                print('resume')
                for thread in self.threads:
                    thread.resume()
        # pause
        elif self.index % 2 == 0:
            self.pause_event(fast_forward_combobox, Play_btn, icons)

    def pause_event(self, fast_forward_combobox, Play_btn, icons):
        fast_forward_combobox.setEnabled(True)
        Play_btn.setIcon(icons[1])
        for thread in self.threads:
            thread.pause()
        
    # threads 全部刪除，重新播放
    def del_mythreads(self):
        for thread in self.threads:
            thread._stop_event.set()
        self.threads.clear()
    
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
            for i, data in enumerate(self.datas):
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
        graphicview = QtWidgets.QGraphicsView(parentlayout)
        graphicscene = QtWidgets.QGraphicsScene(parentlayout)
        graphicscene.addWidget(canvas)
        sublayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, graphicview)
        sublayout.setFormAlignment(QtCore.Qt.AlignCenter)
        graphicview.setScene(graphicscene)
        return graphicview, graphicscene, canvas, axes
        

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
        comboBox.clear()
        
        # 過濾符合條件的項目
        filtered_items = [item for item in self.all_items if filter_text in item.lower()]
        
        # 重新加入篩選後的項目
        comboBox.addItems(filtered_items)
        
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
        

    def resource_path(self, relative_path):
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        return os.path.join(base_path, relative_path)

