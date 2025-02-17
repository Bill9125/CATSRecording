from PyQt5 import QtCore, QtGui, QtWidgets
import os, glob, sys, time
import cv2, threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import json


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
            if self.is_pause:
                continue
                
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
    def __init__(self, gragh, file, barrier, fast_forward_combobox, Frameslider, framenumber):
        threading.Thread.__init__(self, daemon=True)
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()
        self._stop_event.clear()
        self.gragh = gragh
        self.file = file
        self.fast_forward_combobox = fast_forward_combobox
        self.Frameslider = Frameslider
        self.framenumber = framenumber
        self.barrier = barrier
        self.is_pause = False
        self.is_slide_end = False
        self.is_slide_start = False
        
        self.line, = self.gragh['ax'].plot([], [], color="red")
        
    def run(self):
        start_time = time.time()
        x_data = self.file['frames']
        y_data = self.file['values']
        self.gragh['ax'].set_xlim(min(x_data), max(x_data))
        min_length = min(len(x_data), len(y_data))
        x_data = x_data[:min_length]
        y_data = y_data[:min_length]
        if min_length > 200:
            trimmed_data = y_data[100:-100]  # 只取中间部分数据
        else:
            trimmed_data = y_data  # 如果数据少于 200 帧，保留所有数据
        y_min = min(trimmed_data) * 0.9
        y_max = max(trimmed_data) * 1.1
        self.gragh['ax'].set_ylim(y_min, y_max)
        self.gragh['ax'].set_xlabel('frames')
        self.gragh['ax'].set_ylabel(f"{self.file['y_label']}")
        self.gragh['ax'].legend()
        while not self._stop_event.is_set():
            speed_rate = self.fast_forward_combobox.currentText()
            spf = 1 / 30
            # 迴圈暫停條件
            if self.is_pause:
                continue
            
            if self.is_slide_start:
                if self.is_slide_end:
                    self.line.set_data([], [])  # **清除舊數據**
                    self.is_slide_end = False
                    self.is_slide_start = False
                continue
            
            # 迴圈終止條件
            if self.Frameslider.value() >= self.framenumber:
                break
                
            # 等待所有 thread 完成同步
            self.barrier.wait()
            val = self.Frameslider.value()
            if (time.time() - start_time) >= (spf / float(speed_rate)):
                self.line.set_data(x_data[:val], y_data[:val])
                if val % 5 == 0:
                    self.gragh['canvas'].draw()
                
    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def stop(self):
        self._stop_event.set()
            
class Replaybackend():
    def __init__(self):
        super(Replaybackend, self).__init__()
        # init for replay
        self.firstclicked_D = True
        self.firstclicked_B = True
        self.firstclicked_S = True
        self.data_path = {'Deadlift': ['Body_Length.json', 'Hip_Angle.json', 
                                       'Knee_Angle.json', 'Knee_to_Hip.json'],
                          'Benchpress' : ['Armpit_Angles.json', 'Bar_Position.json', 
                                          'Shoulder_Angle.json']}
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
        head_label, bottom_labels, graghs
        ):
        self.currentsport = 'Deadlift'
        self.rp_Vision_labels = head_label + bottom_labels
        self.data_graghs = graghs
        self.rp_btn_press(
            self.currentsport, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn, icons,
            Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout,
            )
        
    def Benchpress_btn_pressed(
        self, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn, icons,
        Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout,
        head_label, bottom_labels, data_labels
        ):
        self.currentsport = 'Benchpress'
        self.rp_Vision_labels = head_label + bottom_labels
        self.data_labels = data_labels
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
        list = os.listdir(self.folders[sport])
        # 這裡combobox有變動
        for folder in list[::-1]:
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
                self.videos[1], self.videos[2] = self.videos[2], self.videos[1]

        ## 硬舉avi只需要 1, 2, 3 視角
        if self.currentsport == 'Deadlift':
            # 未後製
            if len(videos) == 5:
                self.videos = [video for video in videos
                            if os.path.basename(video) in ('vision1.avi', 'vision2.avi', 'vision3.avi')]
                self.videos = [self.videos[1], self.videos[2], self.videos[0]]
                self.datas = []
            # 已後製
            elif len(videos) == 6:
                self.videos = [video for video in videos
                            if os.path.basename(video) in ('vision1_drawed.avi', 'vision2.avi', 'vision3.avi')]
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
                self.barrier_data = threading.Barrier(len(self.data_graghs))
                for i, video in enumerate(self.videos):
                    cap = cv2.VideoCapture(video)
                    self.caps.append(cap)
                    thread_play = MyThread(self.caps, i, Play_btn, icons, fast_forward_combobox,
                                            Frameslider, framenumber, self.rp_Vision_labels,
                                            self.rp_qpixmaps, self.barrier_play)
                    thread_play.start()
                    self.threads.append(thread_play)
                    
                if self.datas:
                    for i, gragh in enumerate(self.data_graghs):
                        data_thread = Thread_data(gragh, self.datas[i], self.barrier_data, fast_forward_combobox, Frameslider, framenumber)
                        data_thread.start()
                        self.threads.append(data_thread)
                    
            # pause 後繼續播放
            else:
                print('resume')
                for thread in self.threads:
                    thread.is_pause = False
                    thread.resume()
        # pause
        elif self.index % 2 == 0:
            self.pause_event(fast_forward_combobox, Play_btn, icons)

    def pause_event(self, fast_forward_combobox, Play_btn, icons):
        fast_forward_combobox.setEnabled(True)
        Play_btn.setIcon(icons[1])
        for thread in self.threads:
            thread.is_pause = True
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
            for i, gragh in enumerate(self.data_graghs):
                file = self.datas[i]
                x_data = file['frames']
                y_data = file['values']
                gragh['ax'].clear()
                min_length = min(len(x_data), len(y_data))
                x_data = x_data[:min_length]
                y_data = y_data[:min_length]
                if min_length > 200:
                    trimmed_data = y_data[100:-100]  # 只取中间部分数据
                else:
                    trimmed_data = y_data  # 如果数据少于 200 帧，保留所有数据
                y_min = min(trimmed_data) * 0.9
                y_max = max(trimmed_data) * 1.1
                gragh['ax'].set_ylim(y_min, y_max)
                gragh['ax'].plot(x_data, y_data, label = f"{file['title']}")
                gragh['ax'].set_xlabel('frames')
                gragh['ax'].set_ylabel(f"{file['y_label']}")
                gragh['ax'].legend()
                gragh['canvas'].draw()
                gragh['graphicscene'].addWidget(gragh['canvas'])
        else:
            for i, gragh in enumerate(self.data_graghs):
                gragh['ax'].clear()
                gragh['canvas'].draw()
                

    def creat_vision_labels_pixmaps(self, labelsize, parentlayout, sublayout, num):
        Vision_labels = []
        qpixmaps = []
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
    
    def creat_matplot_labels(self, size, parentlayout, sublayout):
        figure = Figure(figsize=(17,2.8))
        canvas = FigureCanvas(figure)
        ax = figure.add_subplot(111)
        graphicview = QtWidgets.QGraphicsView(parentlayout)
        graphicscene = QtWidgets.QGraphicsScene(parentlayout)
        graphicscene.addWidget(canvas)
        sublayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, graphicview)
        sublayout.setFormAlignment(QtCore.Qt.AlignCenter)
        graphicview.setScene(graphicscene)
        return graphicview, graphicscene, canvas, ax
        

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

