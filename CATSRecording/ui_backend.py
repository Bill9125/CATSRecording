from PyQt5 import QtCore, QtGui, QtWidgets
import os, glob, sys, time
import cv2, threading
from ultralytics import YOLO
import torch

class MyVideoCapture:
    def __init__(self, video_source):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            print("Unable to open video source", video_source)

        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def isOpened(self):
        # 檢查視頻是否正確打開
        return self.vid.isOpened()

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, frame)
            else:
                return (ret, None)
        else:
            return (False, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            
class MyThread(threading.Thread):
    def __init__(self, video, index, Play_btn, icons, fast_forward_combobox,
                    Frameslider, framenumber, Vision_labels, qpixmaps, barrier, is_pause, is_stop):
        threading.Thread.__init__(self, daemon=True)
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()
        self._stop_event.clear()
        self.video = video
        self.index = index
        self.Play_btn = Play_btn
        self.icons = icons
        self.fast_forward_combobox = fast_forward_combobox
        self.Frameslider = Frameslider
        self.framenumber = framenumber
        self.Vision_labels = Vision_labels
        self.qpixmaps = qpixmaps
        self.barrier = barrier
        self.is_pause = False
        self.is_slide = False
        self.is_stop = is_stop
        self.ended = False
        
    def run(self):
        cap = cv2.VideoCapture(self.video)
        self.Frameslider.setMaximum(int(self.framenumber))
        start_time = time.time()

        while not self._stop_event.is_set():
            speed_rate = self.fast_forward_combobox.currentText()
            val = self.Frameslider.value()
            spf = 1 / 30

            # 迴圈暫停條件
            if self.is_pause:
                print('pause')
                self._pause_event.wait()
                cap.set(cv2.CAP_PROP_POS_FRAMES, val)

            # 迴圈終止條件
            if self.Frameslider.value() >= self.framenumber or self.is_stop or self.ended:
                print('break')
                self.Frameslider.setSliderPosition(0)
                self.Play_btn.setIcon(self.icons[1])
                self.fast_forward_combobox.setEnabled(True)
                break

            # 滑塊拖動處理
            if self.is_slide and self.index == 0:
                backend.pause_event(self.fast_forward_combobox, self.Play_btn, self.icons)
                self.is_slide = False
                continue

            # 等待所有线程完成同步
            self.barrier.wait()

            if (time.time() - start_time) >= (spf / float(speed_rate)):
                # 讓第 0 個 threading 處理 slider
                if self.index == 0:
                    self.Frameslider.setValue(val + 1)
                _ , frame = cap.read()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QtGui.QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QtGui.QImage.Format_RGB888)
                self.qpixmaps[self.index] = QtGui.QPixmap.fromImage(image)
                scaled_pixmap = self.qpixmaps[self.index].scaled(self.Vision_labels[self.index].size(), QtCore.Qt.KeepAspectRatio)
                self.Vision_labels[self.index].setPixmap(scaled_pixmap)
                # 更新時間
                start_time = time.time()

        for i, qpixmap in enumerate(self.qpixmaps):
            qpixmap = QtGui.QPixmap()
            self.Vision_labels[i].setPixmap(qpixmap)

        self.index = 0
        self.Play_btn.setIcon(self.icons[1])
        cap.release()
        self.is_stop = True

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def stop(self):
        self._stop_event.set()
            
class backend():
    def __init__(self):
        super(backend, self).__init__()
        self.vid1 = MyVideoCapture(2)
        self.vid2 = MyVideoCapture(4)
        self.vid3 = MyVideoCapture(0)
        self.vid4 = MyVideoCapture(1)
        self.vid5 = MyVideoCapture(3)

        # Initialize YOLO model
        self.yolov8_model = YOLO("../model/yolo_bar_model/best.pt")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('yolo device:', device)
        self.yolov8_model.to(device)

        self.yolov8_model1 = YOLO("../model/yolov8_model/yolov8n-pose.pt")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('yolo device:', device)
        self.yolov8_model1.to(device)

        # init for replay
        self.firstclicked_D = True
        self.firstclicked_B = True
        self.firstclicked_S = True
        self.folders = {}
        self.threads = []
        self.rp_Vision_labels = []
        self.rp_qpixmaps = []
        self.currentsport = ''
        self.ocv = True
        self.index = 0
        self.is_pause = False
        self.exited = False
        self.is_stop = False
        self.is_slide = False
    
    

    
    def messagebox(self, type, text):
        Form = QtWidgets.QWidget()
        Form.setWindowTitle('message')
        Form.resize(300, 300)
        mbox = QtWidgets.QMessageBox(Form)
        if type == 'Info':
            mbox.information(Form, 'info', f'{text}')
        elif type == 'Error':
            mbox.warning(Form, 'warning', f'{text}')

    def recording_ctrl(self, Vision_labels):
        print(Vision_labels[0].size())
        # if self.vid1.isOpened():
        #     ret1, frame1 = self.vid1.get_frame()
        #     if ret1:
        #         # Detect the barbell position (YOLO model output)
        #         results = self.yolov8_model.predict(source=frame1, imgsz=320, conf=0.5)
        #         boxes = results[0].boxes
        #         if len(boxes.xywh) > 0:
        #             self.initial_position = boxes.xywh[0]  # Capture the first detected box as the initial position
        #             self.messagebox("Info", "Initial position captured.")
        #             self.recording = False  # Ensure recording is off initially
        #             self.auto_recording = True  # Enable automatic recording trigger
        #             self.threshold = 50  # Set a threshold for starting and stopping recording (can be adjusted)
        #         else:
        #             self.messagebox("Error", "No detection found. Try again.")

    def Deadlift_btn_pressed(self, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn,
                            Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout):
        self.currentsport = 'Deadlift'
        self.clear_layout(play_layout)
        self.rp_btn_press(self.currentsport,
                            Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn,
                            Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout)
       

    def Benchpress_btn_pressed(self, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn,
                            Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout):
        self.currentsport = 'Benchpress'
        self.clear_layout(play_layout)
        self.rp_btn_press(self.currentsport, 
                            Deadlift_btn,Benchpress_btn, Squat_btn, Play_btn,
                            Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout)
        
        
    def Squat_btn_pressed(self, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn,
                            Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout):
        self.currentsport = 'Squat'
        self.clear_layout(play_layout)
        self.rp_btn_press(self.currentsport, 
                            Deadlift_btn,Benchpress_btn, Squat_btn, Play_btn,
                            Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout)
        
        
        
    def rp_btn_press(self, sport, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn,
                              Stop_btn, Frameslider, fast_forward_combobox, File_comboBox, rp_tab, play_layout):
        # Clear all the Qpixmap
        self.clear_layout(play_layout)
        if sport == 'Deadlift':
            self.rp_Vision_labels, self.rp_qpixmaps = self.creat_vision_labels_pixmaps([420, 560], rp_tab, play_layout, 5)
            Deadlift_btn.setStyleSheet("font-size:18px;background-color: #888888")
            Benchpress_btn.setStyleSheet("font-size:18px;background-color: #666666")
            Squat_btn.setStyleSheet("font-size:18px;background-color: #666666")
        
        elif sport == 'Benchpress':
            self.rp_Vision_labels, self.rp_qpixmaps = self.creat_vision_labels_pixmaps([640, 480], rp_tab, play_layout, 3)
            Benchpress_btn.setStyleSheet("font-size:18px;background-color: #888888")
            Squat_btn.setStyleSheet("font-size:18px;background-color: #666666")
            Deadlift_btn.setStyleSheet("font-size:18px;background-color: #666666")

        elif sport == 'Squat':
            self.rp_Vision_labels, self.rp_qpixmaps = self.creat_vision_labels_pixmaps([420, 560], rp_tab, play_layout, 5)
            Squat_btn.setStyleSheet("font-size:18px;background-color: #888888")
            Benchpress_btn.setStyleSheet("font-size:18px;background-color: #666666")
            Deadlift_btn.setStyleSheet("font-size:18px;background-color: #666666")
            
        
        for thread in self.threads:
            thread.ended = True
            
        folderPath = self.resource_path(f'C:/jinglun/CATSRecording/data/recording_{sport}')
        self.folders[sport] = folderPath

        File_comboBox.clear()
        list = os.listdir(self.folders[sport])
        for folder in list[::-1]:
            File_comboBox.addItems([folder])

        Play_btn.setEnabled(True)
        Stop_btn.setEnabled(True)
        Frameslider.setEnabled(True)
        fast_forward_combobox.setEnabled(True)


    # 讀取combobox內的資料夾
    def File_combobox_TextChanged(self, file_comboBox, play_btn, icons):
        self.stop()
        videofolder = file_comboBox.currentText()
        self.index = 0
        play_btn.setIcon(icons[1])
        folder = self.folders[self.currentsport]
        self.videos = glob.glob(f'{folder}/{videofolder}/*.avi')
        
        
        # 臥推有六部影片，要抽取三部
        if len(self.videos) >= 6:
            self.videos = [video for video in self.videos 
                           if os.path.basename(video) in ('original_vision1.avi', 'vision2.avi', 'original_vision3.avi')
                        ]
            self.videos[1], self.videos[2] = self.videos[2], self.videos[1]
        # show the first frame of videos
        for i in range(len(self.videos)):
            thread = threading.Thread(target=self.showprevision, args=(self.videos[i], self.rp_qpixmaps[i], self.rp_Vision_labels[i]))
            self.threads.append(thread)
            thread.start()
        for thread in self.threads:
            thread.join()
        self.threads.clear()
    
    def play_btn_clicked(self, fast_forward_combobox, Play_btn, icons, Frameslider):
        self.index += 1
        # play
        if self.index % 2 == 1: 
            fast_forward_combobox.setEnabled(False)
            Play_btn.setIcon(icons[0])
            f_num = []
            for video in self.videos:
                cap = cv2.VideoCapture(video)
                f_num.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            framenumber = min(f_num)

            # stop 後播放
            if self.is_stop:
                self.is_stop = False
                if self.threads:
                    self.del_mythreads()
                self.barrier = threading.Barrier(len(self.videos))
                for i, video in enumerate(self.videos):
                    thread_play = MyThread(video, i, Play_btn, icons, fast_forward_combobox,
                                            Frameslider, framenumber, self.rp_Vision_labels, self.rp_qpixmaps,
                                            self.barrier, self.is_pause, self.is_stop)
                    thread_play.start()
                    self.threads.append(thread_play)

            # pause 後繼續播放
            else:
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
            thread.ended = True
        self.threads.clear()
            
    def slider_released(self):
        for thread in self.threads:
            thread.is_slide = False 

    def slider_Pressed(self):
        for thread in self.threads:
            thread.is_slide = True 

    def sliding(self, Frameslider, TimeCount_LineEdit):
        fps = 30
        val = Frameslider.value()
        sec = val / fps
        minute = "%02d" % int(sec / 60)
        second = "%02d" % int(sec % 60)
        TimeCount_LineEdit.setText(f'{minute}:{second}')

    def closeEvent(self, event):
        self.del_mythreads()
        event.accept()
        

    def showprevision(self, video, qpixmap, label):
        cap = cv2.VideoCapture(video)
        if self.ocv:
            _ , frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QtGui.QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QtGui.QImage.Format_RGB888)
            qpixmap = QtGui.QPixmap.fromImage(image)
            scaled_pixmap = qpixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio)
            label.setPixmap(scaled_pixmap)

    def creat_vision_labels_pixmaps(self, labelsize, parentlayout, sublayout, num):
        Vision_labels = []
        qpixmaps = []
        for i in range(num):
            qpixmap = QtGui.QPixmap()
            qpixmaps.append(qpixmap)
            Vision_label = QtWidgets.QLabel(parentlayout)
            Vision_label.setFrameShape(QtWidgets.QFrame.Panel)
            Vision_label.setMinimumSize(labelsize[0], labelsize[1])
            Vision_label.setMaximumSize(labelsize[0], labelsize[1])
            Vision_label.setPixmap(qpixmap)
            sublayout.addWidget(Vision_label)
            Vision_labels.append(Vision_label)
        return Vision_labels, qpixmaps

        
    def stop(self):
        self.is_stop = True
        
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
