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
        self.is_pause = bool
        self.exited = False
        self.is_stop = bool
        self.is_slide = False
    
    class MyThread(threading.Thread):
        def __init__(self, target, args=()):
            threading.Thread.__init__(self, daemon=True)
            self._pause_event = threading.Event()
            self._pause_event.set()
            self._stop_event = threading.Event()
            self.target = target
            self.args = args
            self.ended = False

        def run(self):
            while not self._stop_event.is_set():
                self._pause_event.wait()
                self.target(*self.args)

        def pause(self):
            self._pause_event.clear()

        def is_pause(self):
            return not self._pause_event.is_set()

        def resume(self):
            self._pause_event.set()

        def stop(self):
            self._stop_event.set()
            self.resume() 
    
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
        folderPath = self.resource_path(f'C:/Users/wujinglun/CATSRecording/data/recording_{sport}')
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
    def File_combobox_TextChanged(self, frameslider, file_comboBox , play_btn,
                                   fast_forward_combobox, icons):
        self.is_stop = True
        videofolder = file_comboBox.currentText()
        self.index = 0
        play_btn.setIcon(icons[1])
        folder = self.folders[self.currentsport]
        self.videos = glob.glob(f'{folder}/{videofolder}/*.avi')
        self.stop(frameslider, play_btn, fast_forward_combobox, icons)
        
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
            self.is_pause = False
            self.is_stop = False
            Play_btn.setIcon(icons[0])
            f_num = []
            for video in self.videos:
                cap = cv2.VideoCapture(video)
                f_num.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            framenumber = min(f_num)

            # stop 後播放
            if not self.is_stop:
                self.is_stop = False
                # threads 全部刪除，重新播放
                for thread in self.threads:
                    thread.ended = True
                self.threads.clear()
                for i, video in enumerate(self.videos):
                    thread_play = self.MyThread(target = self._play,
                                                args = (video, i, fast_forward_combobox, Frameslider,
                                                        framenumber, self.rp_Vision_labels, self.rp_qpixmaps))
                    thread_play.start()
                    self.threads.append(thread_play)

            # pause 後繼續播放
            else:
                for thread in self.threads:
                    thread.resume()
        # pause
        elif self.index % 2 == 0:
            fast_forward_combobox.setEnabled(True)
            self.is_pause = True
            Play_btn.setIcon(icons[1])
            for thread in self.threads:
                thread.pause()
                
    def _play(self, video, index, fast_forward_combobox, Frameslider, framenumber, Vision_labels, qpixmaps):
        cap = cv2.VideoCapture(video)
        while True:
            speed_rate = fast_forward_combobox.currentText()
            Frameslider.setMaximum(int(framenumber))
            spf = (1 / 30) / float(speed_rate)
            time.sleep(spf)
            val = Frameslider.value()
            cap.set(cv2.CAP_PROP_POS_FRAMES, val)
            ret , frame = cap.read()

            # 迴圈暫停條件
            if self.threads[index].is_pause():
                continue
            
            # 迴圈終止條件
            if self.is_stop: # 按下暫停或切換資料夾，迴圈終止
                cap.release()
                print('Stop btn is pressed.')
                break

            if (self.exited or self.threads[index].ended): # 如果退出視窗或thread停止或換資料夾，清空label，迴圈終止
                for i, qpixmap in enumerate(qpixmaps):
                    qpixmap = QtGui.QPixmap()
                    Vision_labels[i].setPixmap(qpixmap)
                cap.release()
                print('You change the file or exit the window, threads break.')
                break
            
            if (ret and not self.is_pause and not self.is_slide):
                if index == 0: # slider 跟著影片的幀數做移動
                    val = Frameslider.value()
                    Frameslider.setValue(val + 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QtGui.QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QtGui.QImage.Format_RGB888)
                qpixmaps[index] = QtGui.QPixmap.fromImage(image)
                scaled_pixmap = qpixmaps[index].scaled(Vision_labels[index].size(), QtCore.Qt.KeepAspectRatio)
                Vision_labels[index].setPixmap(scaled_pixmap)
        
    def slider_released(self):
        self.is_slide = False

    def interupsliding(self):
        self.is_slide = True 

    def sliding(self, Frameslider, TimeCount_LineEdit):
        fps = 30
        val = Frameslider.value()
        sec = val / fps
        minute = "%02d" % int(sec / 60)
        second = "%02d" % int(sec % 60)
        TimeCount_LineEdit.setText(f'{minute}:{second}')

    def closeEvent(self, event):
        self.exited = True
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

        
    def stop(self, frameslider, play_btn, fast_forward_combobox, icons):
        frameslider.setSliderPosition(0)
        play_btn.setIcon(icons[1])
        fast_forward_combobox.setEnabled(True)
        self.index = 0
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
