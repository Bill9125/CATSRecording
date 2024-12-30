from PyQt5 import QtCore, QtGui, QtWidgets
import os, glob, sys
import cv2
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
                            Stop_btn, Frameslider, fast_forward_combobox, File_comboBox):
        currentsport = 'Deadlift'
        self.rp_btn_press(currentsport, self.firstclicked_D, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn,
                              Stop_btn, Frameslider, fast_forward_combobox, File_comboBox)
        
    def Benchpress_btn_pressed(self, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn,
                            Stop_btn, Frameslider, fast_forward_combobox, File_comboBox):
        currentsport = 'Benchpress'
        self.rp_btn_press(currentsport, self.firstclicked_B, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn,
                              Stop_btn, Frameslider, fast_forward_combobox, File_comboBox)
        
    def Squat_btn_pressed(self, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn,
                            Stop_btn, Frameslider, fast_forward_combobox, File_comboBox):
        currentsport = 'Squat'
        self.rp_btn_press(currentsport, self.firstclicked_S, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn,
                              Stop_btn, Frameslider, fast_forward_combobox, File_comboBox)
        

    def rp_btn_press(self, sport, firstclicked, Deadlift_btn, Benchpress_btn, Squat_btn, Play_btn,
                              Stop_btn, Frameslider, fast_forward_combobox, File_comboBox):
        if sport == 'Deadlift':
            Deadlift_btn.setStyleSheet("font-size:18px;background-color: #888888")
            Benchpress_btn.setStyleSheet("font-size:18px;background-color: #666666")
            Squat_btn.setStyleSheet("font-size:18px;background-color: #666666")
        
        elif sport == 'Benchpress':
            Benchpress_btn.setStyleSheet("font-size:18px;background-color: #888888")
            Squat_btn.setStyleSheet("font-size:18px;background-color: #666666")
            Deadlift_btn.setStyleSheet("font-size:18px;background-color: #666666")

        elif sport == 'Squat':
            Squat_btn.setStyleSheet("font-size:18px;background-color: #888888")
            Benchpress_btn.setStyleSheet("font-size:18px;background-color: #666666")
            Deadlift_btn.setStyleSheet("font-size:18px;background-color: #666666")
            
        if firstclicked == True:
            for thread in self.threads:
                thread.ended = True
            folderPath = self.resource_path(f'C:/jinglun/recording_{sport}')
            self.folders[sport] = folderPath
            firstclicked = False

        File_comboBox.clear()
        list = os.listdir(self.folders[sport])
        for folder in list[::-1]:
            File_comboBox.addItems([folder])

        Play_btn.setEnabled(True)
        Stop_btn.setEnabled(True)
        Frameslider.setEnabled(True)
        fast_forward_combobox.setEnabled(True)
        
    def resource_path(self, relative_path):
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        return os.path.join(base_path, relative_path)
