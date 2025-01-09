from ui import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
import os, glob
import cv2
from ultralytics import YOLO
import torch

class MyVideoCapture:
    def __init__(self, video_source):
        try:
            self.vid = cv2.VideoCapture(video_source)
            if not self.vid.isOpened():
                raise ValueError(f"Unable to open video source {video_source}")
            self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except ValueError as e:
            print(f"Error: {e}")
            self.vid = None  # 無法打開相機時設為 None
            self.width = 0
            self.height = 0

    def isOpened(self):
        # 檢查視頻是否正確打開
        return self.vid is not None and self.vid.isOpened()
    
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
        # self.vid1 = MyVideoCapture(2)
        # self.vid2 = MyVideoCapture(4)
        # self.vid3 = MyVideoCapture(0)
        # self.vid4 = MyVideoCapture(1)
        # self.vid5 = MyVideoCapture(3)
        # Initialize YOLO model
        self.yolov8_model = YOLO("../model/yolo_bar_model/best.pt")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('yolo device:', device)
        self.yolov8_model.to(device)

        self.yolov8_model1 = YOLO("../model/yolov8_model/yolov8n-pose.pt")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('yolo device:', device)
        self.yolov8_model1.to(device)
        
        self.cameras = self.initialize_cameras()
        
        self.isclicked = False
        self.recording = False
        
        
    def initialize_cameras(self):
        cameras = []
        for i in range(5):
            try:
                cam = MyVideoCapture(i)
                if cam.isOpened():
                    cameras.append(cam)
                else:
                    print(f"Camera {i} is not available.")
            except Exception as e:
                print(f"Error opening camera {i}: {e}")
        
        if not cameras:
            print("No cameras connected.")
        return cameras

    def messagebox(self, type, text):
        Form = QtWidgets.QWidget()
        Form.setWindowTitle('message')
        Form.resize(300, 300)
        mbox = QtWidgets.QMessageBox(Form)
        if type == 'Info':
            mbox.information(Form, 'info', f'{text}')
        elif type == 'Error':
            mbox.warning(Form, 'warning', f'{text}')

    def manual_checkbox_isclicked(self, state):
        if state == 2:  
            self.isclicked = True
        else:  
            self.isclicked = False
        # return self.isclicked
        print(f"manual recording: {self.isclicked}") 
        
        '''get camera id'''
    def get_frame(self, camera_id):
        if 0 <= camera_id < len(self.cameras):
            ret, frame = self.cameras[camera_id].get_frame()
            if ret:
                return frame
        return None
    
    def update_camera_layout(self, layout_type):
        if layout_type == "benchpress_layout":
            self.cameras = [MyVideoCapture(i) for i in range(3)]
        elif layout_type == "deadlift_layout":
            self.cameras = [MyVideoCapture(i) for i in range(5)]

        print(f"Updated to {layout_type} with {len(self.cameras)} cameras.")
        
    def recording_ctrl_btn_clicked(self,checkbox):
        pass
        
    # def recording_ctrl(self, Vision_labels):
        
        
        
        
        
        
        # print(Vision_labels[0].size())
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
