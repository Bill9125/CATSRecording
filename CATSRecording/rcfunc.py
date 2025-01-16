from PyQt5 import QtCore, QtGui, QtWidgets
import os, glob, sys, time
import cv2, threading
from datetime import datetime
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
            
class Recordingbackend():
    def __init__(self):
        super(Recordingbackend, self).__init__()
        # Initialize YOLO model
        self.yolov8_model = YOLO("../model/yolo_bar_model/best.pt")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('yolo device:', device)
        self.yolov8_model.to(device)

        self.yolov8_model1 = YOLO("../model/yolov8_model/yolov8n-pose.pt")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('yolo device:', device)
        self.yolov8_model1.to(device)
        self.threads = []

        self.struct = {'Deadlift': 5, 'Benchpress': 3, 'Squat': 5}
        self.cameras = self.initialize_cameras()
        self.current_layout = None
        self.recording = False
        
        self.stop_event = threading.Event()

    def creat_threads(self, sport, Vision_labels):
        # Start YOLO and MediaPipe threads
        for i in range(self.struct[sport]):
            thread = threading.Thread(target=self.process_vision,
                                      args = (i, sport, Vision_labels[i]) , daemon=True)
            self.threads.append(thread)
            thread.start()

    def process_vision(self, i, sport, label):
        while True:
            ret, frame = self.cameras[i].getframe()
            if ret:
                cv2.putText(frame, f'FPS: {self.fps3:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                qpixmap = QtGui.QPixmap.fromImage(QtGui.QImage(frame_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888))
                scale_qpixmap = qpixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                label.setPixmap(scale_qpixmap)
                

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
            mbox.addButton(QtWidgets.QMessageBox.Ok)
            mbox.exec()
        elif type == 'Error':
            mbox.warning(Form, 'warning', f'{text}')
            mbox.addButton(QtWidgets.QMessageBox.Ok)
            mbox.exec()

    def manual_checkbox_isclicked(self, state):
        if state == 2:  
            self.isclicked = True
        else:  
            self.isclicked = False
        # return self.isclicked
        print(f"manual recording: {self.isclicked}") 
    
    def update_camera_layout(self, layout_type):
        if layout_type == "benchpress_layout":
            self.cameras = [MyVideoCapture(i) for i in range(3)]
        elif layout_type == "deadlift_layout":
            self.cameras = [MyVideoCapture(i) for i in range(5)]

        self.current_layout = layout_type
        print(f"Updated to {layout_type} with {len(self.cameras)} cameras.")

    def recording_ctrl_btn_clicked(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recording = True
        self.stop_event.clear()  # Clear the stop event before starting threads
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.folder = os.path.join(self.save_folder, f"recording_{timestamp}")
        os.makedirs(self.folder, exist_ok=True)
        file1 = os.path.join(self.folder, "vision1.avi")
        file2 = os.path.join(self.folder, "vision2.avi")
        file3 = os.path.join(self.folder, "vision3.avi")
        file4 = os.path.join(self.folder, "vision4.avi")
        file5 = os.path.join(self.folder, "vision5.avi")
        
        fps1 = 29
        fps2 = 29
        fps3 = 29
        fps4 = 29
        fps5 = 29
        # print(f'fps1: {fps1}, fps2: {fps2}')
        
        self.out1 = cv2.VideoWriter(file1, cv2.VideoWriter_fourcc(*'XVID'), fps1, (int(self.vid1.height), int(self.vid1.width)))
        self.out2 = cv2.VideoWriter(file2, cv2.VideoWriter_fourcc(*'XVID'), fps2, (int(self.vid2.height), int(self.vid2.width)))
        self.out3 = cv2.VideoWriter(file3, cv2.VideoWriter_fourcc(*'XVID'), fps3, (int(self.vid3.height), int(self.vid3.width)))
        self.out4 = cv2.VideoWriter(file4, cv2.VideoWriter_fourcc(*'XVID'), fps4, (int(self.vid4.height), int(self.vid4.width)))
        self.out5 = cv2.VideoWriter(file5, cv2.VideoWriter_fourcc(*'XVID'), fps4, (int(self.vid5.height), int(self.vid5.width)))
        if not self.out1.isOpened() or not self.out2.isOpened() or not self.out3.isOpened() or not self.out4.isOpened() or not self.out5.isOpened():
            self.messagebox("Error", "Failed to initialize video recording")
            self.recording = False
            if self.out1 is not None:
                self.out1.release()
                self.out1 = None
            if self.out2 is not None:
                self.out2.release()
                self.out2 = None
            if self.out3 is not None:
                self.out3.release()
                self.out3 = None
            if self.out4 is not None:
                self.out4.release()
                self.out4 = None
            if self.out5 is not None:
                self.out5.release()
                self.out5 = None    

        self.start_time1 = time.time()
        self.start_time2 = time.time()
        self.start_time3 = time.time()
        self.start_time4 = time.time()
        self.start_time5 = time.time()
        self.frame_count1 = 0
        self.frame_count2 = 0
        self.frame_count3 = 0
        self.frame_count4 = 0
        self.frame_count5 = 0
        self.frame_count_for_detect1 = 0
        
        # # Initialize text files for saving coordinates
        self.yolo_txt_path = os.path.join(self.folder, "yolo_coordinates.txt")
        self.mediapipe_txt_path = os.path.join(self.folder, "mediapipe_landmarks.txt")
        self.yolo_txt_file = open(self.yolo_txt_path, "w")
        self.mediapipe_txt_file = open(self.mediapipe_txt_path, "w")
        print("Recording started")
            
    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.stop_event.set()  # Signal threads to stop writing

            if self.out1 is not None:
                self.out1.release()
                self.out1 = None
            if self.out2 is not None:
                self.out2.release()
                self.out2 = None

            if self.out3 is not None:
                self.out3.release()
                self.out3 = None

            if self.out4 is not None:
                self.out4.release()
                self.out4 = None

            if self.out5 is not None:
                self.out5.release()
                self.out5 = None

            if self.yolo_txt_file is not None:
                self.yolo_txt_file.close()
                self.yolo_txt_file = None

            if self.mediapipe_txt_file is not None:
                self.mediapipe_txt_file.close()
                self.mediapipe_txt_file = None

            # Create an auto-closing message box
            self.messagebox('Info', "Recording stopped and saved.")

            
    def get_frame(self, camera_id):
        if 0 <= camera_id < len(self.cameras):
            ret, frame = self.cameras[camera_id].get_frame()
            if ret:
                # 如果是第三個相機，進行 180 度旋轉
                if camera_id == 2:  # 第三個相機，索引為 2
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                return frame
        return None
    
    
