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
        dir = 'C:/Users/92A27'
        self.save_path = {'Deadlift': os.path.join(dir, 'MOCAP', 'recordings'),
                          'Benchpress': os.path.join(dir, 'benchpress', 'recordings'),
                          'Squat': os.path.join(dir, 'barbell_squat', 'recordings')}
        self.cameras = self.initialize_cameras()
        self.current_layout = None
        self.recording = False
        self.save_sig = False
        
        self.stop_event = threading.Event()

    def creat_threads(self, sport, Vision_labels):
        # Start YOLO and MediaPipe threads
        for i in range(self.struct[sport]):
            thread = threading.Thread(target=self.process_vision,
                                      args = (i, sport, Vision_labels[i]) , daemon=True)
            self.threads.append(thread)
            thread.start()

    def process_vision(self, i, sport, label):
        start_time = time.time()  
        frame_count = 0
        fps = 0
        cap = self.cameras[i]
        file = os.path.join(self.folder, f'vision{i + 1}.avi')
        out = cv2.VideoWriter(file, cv2.VideoWriter_fourcc(*'XVID'), 29, (int(cap.height), int(cap.width)))
        while True:
            ret, frame = cap.get_frame()
            if ret:
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                if self.recording:
                    out.write(frame)
                if self.save_sig:
                    out.release()
                    out = None
                    self.save_sig = False
                        
                    
                cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
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
        self.mbox = QtWidgets.QMessageBox(Form)
        if type == 'Info':
            self.mbox.information(Form, 'info', f'{text}')
            self.mbox.setStandardButtons(QtWidgets.QMessageBox.NoButton)
            self.mbox.show()
        elif type == 'Error':
            self.mbox.warning(Form, 'warning', f'{text}')
            self.mbox.addButton(QtWidgets.QMessageBox.Ok)
            self.mbox.show()

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

    def recording_ctrl_btn_clicked(self, sport):
        if not self.recording:
            self.start_recording(sport)
        else:
            self.stop_recording()

    def start_recording(self, sport):
        self.stop_event.clear()  # Clear the stop event before starting threads
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.folder = os.path.join(self.save_path[sport], f"recording_{timestamp}")
        os.makedirs(self.folder, exist_ok=True)
        self.recording = True
        
        # # Initialize text files for saving coordinates
        self.yolo_txt_path = os.path.join(self.folder, "yolo_coordinates.txt")
        self.mediapipe_txt_path = os.path.join(self.folder, "mediapipe_landmarks.txt")
        self.yolo_txt_file = open(self.yolo_txt_path, "w")
        self.mediapipe_txt_file = open(self.mediapipe_txt_path, "w")
        print("Recording started")
            
    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.save_sig = True
            # self.stop_event.set()  # Signal threads to stop writing
            
            # Create an auto-closing message box
            # self.messagebox('Info', "Recording stopped. Saving...")
            
    def get_frame(self, camera_id):
        if 0 <= camera_id < len(self.cameras):
            ret, frame = self.cameras[camera_id].get_frame()
            if ret:
                # 如果是第三個相機，進行 180 度旋轉
                if camera_id == 2:  # 第三個相機，索引為 2
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                return frame
        return None
    
    
