from PyQt5 import QtWidgets
import os, time, sys, csv
import cv2, threading
from datetime import datetime
from ultralytics import YOLO
import torch
import loop
from subUI import ButtonClickApp
from qt_material import apply_stylesheet

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
        self.subui = ButtonClickApp
        self.threads = []
        self.struct = {'Deadlift': 5, 'Benchpress': 3, 'Squat': 5}
        dir = 'C:/Users/92A27'
        self.save_path = {'Deadlift': os.path.join(dir, 'MOCAP', 'recordings'),
                          'Benchpress': os.path.join(dir, 'benchpress', 'recordings'),
                          'Squat': os.path.join(dir, 'barbell_squat', 'recordings')}
        self.cameras = self.initialize_cameras()
        self.skeleton_connections = [
            (0, 1), (0, 2), (2, 4), (1, 3),  # Right arm
            (5, 7), (5, 6), (7, 9), (6, 8),  # Left arm
            (6, 12), (12, 14), (14, 16),  # Right leg
            (5, 11), (11, 13), (13, 15)   # Left leg
        ]
        self.current_layout = None
        self.recording_sig = False
        self.save_sig = False
        self.src_changing = False
        self.folder = str
        self.yolo_txt_file = str
        self.mediapipe_txt_file = str
        
        self.stop_event = threading.Event()
        
    def source_ctrl_btn_clicked(self, sport, labels):
        n = self.struct[sport]  # 按鈕數量
        window = self.subui(n)
        window.ok_clicked.connect(lambda: self.subUI_close(sport, labels))
        window.show()
        self.stop_event.set()
        
    def subUI_close(self, sport, labels):
        self.stop_event.clear()
        self.source_get()
        self.init_rc_backend(sport, labels)
        
        
    def initialize_cameras(self):
        self.source_get()
        cameras = []
        for source in self.sources:
            try:
                cam = MyVideoCapture(source)
                if cam.isOpened():
                    cameras.append(cam)
                else:
                    print(f"Camera {source} is not available.")
            except Exception as e:
                print(f"Error opening camera {source}: {e}")

        if not cameras:
            print("No cameras connected.")
        return cameras

    def init_rc_backend(self, sport, labels):
        self.bar_model, self.bone_model = self.model_select(sport)
        self.creat_threads(sport, labels)
        
    def source_get(self):
        with open('../config/click_order.csv', mode='r', newline='', encoding='utf-8') as file:
            if file is None:
                self.sources = [0, 1, 2, 3, 4]
            else:
                reader = csv.reader(file)
                sources = [row for row in reader]  # 將每一行存入列表
                self.sources = [int(value) for row in sources for value in row]

        
    def creat_threads(self, sport, labels):
        print('Start catch frame.')
        # Start YOLO and MediaPipe threads
        for i in range(self.struct[sport]):
            thread = threading.Thread(target=self.process_vision,
                                      args = (i, sport, labels[i]) , daemon=True)
            self.threads.append(thread)
            thread.start()
            
    def model_select(self, sport):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if sport == 'Deadlift':
            bar_model = YOLO("../model/deadlift/yolo_bar_model/best.pt")
            bone_model = YOLO("../model/deadlift/yolov8_model/yolov8n-pose.pt")
        elif sport =='Benchpress':
            bar_model = YOLO("../model/benchpress/yolo_bar_model/best.pt")
            bone_model = YOLO("../model/benchpress/yolov8_model/yolov8n-pose.pt")
            
        bar_model.to(device)
        bone_model.to(device)
        return bar_model, bone_model
        
    def process_vision(self, i, sport, label):
        start_time = time.time()  
        frame_count = 0
        frame_count_for_detect = 0
        fps = 0
        out = None
        # 基本錄製結構
        while not self.stop_event.is_set():
            src = self.sources[i]
            cap = self.cameras[src]
            ret, frame = cap.get_frame()
            if ret:
                if sport == 'Deadlift':
                    if i == 0:
                        start_time, frame_count, fps, out, frame_count_for_detect = loop.deadlift_bar_loop(
                            i, frame, label, self.save_sig, self.recording_sig,
                            self.folder, start_time, frame_count, fps, out, self.bar_model,
                            self.yolo_txt_file, frame_count_for_detect)
                    elif i == 1:
                        start_time, frame_count, fps, out = loop.deadlift_bone_loop(
                            i, frame, label, self.save_sig, self.recording_sig,
                            self.folder, start_time, frame_count, fps, out, self.bone_model,
                            self.mediapipe_txt_file, frame_count_for_detect, self.skeleton_connections)
                    else:
                        start_time, frame_count, fps, out = loop.deadlift_general_loop(
                            i, frame, label, self.save_sig, self.recording_sig,
                            self.folder, start_time, frame_count, fps, out)
                
                elif sport == 'Benchpress':
                    if i == 0:
                        start_time, frame_count, fps, out = loop.benchpress_bar_loop(
                            i, frame, label, self.save_sig, self.recording_sig,
                            self.folder, start_time, frame_count, fps, out)
                    elif i == 1:
                        start_time, frame_count, fps, out = loop.benchpress_bone_loop_1(
                            i, frame, label, self.save_sig, self.recording_sig,
                            self.folder, start_time, frame_count, fps, out)
                    else:
                        start_time, frame_count, fps, out = loop.benchpress_bone_loop_2(
                            i, frame, label, self.save_sig, self.recording_sig,
                            self.folder, start_time, frame_count, fps, out)
                
                elif sport == 'squat':
                    if i == 0:
                        start_time, frame_count, fps, out = loop.deadlift_bar_loop(
                            i, frame, label, self.save_sig, self.recording_sig,
                            self.folder, start_time, frame_count, fps, out)
                    elif i == 1:
                        start_time, frame_count, fps, out = loop.deadlift_bone_loop(
                            i, frame, label, self.save_sig, self.recording_sig,
                            self.folder, start_time, frame_count, fps, out)
                    else:
                        start_time, frame_count, fps, out = loop.deadlift_general_loop(
                            i, frame, label, self.save_sig, self.recording_sig,
                            self.folder, start_time, frame_count, fps, out)
        
        if out is not None:
            out.release()
        if self.yolo_txt_file is not None:
            self.yolo_txt_file.close()
            self.yolo_txt_file = None
        if self.mediapipe_txt_file is not None:
            self.mediapipe_txt_file.close()
            self.mediapipe_txt_file = None
        cap.__del__()
        
    

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
    
    def update_camera_layout(self, layout_type):
        if layout_type == "benchpress_layout":
            self.cameras = [MyVideoCapture(i) for i in range(3)]
        elif layout_type == "deadlift_layout":
            self.cameras = [MyVideoCapture(i) for i in range(5)]

        self.current_layout = layout_type
        print(f"Updated to {layout_type} with {len(self.cameras)} cameras.")

    def recording_ctrl_btn_clicked(self, sport):
        if not self.recording_sig:
            self.start_recording(sport)
        else:
            self.stop_recording()

    def start_recording(self, sport):
        self.stop_event.clear()  # Clear the stop event before starting threads
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.folder = os.path.join(self.save_path[sport], f"recording_{timestamp}")
        os.makedirs(self.folder, exist_ok=True)
        self.recording_sig = True
        
        # Initialize text files for saving coordinates
        yolo_txt_path = os.path.join(self.folder, "yolo_coordinates.txt")
        mediapipe_txt_path = os.path.join(self.folder, "mediapipe_landmarks.txt")
        self.yolo_txt_file = open(yolo_txt_path, "w")
        self.mediapipe_txt_file = open(mediapipe_txt_path, "w")
        print("Recording started")
            
    def stop_recording(self):
        if self.recording_sig:
            self.recording_sig = False
            self.save_sig = True
            
    
