from PyQt5 import QtWidgets
import os, time, sys, json
import cv2, threading
from datetime import datetime
from ultralytics import YOLO
import torch
import loop
from subUI import ButtonClickApp
import mediapipe as mp

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
        self.vision_src = {}
        self.struct = {'Deadlift': 4, 'Benchpress': 3, 'Squat': 5}
        dir = 'C:/Users/92A27'
        self.save_path = {'Deadlift': os.path.join(dir, 'MOCAP', 'recordings'),
                          'Benchpress': os.path.join(dir, 'benchpress', 'recordings'),
                          'Squat': os.path.join(dir, 'barbell_squat', 'recordings')}
        self.skeleton_connections = [
            (0, 1), (0, 2), (2, 4), (1, 3),  # Right arm
            (5, 7), (5, 6), (7, 9), (6, 8),  # Left arm
            (6, 12), (12, 14), (14, 16),  # Right leg
            (5, 11), (11, 13), (13, 15)   # Left leg
        ]
        # Initialize MediaPipe Pose 
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.pose2 = self.mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.current_layout = None
        self.player = None
        self.barrier = None
        self.recording_sig = False
        self.save_sig = False
        self.save_sig_1 = False
        self.save_sig_2 = False
        self.save_sig_3 = False
        self.break_sig = False
        self.src_changing = False
        self.folder = str
        self.yolo_txt_file = str
        self.mediapipe_txt_file = str
        self.mediapipe_txt_file2 = str
        
        self.stop_event = threading.Event()
        
    def source_ctrl_btn_clicked(self, sport, labels):
        n = self.struct[sport]  # 按鈕數量
        window = self.subui(n, sport)
        window.ok_clicked.connect(lambda: self.subUI_close(sport, labels))
        window.show()
        self.stop_event.set()
        
    def subUI_close(self, sport, labels):
        self.stop_event.clear()
        self.init_rc_backend(sport, labels)
        
    def init_rc_backend(self, sport, labels):
        self.source_get(sport)
        self.cameras = self.initialize_cameras()
        self.models = self.model_select(sport)
        self.creat_threads(sport, labels)
    
    def source_get(self, sport):
        self.vision_src = {}
        for i in range(self.struct[sport]):
            self.vision_src[f'Vision{i+1}'] = i

        with open('../config/click_order.json', mode='r', newline='', encoding='utf-8') as file:
            data = json.load(file)
            for i in range(self.struct[sport]):
                self.vision_src[f'Vision{i+1}'] = int(data[sport][i])
        
    def initialize_cameras(self):
        cameras = []
        i = 0
        for src in self.vision_src.values():
            try:
                print(f'cam {i} with {src}')
                i+=1
                cam = MyVideoCapture(src)
                if cam.isOpened():
                    cameras.append(cam)
                else:
                    print(f"Camera {src} is not available.")
            except Exception as e:
                print(f"Error opening camera {src}: {e}")

        if not cameras:
            print("No cameras connected.")
        return cameras
        
    def creat_threads(self, sport, labels):
        # Start YOLO and MediaPipe threads
        self.threads =[]
        if self.barrier:
            self.barrier.abort()
        self.stop_event.clear()
        self.barrier = threading.Barrier(self.struct[sport])
        for i in range(self.struct[sport]):
            thread = threading.Thread(target=self.process_vision,
                                      args = (i, sport, labels[i], self.barrier) , daemon=True)
            self.threads.append(thread)
            thread.start()
            
    def model_select(self, sport):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if sport == 'Deadlift':
            bar_model = YOLO("../model/deadlift/yolo_bar_model/best.pt")
            bone_model = YOLO("../model/deadlift/yolov8_model/yolov8n-pose.pt")
            bar_model.to(device)
            bone_model.to(device)
            return [bar_model, bone_model]
        elif sport =='Benchpress':
            bar_model = YOLO("../model/benchpress/yolo_bar_model/best.pt")
            body_model = YOLO("../model/benchpress/body_model/yolov8n-pose.pt")
            head_model = YOLO("../model/benchpress/head_model/best.pt")
            bar_model.to(device)
            body_model.to(device)
            head_model.to(device)
            return [bar_model, body_model, head_model]
        
    def process_vision(self, i, sport, label, barrier):
        start_time = time.time()
        frame_count = 0
        frame_count_for_detect = 0
        fps = 0
        out = None
        self.out_1 = None
        self.out_2 = None
        self.out_3 = None
        original_out = None
        # 基本錄製結構
        while not self.stop_event.is_set():
            cap = self.cameras[i]
            ret, frame = cap.get_frame()
            if ret:
                if sport == 'Deadlift':
                    if i == 0:
                        start_time, frame_count, fps, self.out_1, frame_count_for_detect, self.save_sig_1 = loop.deadlift_bar_loop(
                            i, frame, label, self.save_sig_1, self.recording_sig,
                            self.folder, start_time, frame_count, fps, self.out_1, self.models[i],
                            self.yolo_txt_file, frame_count_for_detect, barrier)
                    elif i == 1:
                        start_time, frame_count, fps, self.out_2, frame_count_for_detect, self.save_sig_2 = loop.deadlift_bone_loop(
                            i, frame, label, self.save_sig_2, self.recording_sig,
                            self.folder, start_time, frame_count, fps, self.out_2, self.models[i],
                            self.mediapipe_txt_file, frame_count_for_detect, self.skeleton_connections, barrier)
                    else:
                        start_time, frame_count, fps, out, self.save_sig_3 = loop.deadlift_general_loop(
                            i, frame, label, self.save_sig_3, self.recording_sig,
                            self.folder, start_time, frame_count, fps, out, barrier)
                
                elif sport == 'Benchpress':
                    if i == 0:
                        start_time, frame_count, fps, self.out_1, frame_count_for_detect, original_out, self.save_sig_1 = loop.benchpress_bar_loop(
                            i, frame, label, self.save_sig_1, self.recording_sig,
                            self.folder, start_time, frame_count, fps, self.out_1, original_out, self.models[i],
                            self.yolo_txt_file, frame_count_for_detect, barrier)
                    elif i == 1:
                        excluded_indices = set(range(0, 11)) | set(range(25, 33)) | set(range(15, 23)) 
                        start_time, frame_count, fps, self.out_2, frame_count_for_detect, original_out, self.save_sig_2 = loop.benchpress_body_loop(
                            i, frame, label, self.save_sig_2, self.recording_sig,
                            self.folder, start_time, frame_count, fps, self.out_2, original_out,
                            excluded_indices, self.mediapipe_txt_file, self.pose, frame_count_for_detect, barrier)
                    else:
                        excluded_indices = set(range(0, 11)) | set(range(25, 33))
                        start_time, frame_count, fps, self.out_3, frame_count_for_detect, original_out, self.save_sig_3 = loop.benchpress_head_loop(
                            i, frame, label, self.save_sig_3, self.recording_sig,
                            self.folder, start_time, frame_count, fps, self.out_3, original_out,
                            excluded_indices, self.mediapipe_txt_file2, self.pose2, frame_count_for_detect, barrier)
                
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

    def recording_ctrl_btn_clicked(self, sport):
        if not self.recording_sig:
            self.start_recording(sport)
        else:
            self.stop_recording()
            
    def player_reset(self, name):
        self.player = name.text()

    def start_recording(self, sport):
        self.stop_event.clear()  # Clear the stop event before starting threads
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        if self.player:
            self.folder = os.path.join(self.save_path[sport], self.player, f"recording_{timestamp}")
        else:
            self.folder = os.path.join(self.save_path[sport], f"recording_{timestamp}")
        os.makedirs(self.folder, exist_ok=True)
        self.out_1 = None  # 確保 `out` 變數重置
        self.out_2 = None
        self.out_3 = None
        
        # Initialize text files for saving coordinates
        yolo_txt_path = os.path.join(self.folder, "yolo_coordinates.txt")
        mediapipe_txt_path = os.path.join(self.folder, "mediapipe_landmarks.txt")
        mediapipe_txt_path2 = os.path.join(self.folder, "mediapipe_landmarks_2.txt")
        self.yolo_txt_file = open(yolo_txt_path, "w")
        self.mediapipe_txt_file = open(mediapipe_txt_path, "w")
        self.mediapipe_txt_file2 = open(mediapipe_txt_path2, "w")
        self.recording_sig = True
        print("Recording started")
            
    def stop_recording(self):
        if self.recording_sig:
            self.recording_sig = False
            self.save_sig_1 = True
            self.save_sig_2 = True
            self.save_sig_3 = True    
            if self.out_1 or self.out_2 or self.out_3:
                self.out_1.release()
                self.out_2.release()
                self.out_3.release()
            if self.yolo_txt_file is not None:
                self.yolo_txt_file.close()
                self.yolo_txt_file = None
            if self.mediapipe_txt_file is not None:
                self.mediapipe_txt_file.close()
                self.mediapipe_txt_file = None
        
    def data_produce_btn_clicked(self):
        # self.folder = 'C:/Users/92A27/MOCAP/recordings/cam_group_1_recording_3'
        os.system(f'python ./tools/interpolate.py {self.folder}')
        os.system(f'python ./tools/data_produce.py {self.folder} --out ../config')
        os.system(f'python ./tools/trajectory.py {self.folder}')
        print('後製已完成')