
from ultralytics import YOLO
import os
import cv2
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime
import time
import mediapipe as mp
import torch
import threading
import pyautogui
# Check CUDA availability
print("Is CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0))
print(f"CUDA available: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")

class WebcamRecorder:
    def __init__(self, window, window_title, save_folder='.', video_source1=0, video_source2=1, video_source3=2,video_source4=3,video_source5=4):
        self.window = window
        self.window.title(window_title)
     
        self.save_folder = save_folder
        self.video_source1 = video_source1
        self.video_source2 = video_source5
        self.video_source3 = video_source3
        self.video_source4 = video_source2
        self.video_source5 = video_source4

        # Initialize YOLO model
        self.yolov8_model = YOLO("C:\\Users\\92A27\\benchpress\\yolo_bar_model\\best.pt")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('yolo device:', device)
        self.yolov8_model.to(device)
        
        self.yolov8_model1 = YOLO("C:\\Users\\92A27\\MOCAP\\yolov8n-pose.pt")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('yolo device:', device)
        self.yolov8_model1.to(device)

        self.model2 = YOLO("yolov8n-pose.pt")
        self.classNames2 = ["person"]



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

        # Custom Pose Connections: Excluding facial landmarks
        # self.POSE_CONNECTIONS_CUSTOM = [
        #     (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Upper body joints
        #     (11, 23), (12, 24), (23, 24),  # Torso connections
        #     (23, 25), (25, 27), (24, 26), (26, 28)  # Lower body joints
        # ]

        self.POSE_CONNECTIONS_CUSTOM = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Upper body joints
            (11, 23), (12, 24), (23, 24),  # Torso connections
        ]


        self.tab_control = ttk.Notebook(window)
        
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.tab1, text='Record')
        self.tab_control.add(self.tab2, text='Open File')
        self.tab_control.pack(expand=1, fill='both')
        
        self.vid1 = MyVideoCapture(self.video_source3)
        self.vid2 = MyVideoCapture(self.video_source2)
        self.vid3 = MyVideoCapture(self.video_source1)
        self.vid4 = MyVideoCapture(self.video_source4)
        self.vid5 = MyVideoCapture(self.video_source5)

        self.canvas_width = 285
        self.canvas_height = 380
        #####tab1 三台相機

        self.label1 = tk.Label(self.tab1, text="Vision 1")
        self.label1.grid(row=0, column=0, padx=5, pady=1)
        self.canvas1 = tk.Canvas(self.tab1, width=self.canvas_width, height=self.canvas_height)
        self.canvas1.grid(row=1, column=0, padx=5, pady=5)

        self.label2 = tk.Label(self.tab1, text="Vision 2")
        self.label2.grid(row=0, column=1, padx=5, pady=1)
        self.canvas2 = tk.Canvas(self.tab1, width=self.canvas_width, height=self.canvas_height)
        self.canvas2.grid(row=1, column=1, padx=5, pady=5)

        self.label3 = tk.Label(self.tab1, text="Vision 3")
        self.label3.grid(row=0, column=2, padx=5, pady=1)
        self.canvas3 = tk.Canvas(self.tab1, width=self.canvas_width, height=self.canvas_height)
        self.canvas3.grid(row=1, column=2, padx=5, pady=5)

        self.label4 = tk.Label(self.tab1, text="Vision 4")
        self.label4.grid(row=0, column=3, padx=5, pady=1)
        self.canvas4 = tk.Canvas(self.tab1, width=self.canvas_width, height=self.canvas_height)
        self.canvas4.grid(row=1, column=3, padx=5, pady=5)

        self.label5 = tk.Label(self.tab1, text="Vision 5")
        self.label5.grid(row=0, column=4, padx=5, pady=1)
        self.canvas5 = tk.Canvas(self.tab1, width=self.canvas_width, height=self.canvas_height)
        self.canvas5.grid(row=1, column=4, padx=5, pady=5)


        self.frame_buttons = tk.Frame(self.tab1)
        self.frame_buttons.grid(row=2, column=0, columnspan=2, pady=1)

        # Combine Start and Stop into a single button
        # Start recording button
        self.btn_record = tk.Button(self.frame_buttons, text="Start", width=50, command=self.toggle_recording)
        self.btn_record.pack(side=tk.BOTTOM, padx=10, pady=10)

        # Capture Initial Position button (for automatic start/stop)
        self.capture_button = tk.Button(self.frame_buttons, text="Capture Initial Position", width=50, command=self.capture_initial_position)
        self.capture_button.pack(side=tk.BOTTOM, padx=10, pady=10)

        # Add a button to restore manual recording control
        self.restore_button = tk.Button(self.frame_buttons, text="Restore Manual Control", width=50, command=self.restore_manual_control)
        self.restore_button.pack(side=tk.BOTTOM, padx=10, pady=10)


        self.auto_recording = False  # 初始化 auto_recording 屬性
        self.initial_position = None  # 初始化 initial_position，作為初始座標
        self.threshold = 50  # 初始化 threshold，用於判斷移動距離


        ####tab2 三台相機

        self.label6 = tk.Label(self.tab2, text="Vision 1")
        self.label6.grid(row=0, column=0, padx=5, pady=1)
        self.video1_canvas = tk.Canvas(self.tab2, width=self.canvas_width, height=self.canvas_height)
        self.video1_canvas.grid(row=1, column=0, padx=5, pady=5)

        self.label7 = tk.Label(self.tab2, text="Vision 2")
        self.label7.grid(row=0, column=1, padx=5, pady=1)
        self.video2_canvas = tk.Canvas(self.tab2, width=self.canvas_width, height=self.canvas_height)
        self.video2_canvas.grid(row=1, column=1, padx=5, pady=5)

        self.label8 = tk.Label(self.tab2, text="Vision 3")
        self.label8.grid(row=0, column=2, padx=5, pady=1)
        self.video3_canvas = tk.Canvas(self.tab2, width=self.canvas_width, height=self.canvas_height)
        self.video3_canvas.grid(row=1, column=2, padx=5, pady=5)

        self.label9 = tk.Label(self.tab2, text="Vision 4")
        self.label9.grid(row=0, column=3, padx=5, pady=1)
        self.video4_canvas = tk.Canvas(self.tab2, width=self.canvas_width, height=self.canvas_height)
        self.video4_canvas.grid(row=1, column=3, padx=5, pady=5)

        self.label10 = tk.Label(self.tab2, text="Vision 5")
        self.label10.grid(row=0, column=4, padx=5, pady=1)
        self.video5_canvas = tk.Canvas(self.tab2, width=self.canvas_width, height=self.canvas_height)
        self.video5_canvas.grid(row=1, column=4, padx=5, pady=5)

        self.frame_buttons2 = tk.Frame(self.tab2)
        self.frame_buttons2.grid(row=2, column=0, columnspan=2, pady=10)

        self.open_file_btn = tk.Button(self.frame_buttons2, text="Open File", width=50, command=self.open_file)
        self.open_file_btn.pack(side=tk.BOTTOM, padx=10, pady=10)

        self.play_btn = tk.Button(self.frame_buttons2, text="Play", width=25, command=self.play_video)
        self.play_btn.pack(side=tk.LEFT, padx=10, pady=10)
        self.pause_btn = tk.Button(self.frame_buttons2, text="Pause", width=25, command=self.pause_video)
        self.pause_btn.pack(side=tk.RIGHT, padx=10, pady=10)
        
        self.progress = tk.Scale(self.tab2, from_=0, to=100, orient=tk.HORIZONTAL, length=500)
        self.progress.grid(row=5, column=0, columnspan=2, pady=10)

        #這邊好像跟回放的速度有關
        self.delay = 33
        self.update_interval = 17 # Update interval for video frames display  這個是調回放的影片的速度

        self.recording = False  # Initialize recording attribute
        self.playing = False  # Initialize playing attribute
        self.progress_dragging = False  # Initialize dragging attribute

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)  # Set closing event handler
        self.window.after(self.update_interval, self.update)  ###這邊應該回放的update
        self.window.after(self.update_interval, self.update_vision1_canvas)
        self.window.after(self.update_interval, self.update_vision2_canvas)
        self.window.after(self.update_interval, self.update_vision3_canvas)
        self.window.after(self.update_interval, self.update_vision4_canvas)
        self.window.after(self.update_interval, self.update_vision5_canvas)


        
        self.out1 = None
        self.out2 = None
        self.out3 = None
        self.out4 = None
        self.out5 = None
        
        self.vision1_cap = None
        self.vision2_cap = None
        self.vision3_cap = None
        self.vision4_cap = None
        self.vision5_cap = None

        self.current_frame = 0
        self.total_frames = 0
        
        self.progress.bind("<Button-1>", self.on_progress_drag_start)
        self.progress.bind("<ButtonRelease-1>", self.on_progress_drag_end)

        self.start_time1 = None
        self.start_time2 = None
        self.start_time3 = None
        self.start_time4 = None
        self.start_time5 = None
        self.frame_count1 = 0
        self.frame_count2 = 0
        self.frame_count3 = 0
        self.frame_count4 = 0
        self.frame_count5 = 0
        self.yolo_txt_file = None
        self.mediapipe_txt_file  = None
        # self.mediapipe_txt_file2  = None

        self.fps1 = 0
        self.fps2 = 0
        self.fps3 = 0
        self.fps4 = 0
        self.fps5 = 0
        self.fps_time1 = time.time()
        self.fps_time2 = time.time()
        self.fps_time3 = time.time()
        self.fps_time4 = time.time()
        self.fps_time5 = time.time()

        self.stop_event = threading.Event()

        # Start YOLO and MediaPipe threads
        self.yolo_thread = threading.Thread(target=self.process_vision1, daemon=True)
        self.mediapipe_thread = threading.Thread(target=self.process_vision2, daemon=True)
        self.mediapipe_thread_2 = threading.Thread(target=self.process_vision3, daemon=True)
        self.mediapipe_thread_4 = threading.Thread(target=self.process_vision4, daemon=True)
        self.mediapipe_thread_5 = threading.Thread(target=self.process_vision5, daemon=True)
        self.yolo_thread.start()
        self.mediapipe_thread.start()
        self.mediapipe_thread_2.start()
        self.mediapipe_thread_4.start()
        self.mediapipe_thread_5.start()
       
        
        self.window.mainloop()

    # Capture initial position function
    def capture_initial_position(self):
        # 隱藏 Start 按鈕
        self.btn_record.pack_forget()
        if self.vid1.isOpened():
            ret1, frame1 = self.vid1.get_frame()
            if ret1:
                # Detect the barbell position (YOLO model output)
                results = self.yolov8_model.predict(source=frame1, imgsz=320, conf=0.5)
                boxes = results[0].boxes

                if len(boxes.xywh) > 0:
                    self.initial_position = boxes.xywh[0]  # Capture the first detected box as the initial position
                    messagebox.showinfo("Info", "Initial position captured.")
                    self.recording = False  # Ensure recording is off initially
                    self.auto_recording = True  # Enable automatic recording trigger
                    self.threshold = 50  # Set a threshold for starting and stopping recording (can be adjusted)
                else:
                    messagebox.showerror("Error", "No detection found. Try again.")

    def restore_manual_control(self):
        # 顯示 Start 按鈕
        self.btn_record.pack(side=tk.BOTTOM, padx=10, pady=10)
        self.auto_recording = False
        messagebox.showinfo("Info", "Switched to manual recording control.")


    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
            self.btn_record.config(text="Stop")  # Change button text to "Stop"
        else:
            self.stop_recording()
            self.btn_record.config(text="Start")  # Change button text to "Start"


    def start_recording(self):
        if not self.recording:
            frame_count_for_detect1 = 0
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
                messagebox.showerror("Error", "Failed to initialize video recording")
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
            # self.mediapipe_txt_path2 = os.path.join(folder, "mediapipe_landmarks_2.txt")
            # self.mediapipe_txt_file2 = open(self.mediapipe_txt_path2, "w")
            # threading.Thread(target=self.record_screen).start()

            print("Recording started")
    # def record_screen(self):
    #     # 取得 tkinter 窗口的大小和位置
    #     x = self.window.winfo_rootx()
    #     y = self.window.winfo_rooty()
    #     width = self.window.winfo_width()
    #     height = self.window.winfo_height()

    #     # 設定影片儲存參數
    #     fourcc = cv2.VideoWriter_fourcc(*"XVID")
    #     now = datetime.now()
    #     timestamp = now.strftime("%Y%m%d_%H%M%S")
    #     folder = os.path.join(self.save_folder, f"recording_{timestamp}")
    #     os.makedirs(folder, exist_ok=True)
    #     file6 = os.path.join(folder, "gui_record.avi")
    #     self.video_writer = cv2.VideoWriter(file6, fourcc, 10.0, (width, height))

    #     while self.recording:
    #         # 擷取指定範圍畫面
    #         img = pyautogui.screenshot(region=(x, y, width, height))
    #         frame = np.array(img)
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #         # 寫入影片檔案
    #         self.video_writer.write(frame)

    #         # 可選擇顯示即時畫面
    #         # cv2.imshow("Recording", frame)
    #         # if cv2.waitKey(1) & 0xFF == ord("q"):
    #         #     break
            
    #         # 小延遲讓畫面更新流暢
    #         time.sleep(0.05)
    
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

            # if self.mediapipe_txt_file2 is not None:
            #     self.mediapipe_txt_file2.close()
            #     self.mediapipe_txt_file2 = None
                
            # messagebox.showinfo("Info", "Recording stopped and saved.")
            # Create an auto-closing message box
            self.show_auto_closing_message("Recording stopped and saved.", duration=2000)  # 2000 milliseconds = 2 seconds


    def show_auto_closing_message(self, message, duration=2000):
        # Create a new Toplevel window
        message_box = tk.Toplevel(self.window)
        message_box.title("Info")
        
        # Add a label with the message
        label = tk.Label(message_box, text=message)
        label.pack(padx=20, pady=20)

        # Center the message box on the main window
        x = self.window.winfo_x() + (self.window.winfo_width() // 2) - (message_box.winfo_width() // 2)
        y = self.window.winfo_y() + (self.window.winfo_height() // 2) - (message_box.winfo_height() // 2)
        message_box.geometry(f"+{x}+{y}")

        # After the specified duration, close the message box
        message_box.after(duration, message_box.destroy)

    def open_file(self):
        folder_path = self.folder
        if folder_path:
            file1 = os.path.join(folder_path, "vision1.avi")
            file2 = os.path.join(folder_path, "vision2.avi")
            file3 = os.path.join(folder_path, "vision3.avi")
            file4 = os.path.join(folder_path, "vision4.avi")
            file5 = os.path.join(folder_path, "vision5.avi")
            if os.path.exists(file1) and os.path.exists(file2) and os.path.exists(file3) and os.path.exists(file4) and os.path.exists(file5):
                self.vision1_cap = cv2.VideoCapture(file1)
                self.vision2_cap = cv2.VideoCapture(file2)
                self.vision3_cap = cv2.VideoCapture(file3)
                self.vision4_cap = cv2.VideoCapture(file4)
                self.vision5_cap = cv2.VideoCapture(file5)
                if not self.vision1_cap.isOpened() or not self.vision2_cap.isOpened() or not self.vision3_cap.isOpened() or not self.vision4_cap.isOpened() or not self.vision5_cap.isOpened():
                    messagebox.showerror("Error", "Failed to open video files.")
                else:
                    self.total_frames = int(self.vision1_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.progress.config(to=self.total_frames)
                    self.playing = True
                    self.current_frame = 0
            else:
                messagebox.showerror("Error", "Video files not found in the selected folder.")

    def play_video(self):
        self.playing = True

    def pause_video(self):
        self.playing = False

    def on_progress_drag_start(self, event):
        self.progress_dragging = True

    def on_progress_drag_end(self, event):
        self.progress_dragging = False
        frame_no = self.progress.get()
        if self.vision1_cap is not None:
            self.vision1_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        if self.vision2_cap is not None:
            self.vision2_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        if self.vision3_cap is not None:
            self.vision3_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        if self.vision4_cap is not None:
            self.vision4_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        if self.vision5_cap is not None:
            self.vision5_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        self.current_frame = frame_no

    def update(self):
        # Keep updating frames for display
        self.play_vision1()
        self.play_vision2()
        self.play_vision3()
        self.play_vision4()
        self.play_vision5()
        self.window.after(self.update_interval, self.update)

##################################################################################改成只存信心最高的
##########這邊有自動起回槓的功能
    def process_vision1(self):
        self.start_time1 = time.time()
        frame_count_for_detect1 = 0
        while True:
            ret1, frame1 = self.vid1.get_frame()
            if ret1:
                frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
                results = self.yolov8_model(source=frame1, imgsz=320, conf=0.5, verbose=False)
                boxes = results[0].boxes
                detected = False
                image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                for result in results:
                    frame1 = result.plot()
                for box in boxes.xywh:
                    detected = True
                    x_center, y_center, width, height = box
                    if self.recording and self.yolo_txt_file is not None:
                        frame_count_for_detect1 += 1
                        self.yolo_txt_file.write(f"{frame_count_for_detect1},{x_center},{y_center},{width},{height}\n")
                    
                    # Check for automatic recording
                    if self.auto_recording:
                        if self.initial_position is not None:
                            # Calculate the distance between the current and initial position
                            distance = ((x_center - self.initial_position[0]) ** 2 + (y_center - self.initial_position[1]) ** 2) ** 0.5
                            if distance > self.threshold:
                                if not self.recording:
                                    self.start_recording()  # Start recording if the distance exceeds threshold
                            elif distance < self.threshold / 2:  # Stop when within a smaller threshold
                                if self.recording:
                                    self.stop_recording()  # Stop recording

                    
                   

                if not detected and self.recording and self.yolo_txt_file is not None:
                    frame_count_for_detect1 += 1
                    self.yolo_txt_file.write(f"{frame_count_for_detect1},no detection\n")

                if self.recording is False:
                    frame_count_for_detect1 = 0
                # frame1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)

                # if self.recording and self.out1 is not None:
                #     self.out1.write(frame1)

                self.frame_count1 += 1
                elapsed_time = time.time() - self.start_time1
                if elapsed_time >= 1:
                    self.fps1 = self.frame_count1 / elapsed_time
                    self.frame_count1 = 0
                    self.start_time1 = time.time()
                cv2.putText(frame1, f'FPS: {self.fps1:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Update the image in the main thread
                self.window.after(0, self.update_vision1_canvas, frame1)
                frame1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)

                if self.recording and self.out1 is not None:
                    self.out1.write(frame1)

    

#####################################################################################

    def process_vision2(self):
        # Initialize the start time for FPS calculation
        self.start_time2 = time.time()  
        frame_count_for_detect2 = 0
        # face_indices = set(range(0, 11))
        
        while True:
            ret2, frame2 = self.vid2.get_frame()
            if ret2:
                frame2 = cv2.rotate(frame2, cv2.ROTATE_90_CLOCKWISE)
                # MediaPipe Pose Detection
                image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                # results = self.pose.process(image)
                results = self.model2(source=frame2, stream=True, verbose=False)
                frame_count_for_detect2 += 1
                keypoint_list = []

                # for result in results:
                #     frame2 = result.plot()

                for r2 in results:
                    boxes2 = r2.boxes
                    keypoints2 = r2.keypoints
                    
                    
                    for i, box2 in enumerate(boxes2):
                        x1, y1, x2, y2 = map(int, box2.xyxy[0])
                        # cv2.rectangle(frame2, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        confidence2 = round(box2.conf[0].item(), 2)
                        cls2 = int(box2.cls[0].item())
                        cv2.putText(frame2, f"{self.classNames2[cls2]} {confidence2}",
                                    (max(0, x1), max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        kpts2 = keypoints2[i]
                                        # 獲取關鍵點的座標和置信度
                        keypoints_xy = kpts2.xy  # shape: (1, 17, 2) -> 1 組 17 個關鍵點，每個關鍵點有 (x, y) 座標
                        keypoints_conf = kpts2.conf  # shape: (1, 17) -> 1 組 17 個關鍵點的置信度

                        for j in range(keypoints_xy.shape[1]):  # keypoints_xy.shape[1] 為 17，表示有 17 個關鍵點
                            
                            if results:
                                if self.recording and self.mediapipe_txt_file is not None:
                                    self.mediapipe_txt_file.write(f"{frame_count_for_detect2},{j},{int(keypoints_xy[0, j, 0])},{int(keypoints_xy[0, j, 1])}\n")
                            else:
                                # Write "no detection" if no landmarks are detected
                                if self.recording and self.mediapipe_txt_file is not None:
                                    self.mediapipe_txt_file.write(f"{frame_count_for_detect2},no detection\n")
                        # 繪製骨架

                        kp_coords = []
                        for kp in keypoints2.xy[i]:
                            x_kp, y_kp = int(kp[0].item()), int(kp[1].item())  # Get x, y coordinates
                            kp_coords.append((x_kp, y_kp))
                            cv2.circle(frame2, (x_kp, y_kp), 5, (0, 255, 0), cv2.FILLED)
                            print(x_kp, y_kp)

                        # Draw skeleton
                        for start_idx, end_idx in self.skeleton_connections:
                            if start_idx < len(kp_coords) and end_idx < len(kp_coords):
                                # Skip lines that connect to (0, 0)
                                if kp_coords[start_idx] == (0, 0) or kp_coords[end_idx] == (0, 0):
                                    continue
                                cv2.line(frame2, kp_coords[start_idx], kp_coords[end_idx], (0, 255, 255), 2)


                
                if self.recording is False:
                    frame_count_for_detect2 = 0

                # frame2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


                # if self.recording and self.out2 is not None:
                #     self.out2.write(frame2)

                self.frame_count2 += 1
                elapsed_time = time.time() - self.start_time2
                if elapsed_time >= 1:
                    self.fps2 = self.frame_count2 / elapsed_time
                    self.frame_count2 = 0
                    self.start_time2 = time.time()
                cv2.putText(frame2, f'FPS: {self.fps2:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Update the image in the main thread
                self.window.after(0, self.update_vision2_canvas, frame2)
                frame2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if self.recording and self.out2 is not None:
                    self.out2.write(frame2)


    def process_vision3(self):
        # Initialize the start time for FPS calculation
        self.start_time3 = time.time()  
        frame_count_for_detect3 = 0
        # face_indices = set(range(0, 11))
        while True:
            ret3, frame3 = self.vid3.get_frame()
            if ret3:
                frame3 = cv2.rotate(frame3, cv2.ROTATE_90_CLOCKWISE)
                # MediaPipe Pose Detection
                image = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
                results = self.pose2.process(image)
                frame_count_for_detect3 += 1  # Increment frame count for each frame
                


                if self.recording is False:
                    frame_count_for_detect3 = 0

                frame3 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if self.recording and self.out3 is not None:
                    self.out3.write(frame3)

                self.frame_count3 += 1
                elapsed_time = time.time() - self.start_time3
                if elapsed_time >= 1:
                    self.fps3 = self.frame_count3 / elapsed_time
                    self.frame_count3 = 0
                    self.start_time3 = time.time()
                cv2.putText(frame3, f'FPS: {self.fps3:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Update the image in the main thread
                self.window.after(0, self.update_vision3_canvas, frame3)
    def process_vision4(self):
        # Initialize the start time for FPS calculation
        self.start_time4 = time.time()  
        frame_count_for_detect4 = 0
        # face_indices = set(range(0, 11))
        excluded_indices = set(range(0, 11)) | set(range(25, 33))  # Exclude face (0-10) and leg (25-32) landmarks
        while True:
            ret4, frame4 = self.vid4.get_frame()
            if ret4:
                frame4 = cv2.rotate(frame4, cv2.ROTATE_90_CLOCKWISE)
                # MediaPipe Pose Detection
                image = cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB)
                frame_count_for_detect4 += 1  # Increment frame count for each frame
                

                if self.recording is False:
                    frame_count_for_detect4 = 0

                frame4 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if self.recording and self.out4 is not None:
                    self.out4.write(frame4)

                self.frame_count4 += 1
                elapsed_time = time.time() - self.start_time4
                if elapsed_time >= 1:
                    self.fps4 = self.frame_count4 / elapsed_time
                    self.frame_count4 = 0
                    self.start_time4 = time.time()
                cv2.putText(frame4, f'FPS: {self.fps4:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Update the image in the main thread
                self.window.after(0, self.update_vision4_canvas, frame4)

    def process_vision5(self):
        # Initialize the start time for FPS calculation
        self.start_time5 = time.time()  
        frame_count_for_detect5 = 0
        # face_indices = set(range(0, 11))
        while True:
            ret5, frame5 = self.vid5.get_frame()
            if ret5:
                frame5 = cv2.rotate(frame5, cv2.ROTATE_90_CLOCKWISE)
                # MediaPipe Pose Detection
                image = cv2.cvtColor(frame5, cv2.COLOR_BGR2RGB)
                frame_count_for_detect5 += 1  # Increment frame count for each frame

                if self.recording is False:
                    frame_count_for_detect5 = 0

                frame5 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if self.recording and self.out5 is not None:
                    self.out5.write(frame5)

                self.frame_count5 += 1
                elapsed_time = time.time() - self.start_time5
                if elapsed_time >= 1:
                    self.fps5 = self.frame_count5 / elapsed_time
                    self.frame_count5 = 0
                    self.start_time5 = time.time()
                cv2.putText(frame5, f'FPS: {self.fps5:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Update the image in the main thread
                self.window.after(0, self.update_vision5_canvas, frame5)


    def resize_frame(self, frame, max_width, max_height):
        # Get the original dimensions
        h, w = frame.shape[:2]
        
        # Calculate scaling factors
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h)  # Maintain aspect ratio
        
        # Calculate new dimensions
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        # Resize the frame while maintaining aspect ratio
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return resized_frame

    def update_vision1_canvas(self, frame1):
        # Resize the frame to fit the canvas
        resized_frame = self.resize_frame(frame1, self.canvas1.winfo_width(), self.canvas1.winfo_height())
        
        # Convert the resized frame to PhotoImage
        self.photo1 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)))
        
        # Update the canvas with the new image
        self.canvas1.create_image(0, 0, image=self.photo1, anchor=tk.NW)

    def update_vision2_canvas(self, frame2):
        resized_frame = self.resize_frame(frame2, self.canvas2.winfo_width(), self.canvas2.winfo_height())
        self.photo2 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)))
        self.canvas2.create_image(0, 0, image=self.photo2, anchor=tk.NW)

    def update_vision3_canvas(self, frame3):
        resized_frame = self.resize_frame(frame3, self.canvas3.winfo_width(), self.canvas3.winfo_height())
        self.photo3 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)))
        self.canvas3.create_image(0, 0, image=self.photo3, anchor=tk.NW)

    def update_vision4_canvas(self, frame4):
        resized_frame = self.resize_frame(frame4, self.canvas4.winfo_width(), self.canvas4.winfo_height())
        self.photo4 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)))
        self.canvas4.create_image(0, 0, image=self.photo4, anchor=tk.NW)

    def update_vision5_canvas(self, frame5):
        resized_frame = self.resize_frame(frame5, self.canvas5.winfo_width(), self.canvas5.winfo_height())
        self.photo5 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)))
        self.canvas5.create_image(0, 0, image=self.photo5, anchor=tk.NW)



    def draw_landmarks_custom(self, image, landmarks):
        excluded_indices = set(range(0, 11)) | set(range(25, 33))  # Exclude face (0-10) and leg (25-32) landmarks
        
        for idx, landmark in enumerate(landmarks.landmark):
            if idx not in excluded_indices:
                # 只畫出不在排除範圍內的 landmarks
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 255, 255), -1)  # Example: Draw a blue circle for landmarks

    

        # Draw connections
        # for connection in self.POSE_CONNECTIONS_CUSTOM:
        #     start_idx, end_idx = connection
        #     if start_idx not in excluded_indices and end_idx not in excluded_indices:
        #         start_landmark = landmarks.landmark[start_idx]
        #         end_landmark = landmarks.landmark[end_idx]
        #         x1, y1 = int(start_landmark.x * image.shape[1]), int(start_landmark.y * image.shape[0])
        #         x2, y2 = int(end_landmark.x * image.shape[1]), int(end_landmark.y * image.shape[0])
        #         cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    def play_vision1(self):
        if self.vision1_cap is not None and self.playing:
            ret, frame = self.vision1_cap.read()
            if ret:
                # Get the canvas size
                canvas_width = self.video1_canvas.winfo_width()
                canvas_height = self.video1_canvas.winfo_height()

                # Resize the frame to fit the canvas while maintaining the aspect ratio
                frame_resized = self.resize_frame(frame, canvas_width, canvas_height)

                # Display the resized frame
                self.photo6 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))
                self.video1_canvas.create_image(0, 0, image=self.photo6, anchor=tk.NW)

                # Update progress
                if not self.progress_dragging:
                    self.current_frame += 1
                    self.progress.set(self.current_frame)

    def play_vision2(self):
        if self.vision2_cap is not None and self.playing:
            ret, frame = self.vision2_cap.read()
            if ret:
                # Get the canvas size
                canvas_width = self.video2_canvas.winfo_width()
                canvas_height = self.video2_canvas.winfo_height()

                # Resize the frame to fit the canvas while maintaining the aspect ratio
                frame_resized = self.resize_frame(frame, canvas_width, canvas_height)

                # Display the resized frame
                self.photo7 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))
                self.video2_canvas.create_image(0, 0, image=self.photo7, anchor=tk.NW)

    def play_vision3(self):
        if self.vision3_cap is not None and self.playing:
            ret, frame = self.vision3_cap.read()
            if ret:
                # Get the canvas size
                canvas_width = self.video3_canvas.winfo_width()
                canvas_height = self.video3_canvas.winfo_height()

                # Resize the frame to fit the canvas while maintaining the aspect ratio
                frame_resized = self.resize_frame(frame, canvas_width, canvas_height)

                # Display the resized frame
                self.photo8 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))
                self.video3_canvas.create_image(0, 0, image=self.photo8, anchor=tk.NW)

    def play_vision4(self):
        if self.vision4_cap is not None and self.playing:
            ret, frame = self.vision4_cap.read()
            if ret:
                # Get the canvas size
                canvas_width = self.video4_canvas.winfo_width()
                canvas_height = self.video4_canvas.winfo_height()

                # Resize the frame to fit the canvas while maintaining the aspect ratio
                frame_resized = self.resize_frame(frame, canvas_width, canvas_height)

                # Display the resized frame
                self.photo9 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))
                self.video4_canvas.create_image(0, 0, image=self.photo9, anchor=tk.NW)

    def play_vision5(self):
        if self.vision5_cap is not None and self.playing:
            ret, frame = self.vision5_cap.read()
            if ret:
                # Get the canvas size
                canvas_width = self.video5_canvas.winfo_width()
                canvas_height = self.video5_canvas.winfo_height()

                # Resize the frame to fit the canvas while maintaining the aspect ratio
                frame_resized = self.resize_frame(frame, canvas_width, canvas_height)

                # Display the resized frame
                self.photo10 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))
                self.video5_canvas.create_image(0, 0, image=self.photo10, anchor=tk.NW)

    def on_closing(self):
        # Stop recording if active
        self.stop_recording()
        
        # Stop the threads
        self.stop_event.set()
        
        # Release video capture objects
        if self.vision1_cap is not None:
            self.vision1_cap.release()
            self.vision1_cap = None
        if self.vision2_cap is not None:
            self.vision2_cap.release()
            self.vision2_cap = None

        if self.vision3_cap is not None:
            self.vision3_cap.release()
            self.vision3_cap = None

        if self.vision4_cap is not None:
            self.vision4_cap.release()
            self.vision4_cap = None

        if self.vision5_cap is not None:
            self.vision5_cap.release()
            self.vision5_cap = None
        
        # Release video writer objects
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
        
        # Destroy the tkinter window
        self.window.destroy()
        
        # Cleanup video capture objects
        self.vid1.__del__()
        self.vid2.__del__()
        self.vid3.__del__()
        self.vid4.__del__()
        self.vid5.__del__()



class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

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

def main():
    save_folder = "./recordings"
    os.makedirs(save_folder, exist_ok=True)
    WebcamRecorder(tk.Tk(), "Deadlift Video Recorder", save_folder)

if __name__ == "__main__":
    main()
