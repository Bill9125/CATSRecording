
from ultralytics import YOLO
import os
import cv2
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk
from datetime import datetime
import time
import mediapipe as mp
import torch
import threading
import interpolate_function
import numpy as np

# Check CUDA availability
print("Is CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0))
print(f"CUDA available: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")

class WebcamRecorder:
    def __init__(self, window, window_title, save_folder='.', video_source1=0, video_source2=1, video_source3=2):
        self.window = window
        self.window.title(window_title)
        self.save_folder = save_folder
        self.video_source1 = video_source2
        self.video_source2 = video_source3
        self.video_source3 = video_source1

        # Initialize YOLO model
        self.yolov8_model = YOLO("C:\\Users\\92A27\\benchpress\\yolo_bar_model\\best.pt")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('yolo device:', device)
        self.yolov8_model.to(device)

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


        self.POSE_CONNECTIONS_CUSTOM = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Upper body joints
            (11, 23), (12, 24), (23, 24),  # Torso connections
        ]

        self.POSE_CONNECTIONS_CUSTOM2 = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Upper body joints
            (11, 23), (12, 24), (23, 24),  # Torso connections
        ]


        self.tab_control = ttk.Notebook(window)
        
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.tab1, text='Record')
        self.tab_control.add(self.tab2, text='Open File')
        self.tab_control.pack(expand=1, fill='both')
        
        self.vid1 = MyVideoCapture(self.video_source1)
        self.vid2 = MyVideoCapture(self.video_source2)
        self.vid3 = MyVideoCapture(self.video_source3)

        self.canvas_width = 480
        self.canvas_height = 360
        #####tab1 三台相機

        self.label1 = tk.Label(self.tab1, text="Vision 1")
        self.label1.grid(row=0, column=0, padx=5, pady=5)
        self.canvas1 = tk.Canvas(self.tab1, width=self.canvas_width, height=self.canvas_height)
        self.canvas1.grid(row=1, column=0, padx=5, pady=5)

        self.label2 = tk.Label(self.tab1, text="Vision 2")
        self.label2.grid(row=0, column=1, padx=5, pady=5)
        self.canvas2 = tk.Canvas(self.tab1, width=self.canvas_width, height=self.canvas_height)
        self.canvas2.grid(row=1, column=1, padx=5, pady=5)

        self.label3 = tk.Label(self.tab1, text="Vision 3")
        self.label3.grid(row=0, column=2, padx=5, pady=5)
        self.canvas3 = tk.Canvas(self.tab1, width=self.canvas_width, height=self.canvas_height)
        self.canvas3.grid(row=1, column=2, padx=5, pady=5)


        self.frame_buttons = tk.Frame(self.tab1)
        self.frame_buttons.grid(row=2, column=0, columnspan=2, pady=10)

        # Combine Start and Stop into a single button
        # Start recording button
        self.btn_record = tk.Button(self.frame_buttons, text="Start Record", width=50, command=self.toggle_recording)
        self.btn_record.pack(side=tk.BOTTOM, padx=10, pady=10)

        # Capture Initial Position button (for automatic start/stop)
        self.capture_button = tk.Button(self.frame_buttons, text="Auto Record", width=50, command=self.capture_initial_position)
        self.capture_button.pack(side=tk.BOTTOM, padx=10, pady=10)

        # Add a button to restore manual recording control
        self.restore_button = tk.Button(self.frame_buttons, text="Restore Manual Record", width=50, command=self.restore_manual_control)
        self.restore_button.pack(side=tk.BOTTOM, padx=10, pady=10)


        self.auto_recording = False  # 初始化 auto_recording 屬性
        self.initial_position = None  # 初始化 initial_position，作為初始座標
        self.threshold = 50  # 初始化 threshold，用於判斷移動距離


        ####tab2 三台相機

        # Video 1
        self.label4 = tk.Label(self.tab2, text="Vision 1")
        self.label4.grid(row=0, column=0, padx=5, pady=5)
        self.video1_canvas = tk.Canvas(self.tab2, width=self.canvas_width, height=self.canvas_height)
        self.video1_canvas.grid(row=1, column=0, padx=5, pady=5)

        # Video 2
        self.label5 = tk.Label(self.tab2, text="Vision 2")
        self.label5.grid(row=0, column=1, padx=5, pady=5)
        self.video2_canvas = tk.Canvas(self.tab2, width=self.canvas_width, height=self.canvas_height)
        self.video2_canvas.grid(row=1, column=1, padx=5, pady=5)

        # Video 3
        self.label6 = tk.Label(self.tab2, text="Vision 3")
        self.label6.grid(row=0, column=2, padx=5, pady=5)
        self.video3_canvas = tk.Canvas(self.tab2, width=self.canvas_width, height=self.canvas_height)
        self.video3_canvas.grid(row=1, column=2, padx=5, pady=5)

        # Buttons Frame
        self.frame_buttons2 = tk.Frame(self.tab2)
        self.frame_buttons2.grid(row=2, column=0, columnspan=3, pady=10)  # 調整 columnspan 覆蓋 3 列

        # Buttons
        self.open_file_btn = tk.Button(self.frame_buttons2, text="Open File", width=50, command=self.open_file)
        self.open_file_btn.pack(side=tk.BOTTOM, padx=10, pady=10)

        self.play_btn = tk.Button(self.frame_buttons2, text="Play", width=25, command=self.play_video)
        self.play_btn.pack(side=tk.LEFT, padx=5, pady=10)
        self.pause_btn = tk.Button(self.frame_buttons2, text="Pause", width=25, command=self.pause_video)
        self.pause_btn.pack(side=tk.LEFT, padx=5, pady=10)
        self.trajectory_btn = tk.Button(self.frame_buttons2, text="Trajectory", width=25, command=self.show_trajectory)
        self.trajectory_btn.pack(side=tk.LEFT, padx=5, pady=10)

        # Progress Bar
        self.progress = tk.Scale(self.tab2, from_=0, to=100, orient=tk.HORIZONTAL, length=500)
        self.progress.grid(row=3, column=0, columnspan=3, pady=10, sticky="ew")  # 調整成與畫布一致

        #這邊跟回放的速度有關
        # self.delay = 33
        self.update_interval = 33 # Update interval for video frames display  這個是調回放的影片的速度

        self.recording = False  # Initialize recording attribute
        self.playing = False  # Initialize playing attribute
        self.progress_dragging = False  # Initialize dragging attribute

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)  # Set closing event handler
        self.window.after(self.update_interval, self.update)  ###這邊應該回放的update
        self.window.after(self.update_interval, self.update_vision1_canvas)
        self.window.after(self.update_interval, self.update_vision2_canvas)
        self.window.after(self.update_interval, self.update_vision3_canvas)

        
        self.out1 = None
        self.out2 = None
        self.out3 = None

        self.original_out1 = None
        self.original_out2 = None
        self.original_out3 = None
        
        self.vision1_cap = None
        self.vision2_cap = None
        self.vision3_cap = None
        self.current_frame = 0
        self.total_frames = 0
        
        self.progress.bind("<Button-1>", self.on_progress_drag_start)
        self.progress.bind("<ButtonRelease-1>", self.on_progress_drag_end)

        self.start_time1 = None
        self.start_time2 = None
        self.start_time3 = None
        self.frame_count1 = 0
        self.frame_count2 = 0
        self.frame_count3 = 0
        self.yolo_txt_file = None
        self.mediapipe_txt_file  = None
        self.mediapipe_txt_file2  = None

        self.fps1 = 0
        self.fps2 = 0
        self.fps3 = 0
        self.fps_time1 = time.time()
        self.fps_time2 = time.time()
        self.fps_time3 = time.time()

        self.stop_event = threading.Event()

        # Start YOLO and MediaPipe threads
        self.yolo_thread = threading.Thread(target=self.process_vision1, daemon=True)
        self.mediapipe_thread = threading.Thread(target=self.process_vision2, daemon=True)
        self.mediapipe_thread_2 = threading.Thread(target=self.process_vision3, daemon=True)
        self.yolo_thread.start()
        self.mediapipe_thread.start()
        self.mediapipe_thread_2.start()

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
            self.recording_folder = os.path.join(self.save_folder, f"recording_{timestamp}")
            os.makedirs(self.recording_folder, exist_ok=True)

            ###video after process
            file1 = os.path.join(self.recording_folder, "vision1.avi")
            file2 = os.path.join(self.recording_folder, "vision2.avi")
            file3 = os.path.join(self.recording_folder, "vision3.avi")

            ###original video
            original_vision1_file = os.path.join(self.recording_folder, "original_vision1.avi")
            original_vision2_file = os.path.join(self.recording_folder, "original_vision2.avi")
            original_vision3_file = os.path.join(self.recording_folder, "original_vision3.avi")
            
            ###processed video fps
            fps1 = 29
            fps2 = 29
            fps3 = 29

            ###original video fps
            fps1_original = 29
            fps2_original = 29
            fps3_original = 29
            
            self.out1 = cv2.VideoWriter(file1, cv2.VideoWriter_fourcc(*'XVID'), fps1, (int(self.vid1.width), int(self.vid1.height)))
            self.out2 = cv2.VideoWriter(file2, cv2.VideoWriter_fourcc(*'XVID'), fps2, (int(self.vid2.width), int(self.vid2.height)))
            self.out3 = cv2.VideoWriter(file3, cv2.VideoWriter_fourcc(*'XVID'), fps3, (int(self.vid3.width), int(self.vid3.height)))


            # 初始化原始影片的 VideoWriter
            self.original_out1 = cv2.VideoWriter(original_vision1_file, cv2.VideoWriter_fourcc(*'XVID'), fps1_original, (int(self.vid1.width), int(self.vid1.height)))
            self.original_out2 = cv2.VideoWriter(original_vision2_file, cv2.VideoWriter_fourcc(*'XVID'), fps2_original, (int(self.vid1.width), int(self.vid1.height)))
            self.original_out3 = cv2.VideoWriter(original_vision3_file, cv2.VideoWriter_fourcc(*'XVID'), fps3_original, (int(self.vid1.width), int(self.vid1.height)))

            # 檢查 VideoWriter 是否成功初始化
            if not all([self.original_out1.isOpened(), self.original_out2.isOpened(), self.original_out3.isOpened()]):
                messagebox.showerror("Error", "無法初始化影片錄製")
                self.recording = False
                return


            if not self.out1.isOpened() or not self.out2.isOpened() or not self.out3.isOpened():
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

            self.start_time1 = time.time()
            self.start_time2 = time.time()
            self.start_time3 = time.time()
            self.frame_count1 = 0
            self.frame_count2 = 0
            self.frame_count3 = 0
            self.frame_count_for_detect1 = 0
            
            # Initialize text files for saving coordinates
            self.yolo_txt_path = os.path.join(self.recording_folder, "yolo_coordinates.txt")
            self.mediapipe_txt_path = os.path.join(self.recording_folder, "mediapipe_landmarks.txt")
            self.yolo_txt_file = open(self.yolo_txt_path, "w")
            self.mediapipe_txt_file = open(self.mediapipe_txt_path, "w")
            self.mediapipe_txt_path2 = os.path.join(self.recording_folder, "mediapipe_landmarks_2.txt")
            self.mediapipe_txt_file2 = open(self.mediapipe_txt_path2, "w")

            print("Recording started")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.stop_event.set()  # Signal threads to stop writing

            # 釋放所有資源
            if self.out1 is not None:
                self.out1.release()
                self.out1 = None
            if self.out2 is not None:
                self.out2.release()
                self.out2 = None
            if self.out3 is not None:
                self.out3.release()
                self.out3 = None

            if self.original_out1 is not None:
                self.original_out1.release()
                self.original_out1 = None
            if self.original_out2 is not None:
                self.original_out2.release()
                self.original_out2 = None
            if self.original_out3 is not None:
                self.original_out3.release()
                self.original_out3 = None

            if self.yolo_txt_file is not None:
                self.yolo_txt_file.close()
                self.yolo_txt_file = None

            if self.mediapipe_txt_file is not None:
                self.mediapipe_txt_file.close()
                self.mediapipe_txt_file = None

            if self.mediapipe_txt_file2 is not None:
                self.mediapipe_txt_file2.close()
                self.mediapipe_txt_file2 = None

            # 呼叫內插處理
            self.run_interpolation()

            # 繪製軌跡並生成影片
            self.generate_trajectory_video()

            # 顯示通知訊息
            self.show_auto_closing_message("Recording stopped and saved.", duration=2000)  # 2000 milliseconds = 2 seconds

    def run_interpolation(self):
        # 確認資料夾路徑是正確的
        latest_folder_path = self.recording_folder

        # 設定輸入檔案
        mediapipe_files = [
            os.path.join(latest_folder_path, "mediapipe_landmarks.txt"),
            os.path.join(latest_folder_path, "mediapipe_landmarks_2.txt")
        ]

        # 檢查檔案是否存在，若不存在則不執行插值
        for input_file in mediapipe_files:
            if not os.path.exists(input_file):
                print(f"File not found: {input_file}. Interpolation will not be performed.")
                return

        # 設定輸出檔案
        mediapipe_output_files = [
            os.path.join(latest_folder_path, "mediapipe_landmarks_1st_interp.txt"),
            os.path.join(latest_folder_path, "mediapipe_landmarks_2_1st_interp.txt")
        ]

        # # 進行第一步和第二步的插值
        # for input_file, output_file in zip(mediapipe_files, mediapipe_output_files):
        #     interpolate_function.interpolate_landmarks(input_file)

        # 進行第一步和第二步的插值
        for input_file, output_file in zip(mediapipe_files, mediapipe_output_files):
            interpolated_data = interpolate_function.interpolate_landmarks(input_file)
            np.savetxt(output_file, interpolated_data.values, delimiter=',')


        # 載入 YOLO 資料
        yolo_data = interpolate_function.load_yolo_data(os.path.join(latest_folder_path, 'yolo_coordinates.txt'))

        # 內插 YOLO 中的 "no detection" 資料
        interpolated_yolo_data = interpolate_function.interpolate_missing_detections(yolo_data)
        yolo_output_file = os.path.join(latest_folder_path, 'yolo_coordinates_interpolated.txt')
        np.savetxt(yolo_output_file, interpolated_yolo_data, delimiter=',', fmt='%d,%.8f,%.8f,%.8f,%.8f')

        print("YOLO 'no detection' frames have been interpolated and saved.")

        # 讀取插值後的 YOLO 資料
        yolo_data = np.loadtxt(yolo_output_file, delimiter=',')
        yolo_frames = yolo_data[:, 0]  # frame numbers

        # 讀取第一份 MediaPipe 的資料
        mediapipe_data_1 = np.loadtxt(mediapipe_output_files[0], delimiter=',')
        landmarks_1 = np.unique(mediapipe_data_1[:, 1])  # unique landmark numbers

        # 讀取第二份 MediaPipe 的資料
        mediapipe_data_2 = np.loadtxt(mediapipe_output_files[1], delimiter=',')
        landmarks_2 = np.unique(mediapipe_data_2[:, 1])  # unique landmark numbers

        # 處理第一份 MediaPipe 資料
        interpolated_data_1 = interpolate_function.interpolate_mediapipe_to_yolo(yolo_frames, mediapipe_data_1)

        # 處理第二份 MediaPipe 資料
        interpolated_data_2 = interpolate_function.interpolate_mediapipe_to_yolo(yolo_frames, mediapipe_data_2)


        # 將內插後的資料寫入 TXT 檔案
        with open(os.path.join(latest_folder_path, 'interpolated_mediapipe_landmarks_1.txt'), 'w') as f:
            for entry in interpolated_data_1:
                f.write(f"{entry[0]},{entry[1]},{entry[2]},{entry[3]},{entry[4]}\n")

        with open(os.path.join(latest_folder_path, 'interpolated_mediapipe_landmarks_2.txt'), 'w') as f:
            for entry in interpolated_data_2:
                f.write(f"{entry[0]},{entry[1]},{entry[2]},{entry[3]},{entry[4]}\n")

        print("Interpolation complete and results saved.")

    def generate_trajectory_video(self):
        # 確認檔案路徑
        latest_folder_path = self.recording_folder
        yolo_output_file = os.path.join(latest_folder_path, 'yolo_coordinates_interpolated.txt')
        original_video_path = os.path.join(latest_folder_path, 'original_vision1.avi')
        trajectory_video_path = os.path.join(latest_folder_path, 'vision1_trajectory.avi')

        # 檢查檔案是否存在
        if not os.path.exists(yolo_output_file) or not os.path.exists(original_video_path):
            print("Required files not found. Skipping trajectory video generation.")
            return

        # 讀取 YOLO 座標
        coordinates = {}
        with open(yolo_output_file, 'r') as file:
            for line in file:
                if line.strip():  # 跳過空行
                    data = line.strip().split(',')  # 使用逗號分隔
                    frame_number = int(data[0])    # 幀號
                    x = float(data[1])             # X 座標
                    y = float(data[2])             # Y 座標
                    coordinates[frame_number] = (int(x), int(y))  # 儲存座標

        # 打開原始影片
        cap = cv2.VideoCapture(original_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(trajectory_video_path, fourcc, fps, (width, height))

        # 設定繪圖參數
        trajectory_window = 70  # 顯示最近 N 幀點
        draw_interval = 2        # 每隔幾幀新增新的軌跡點
        full_trajectory = []     # 儲存完整的軌跡點

        # 開始處理影片
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 只在間隔幀加入新的點
            if frame_count in coordinates and frame_count % draw_interval == 0:
                full_trajectory.append(coordinates[frame_count])  # 加入座標

            # 限制軌跡範圍為最近 N 幀
            trajectory = full_trajectory[-trajectory_window:]

            # 繪製軌跡
            for i in range(len(trajectory)):
                # 計算顏色深度
                alpha = int(255 * (i + 1) / len(trajectory))
                color = (255, 255 - alpha, 255)  # 漸層藍色
                size = 6  # 叉叉大小
                center = trajectory[i]
                cv2.line(frame, (center[0] - size, center[1] - size), (center[0] + size, center[1] + size), color, thickness=2)
                cv2.line(frame, (center[0] - size, center[1] + size), (center[0] + size, center[1] - size), color, thickness=2)

            # 寫入輸出影片
            out.write(frame)

        cap.release()
        out.release()
        print(f"Trajectory video saved to: {trajectory_video_path}")

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
        folder_path = filedialog.askdirectory()
        if folder_path:
            file1 = os.path.join(folder_path, "vision1.avi")
            file2 = os.path.join(folder_path, "vision2.avi")
            file3 = os.path.join(folder_path, "vision3.avi")
            file_trajectory = os.path.join(folder_path, "vision1_trajectory.avi")

            # 檢查檔案是否存在
            if os.path.exists(file1) and os.path.exists(file2) and os.path.exists(file3) and os.path.exists(file_trajectory):
                self.vision1_cap = cv2.VideoCapture(file1)
                self.vision2_cap = cv2.VideoCapture(file2)
                self.vision3_cap = cv2.VideoCapture(file3)
                self.trajectory_cap = cv2.VideoCapture(file_trajectory)

                # 確認影片是否成功開啟
                if not all([self.vision1_cap.isOpened(), self.vision2_cap.isOpened(), self.vision3_cap.isOpened(), self.trajectory_cap.isOpened()]):
                    messagebox.showerror("Error", "Failed to open one or more video files.")
                    return

                # 設定進度條範圍
                self.total_frames = int(self.vision1_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.progress.config(to=self.total_frames)
                self.current_frame = 0
            else:
                messagebox.showerror("Error", "Required video files not found.")

    def play_video(self):
        self.playing = True
        self.trajectory_mode = False
        self.update()

    def pause_video(self):
        self.playing = False

    def show_trajectory(self):
        if self.trajectory_cap is None:
            messagebox.showerror("Error", "Trajectory video not loaded.")
            return
        self.playing = True
        self.trajectory_mode = True
        self.update()

    def update(self):
        if not self.playing:
            return

        # 設定檔案的影片來源
        caps = [self.vision1_cap, self.vision2_cap, self.vision3_cap] if not self.trajectory_mode else \
            [self.vision2_cap, self.vision3_cap, self.trajectory_cap]
        canvases = [self.video1_canvas, self.video2_canvas, self.video3_canvas] if not self.trajectory_mode else \
                [self.video2_canvas, self.video3_canvas, self.video1_canvas]

        for cap, canvas in zip(caps, canvases):
            if cap:
                ret, frame = cap.read()
                if ret:
                    # 獲取幀的原始大小
                    frame_height, frame_width, _ = frame.shape

                    # 設置Canvas大小為影片原始大小
                    canvas.config(width=frame_width, height=frame_height)

                    # 將OpenCV影像轉為PIL影像
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = ImageTk.PhotoImage(Image.fromarray(frame))

                    # 清空Canvas，避免畫面疊加
                    canvas.delete("all")

                    # 將圖片顯示在Canvas上
                    canvas.create_image(0, 0, anchor=tk.NW, image=img)
                    canvas.image = img
                else:
                    self.playing = False  # 如果影片結束則停止播放

        self.current_frame += 1
        self.progress.set(self.current_frame)

        # 設置下一幀更新
        if self.playing:
            self.window.after(33, self.update)

    def on_progress_drag_start(self, event):
        self.playing = False

    def on_progress_drag_end(self, event):
        frame_no = self.progress.get()
        self.current_frame = frame_no
        caps = [self.vision1_cap, self.vision2_cap, self.vision3_cap, self.trajectory_cap]
        for cap in caps:
            if cap:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        root = tk.Tk()
        app = WebcamRecorder(root)
        root.mainloop()



##################################################################################改成只存信心最高的
##########這邊有自動起回槓的功能
    def process_vision1(self):
        self.start_time1 = time.time()
        frame_count_for_detect1 = 0
        while True:
            ret1, frame1 = self.vid1.get_frame()
            if ret1:
                # Save the original video frame
                if self.recording and self.original_out1 is not None:
                    self.original_out1.write(frame1)

                # Perform detection
                results = self.yolov8_model.predict(source=frame1, imgsz=320, conf=0.5, verbose=False)
                boxes = results[0].boxes
                detected = False  # Initialize detected to False at the start of each frame
                for result in results:
                    frame1 = result.plot()
                if boxes is not None and len(boxes) > 0:
                    # Select the box with the highest confidence
                    max_confidence_index = boxes.conf.argmax()  # Get index of highest confidence
                    best_box = boxes[max_confidence_index].xywh[0]  # Ensure it's a flat array or list

                    # Check if best_box has the required four elements
                    if len(best_box) == 4:
                        detected = True
                        x_center, y_center, width, height = best_box
                        if self.recording and self.yolo_txt_file is not None:
                            frame_count_for_detect1 += 1
                            self.yolo_txt_file.write(f"{frame_count_for_detect1},{x_center},{y_center},{width},{height}\n")

                        # Check for automatic recording based on movement
                        if self.auto_recording and self.initial_position is not None:
                            distance = ((x_center - self.initial_position[0]) ** 2 + (y_center - self.initial_position[1]) ** 2) ** 0.5
                            if distance > self.threshold and not self.recording:
                                self.start_recording()
                            elif distance < self.threshold / 2 and self.recording:
                                self.stop_recording()

                # Handle case where no detection is made
                if not detected and self.recording and self.yolo_txt_file is not None:
                    frame_count_for_detect1 += 1
                    self.yolo_txt_file.write(f"{frame_count_for_detect1},no detection\n")

                # Handle frame recording and FPS calculation
                if not self.recording:
                    frame_count_for_detect1 = 0
                if self.recording and self.out1 is not None:
                    self.out1.write(frame1)

                self.frame_count1 += 1
                elapsed_time = time.time() - self.start_time1
                if elapsed_time >= 1:
                    self.fps1 = self.frame_count1 / elapsed_time
                    self.frame_count1 = 0
                    self.start_time1 = time.time()
                cv2.putText(frame1, f'FPS: {self.fps1:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Update the image in the main thread
                self.window.after(0, self.update_vision1_canvas, frame1)

#####################################################################################

    def process_vision2(self):
        # Initialize the start time for FPS calculation
        self.start_time2 = time.time()  
        frame_count_for_detect2 = 0
        if self.original_out2 is not None:
            self.original_out2.write(frame2)
        # face_indices = set(range(0, 11))
        excluded_indices1 = set(range(0, 11)) | set(range(25, 33)) | set(range(15, 23))  # Exclude face (0-10) and leg (25-32) landmarks
        while True:
            ret2, frame2 = self.vid2.get_frame()
            if ret2:
            # 儲存原始影像幀
                if self.recording and self.original_out2 is not None:
                    self.original_out2.write(frame2)

            if ret2:
                # MediaPipe Pose Detection
                image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image)
                frame_count_for_detect2 += 1  # Increment frame count for each frame
                
                if results.pose_landmarks:
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        if idx not in excluded_indices1:
                            x, y, z = landmark.x, landmark.y, landmark.z
                            if self.recording and self.mediapipe_txt_file is not None:
                                self.mediapipe_txt_file.write(f"{frame_count_for_detect2},{idx},{x},{y},{z}\n")
                else:
                    # Write "no detection" if no landmarks are detected
                    if self.recording and self.mediapipe_txt_file is not None:
                        self.mediapipe_txt_file.write(f"{frame_count_for_detect2},no detection\n")

                if self.recording is False:
                    frame_count_for_detect2 = 0

                if results.pose_landmarks:
                    self.draw_landmarks_custom1(image, results.pose_landmarks)

                frame2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                frame2 = cv2.rotate(frame2, cv2.ROTATE_180)

                if self.recording and self.out2 is not None:
                    self.out2.write(frame2)

                self.frame_count2 += 1
                elapsed_time = time.time() - self.start_time2
                if elapsed_time >= 1:
                    self.fps2 = self.frame_count2 / elapsed_time
                    self.frame_count2 = 0
                    self.start_time2 = time.time()
                cv2.putText(frame2, f'FPS: {self.fps2:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Update the image in the main thread
                self.window.after(0, self.update_vision2_canvas, frame2)

    def process_vision3(self):
        # Initialize the start time for FPS calculation
        self.start_time3 = time.time()  
        frame_count_for_detect3 = 0
        if self.original_out3 is not None:
            # frame3 = cv2.rotate(frame3, cv2.ROTATE_180)
            self.original_out3.write(frame3)
        # face_indices = set(range(0, 11))
        excluded_indices2 = set(range(0, 11)) | set(range(25, 33))  # Exclude face (0-10) and leg (25-32) landmarks
        while True:
            ret3, frame3 = self.vid3.get_frame()
            frame3 = cv2.rotate(frame3, cv2.ROTATE_180)
            if ret3:
            # 儲存原始影像幀
                if self.recording and self.original_out3 is not None:
                    self.original_out3.write(frame3)
            if ret3:
                # MediaPipe Pose Detection
                image = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
                results = self.pose2.process(image)
                frame_count_for_detect3 += 1  # Increment frame count for each frame
                
                if results.pose_landmarks:
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        if idx not in excluded_indices2:
                            x, y, z = landmark.x, landmark.y, landmark.z
                            if self.recording and self.mediapipe_txt_file2 is not None:
                                self.mediapipe_txt_file2.write(f"{frame_count_for_detect3},{idx},{x},{y},{z}\n")
                else:
                    # Write "no detection" if no landmarks are detected
                    if self.recording and self.mediapipe_txt_file2 is not None:
                        self.mediapipe_txt_file2.write(f"{frame_count_for_detect3},no detection\n")

                if self.recording is False:
                    frame_count_for_detect3 = 0

                if results.pose_landmarks:
                    self.draw_landmarks_custom2(image, results.pose_landmarks)

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

    def draw_landmarks_custom1(self, image, landmarks):
        # excluded_indices1 = set(range(0, 11)) | set(range(25, 33))  # Exclude face (0-10) and leg (25-32) landmarks
        excluded_indices1 = set(range(0, 11)) | set(range(25, 33)) | set(range(15, 23))
        for idx, landmark in enumerate(landmarks.landmark):
            if idx not in excluded_indices1:
                # 只畫出不在排除範圍內的 landmarks
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 255, 255), -1)  # Example: Draw a blue circle for landmarks

    

        # Draw connections
        for connection in self.POSE_CONNECTIONS_CUSTOM2:
            start_idx, end_idx = connection
            if start_idx not in excluded_indices1 and end_idx not in excluded_indices1:
                start_landmark = landmarks.landmark[start_idx]
                end_landmark = landmarks.landmark[end_idx]
                x1, y1 = int(start_landmark.x * image.shape[1]), int(start_landmark.y * image.shape[0])
                x2, y2 = int(end_landmark.x * image.shape[1]), int(end_landmark.y * image.shape[0])
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)


    def draw_landmarks_custom2(self, image, landmarks):
        excluded_indices2 = set(range(0, 11)) | set(range(25, 33))  # Exclude face (0-10) and leg (25-32) landmarks
        
        for idx, landmark in enumerate(landmarks.landmark):
            if idx not in excluded_indices2:
                # 只畫出不在排除範圍內的 landmarks
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 255, 255), -1)  # Example: Draw a blue circle for landmarks

    

        # Draw connections
        for connection in self.POSE_CONNECTIONS_CUSTOM2:
            start_idx, end_idx = connection
            if start_idx not in excluded_indices2 and end_idx not in excluded_indices2:
                start_landmark = landmarks.landmark[start_idx]
                end_landmark = landmarks.landmark[end_idx]
                x1, y1 = int(start_landmark.x * image.shape[1]), int(start_landmark.y * image.shape[0])
                x2, y2 = int(end_landmark.x * image.shape[1]), int(end_landmark.y * image.shape[0])
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

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
                self.photo4 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))
                self.video1_canvas.create_image(0, 0, image=self.photo4, anchor=tk.NW)

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
                self.photo5 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))
                self.video2_canvas.create_image(0, 0, image=self.photo5, anchor=tk.NW)

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
                self.photo6 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))
                self.video3_canvas.create_image(0, 0, image=self.photo6, anchor=tk.NW)

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
        
        # Destroy the tkinter window
        self.window.destroy()
        
        # Cleanup video capture objects
        self.vid1.__del__()
        self.vid2.__del__()
        self.vid3.__del__()



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
    WebcamRecorder(tk.Tk(), "Tkinter Video Recorder", save_folder)

if __name__ == "__main__":
    main()
