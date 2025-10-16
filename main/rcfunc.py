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
        self.struct = {'Deadlift': 5, 'Benchpress': 3, 'Squat': 6}
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
        self.POSE_CONNECTIONS_CUSTOM = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Upper body joints
            (11, 23), (12, 24), (23, 24),  # Torso connections
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
        self.head_skeleton_txt_file = str
        
        self.stop_event = threading.Event()
        
        self.shared_state = {                         # 跨執行緒共享旗標                   # 初始化共享旗標
            "auto_recording_sig": False,              # 由 UI 控制是否允許錄影              # 初始 False
            "body_detected": False,                   # 由 body loop 更新是否有人體           # 初始 False
            "bar_y_changed": False,                   # 由 bar loop 更新槓是否有位移          # 初始 False
            "prev_bar_y": None                        # 由 bar loop 記憶上一幀 y             # 初始 None
        }                                             #                                     # —
        self.shared_lock = threading.Lock()           # 保護 shared_state 的互斥鎖            # 鎖
        self.BAR_MOVE_THRESH = 3.0                    # 槓 y 位移閾值（像素）                 # 可依影像尺寸調整

        
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
    
    def auto_recording_btn_clicked(self, sport, data_btn, source_btn, back_btn):                  # AUTO錄影按鈕事件
        if sport != 'Benchpress':                                                                  # 僅限 Benchpress
            return                                                                                # 其他運動不處理 
        with self.shared_lock:                                                                     # 進入臨界區  
            cur = self.shared_state.get("auto_recording_sig", False)                               # 讀目前自動錄影旗標 
            self.shared_state["auto_recording_sig"] = (not cur)                                    # 反轉旗標  
        # 你也可以在這裡更新 UI 樣式或文字提示，例如：data_btn.setText(...) 等                          # UI 提示  

    def source_get(self, sport):
        # 讀取來源順序與啟用設定（-1 代表停用）                       # 功能說明
        self.vision_src = {}                                            # 重置來源映射
        max_slots = self.struct[sport]                                  # 該運動最大插槽數
        try:
            with open('./config/click_order.json', mode='r', encoding='utf-8') as file:  # 修正為 'r'
                data = json.load(file)                                  # 載入JSON
                raw_list = data.get(sport, [])                          # 取對應運動的清單
        except Exception as e:
            print(f"[WARN] read click_order.json failed: {e}")          # 讀檔失敗警告
            raw_list = list(range(max_slots))                           # 退回預設 0..N-1

        # 規範化：長度對齊最大插槽數，不足以 -1 補齊                     # 對齊長度
        if len(raw_list) < max_slots:
            raw_list = raw_list + [-1] * (max_slots - len(raw_list))    # 不足補 -1
        else:
            raw_list = raw_list[:max_slots]                             # 超過則截斷

        # 過濾出實際啟用的來源（非 -1 才算）                              # 建立啟用順序
        self.active_sources = [src for src in raw_list if isinstance(src, int) and src >= 0]  # 實際要開的來源
        # 也保留「插槽到來源」的可讀映射（Vision1..N -> src or -1）         # 除錯觀察
        for i in range(max_slots):
            self.vision_src[f'Vision{i+1}'] = raw_list[i]               # 保留原始設定（可能為 -1）
        print(f"[INFO] enabled sources: {self.active_sources}")         # 列出將啟用的來源

        
    def initialize_cameras(self):
        cameras = []                                                    # 實際要用的相機容器
        if not hasattr(self, 'active_sources') or len(self.active_sources) == 0:  # 若沒有任何啟用
            print("[WARN] No active sources configured.")               # 警告
            return cameras                                              # 回傳空清單

        for idx, src in enumerate(self.active_sources):                 # 逐一嘗試開啟
            try:
                print(f'opening cam slot {idx} -> src {src}')           # 除錯訊息
                cam = MyVideoCapture(src)                               # 開相機
                if cam.isOpened():                                      # 檢查狀態
                    cameras.append(cam)                                 # 加入啟用清單
                else:
                    print(f"[WARN] Camera {src} is not available.")     # 無法開啟警告
            except Exception as e:
                print(f"[ERROR] Error opening camera {src}: {e}")       # 例外處理

        if not cameras:                                                 # 若完全沒開到
            print("[ERROR] No cameras connected or all disabled.")      # 錯誤訊息
        return cameras                                                  # 回傳「實際啟用」的相機清單

        
    def creat_threads(self, sport, labels):
        # 用「實際啟用的相機數」來建立 Barrier 與 Threads                  # 核心修正
        self.threads = []                                               # 重置執行緒清單
        if self.barrier:                                                # 若舊 barrier 存在
            try:
                self.barrier.abort()                                    # 中止舊 barrier（避免阻塞）
            except:                                                     # 可能已經 broken
                pass
        self.stop_event.clear()                                         # 清除停止旗標

        active_n = len(self.cameras)                                    # 實際啟用數
        if active_n == 0:                                               # 若沒有相機
            print("[ERROR] No active cameras; skip thread creation.")   # 錯誤訊息
            return                                                      # 直接返回

        self.barrier = threading.Barrier(active_n)                      # ★ 以啟用數建立 Barrier

        # labels 也要依啟用數裁剪（缺的就給預設）                           # 保護 labels
        safe_labels = (labels or [])[:active_n]                         # 取前 active_n 個
        if len(safe_labels) < active_n:                                 # 若不足
            safe_labels += [f"Cam{j}" for j in range(len(safe_labels), active_n)]  # 補上預設

        for i in range(active_n):                                       # 以 0..active_n-1 迭代
            thread = threading.Thread(                                  # 建立執行緒
                target=self.process_vision,                             # 跑影像流程
                args=(i, sport, safe_labels[i], self.barrier),          # 傳入序號/運動/標籤/Barrier
                daemon=True                                             # 設為daemon
            )
            self.threads.append(thread)                                 # 收集
            thread.start()                                              # 啟動

            
    def model_select(self, sport):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if sport == 'Deadlift':
            bar_model = YOLO("./model/deadlift/yolo_bar_model/best.pt")
            bone_model = YOLO("./model/deadlift/yolov8_model/yolov8n-pose.pt")
            bar_model.to(device)
            bone_model.to(device)
            return [bar_model, bone_model]
        
        elif sport == 'Squat':
            bar_model = YOLO("./model/squat/yolo_bar_model/best.pt")
            bone_model = YOLO("./model/squat/yolov8_model/yolov8n-pose.pt")
            bar_model.to(device)
            bone_model.to(device)
            return [bar_model, bone_model]
        
        elif sport =='Benchpress':
            bar_model = YOLO("./model/benchpress/yolo_bar_model/best.pt")
            body_model = YOLO("./model/benchpress/body_model/top11n.pt")
            head_model = YOLO("./model/benchpress/head_model/yolo11n.pt")
            bar_model.to(device)
            body_model.to(device)
            head_model.to(device)
            return [bar_model, body_model, head_model]
    
    def process_vision(self, i, sport, label, barrier):
        start_time = time.time()                                        # 起始時間（供 FPS/命名用）
        frame_count = 0                                                 # 幀計數
        frame_count_for_detect = 0                                      # 偵測節流幀計數
        fps = 0                                                         # 每秒幀數
        out = None                                                      # 疊圖輸出 writer（各 loop 內部建立/回傳）
        original_out = None                                             # 原始畫面 writer（需要雙軌時使用）
        txt_file = None                                                 # 文字紀錄檔 handle 或路徑（交由各 loop 管）

        while not self.stop_event.is_set():                             # 主迴圈直到收到停止事件
            cap = self.cameras[i]                                       # 取用第 i 支「啟用中的」相機
            ret, frame = cap.get_frame()                                # 讀一幀影像
            if not ret:                                                 # 取幀失敗（USB 抖動/暫時無幀）
                continue                                                # 跳過本輪避免 thread 中斷

            # ========================= Deadlift =========================
            if sport == 'Deadlift':                                     # 硬舉模式
                if i == 0:                                              # 啟用序 0 → 槓視角（bar）
                    start_time, frame_count, fps, out, frame_count_for_detect, self.save_sig_1, txt_file = loop.deadlift_bar_loop(
                        i, frame, label, self.save_sig_1,               # 相機序/畫面/標籤/存檔旗標
                        self.recording_sig,                             # UI 錄影 Gate
                        self.folder, start_time, frame_count, fps, out, # I/O 與計時/輸出 writer
                        self.models[0],                                 # bar 模型固定用 models[0]
                        txt_file, frame_count_for_detect, barrier       # txt/偵測幀/多機同步
                    )
                elif i == 1:                                            # 啟用序 1 → 骨架視角（bone）
                    start_time, frame_count, fps, out, frame_count_for_detect, self.save_sig_2, txt_file = loop.deadlift_bone_loop(
                        i, frame, label, self.save_sig_2,               # 旗標
                        self.recording_sig,                             # Gate
                        self.folder, start_time, frame_count, fps, out, # I/O
                        self.models[1],                                 # bone 模型固定用 models[1]
                        txt_file, frame_count_for_detect,               # txt/偵測幀
                        self.skeleton_connections, barrier              # 骨架連線/同步
                    )
                else:                                                   # 其餘 → 一般錄影
                    start_time, frame_count, fps, out, self.save_sig_3 = loop.deadlift_general_loop(
                        i, frame, label, self.save_sig_3,               # 旗標
                        self.recording_sig,                             # Gate
                        self.folder, start_time, frame_count, fps, out, # I/O
                        barrier                                         # 同步
                    )

            # ======================== Benchpress ========================
            elif sport == 'Benchpress':                                 # 臥推模式
                if i == 0:                                              # 啟用序 0 → 槓視角（bar）：更新 bar_y_changed + Gate 錄影
                    start_time, frame_count, fps, out, frame_count_for_detect, original_out, self.save_sig_1, txt_file = loop.benchpress_bar_loop(
                        i, frame, label, self.save_sig_1,               # 旗標
                        self.folder, start_time, frame_count, fps,      # I/O 與計時
                        out, original_out,                              # 疊圖/原始 writer
                        self.models[0],                                 # bar 模型固定用 models[0]
                        txt_file, frame_count_for_detect, barrier,      # txt/偵測幀/同步
                        self.shared_state, self.shared_lock,            # 共享狀態（更新 bar_y_changed）
                        self.BAR_MOVE_THRESH                            # 槓 y 位移閾值（像素）
                    )
                elif i == 1:                                            # 啟用序 1 → 人體視角（body）：更新 body_detected + Gate 錄影
                    start_time, frame_count, fps, out, frame_count_for_detect, self.save_sig_2, txt_file = loop.benchpress_body_loop(
                        i, frame, label, self.save_sig_2,               # 旗標
                        self.folder, start_time, frame_count, fps, out, # I/O 與計時
                        self.models[1],                                 # body 模型固定用 models[1]
                        txt_file, frame_count_for_detect,               # txt/偵測幀
                        None,                                           # skeleton_connections（給 None 走預設）
                        barrier,                                        # 同步
                        self.shared_state, self.shared_lock             # 共享狀態（更新 body_detected）
                    )
                elif i == 2:                                            # 啟用序 2 → 頭部視角（head）：不需要 model
                    start_time, frame_count, fps, out, original_out, self.save_sig_3, frame_count_for_detect = loop.benchpress_head_loop(
                        i, frame, label, self.save_sig_3,               # 與原介面一致：save_sig
                        self.folder, start_time, frame_count, fps,      # I/O 與計時
                        out, original_out,                              # 疊圖/原始 writer
                        frame_count_for_detect, barrier,                # 偵測幀/同步（★ 不傳 model）
                        self.shared_state, self.shared_lock             # 共享狀態（如需 Gate 條件）
                    )
                else:                                                   # 多於三路 → 一般錄影（或可改成 benchpress_general_loop）
                    start_time, frame_count, fps, out, self.save_sig_3 = loop.deadlift_general_loop(
                        i, frame, label, self.save_sig_3,               # 旗標
                        self.recording_sig,                             # Gate
                        self.folder, start_time, frame_count, fps, out, # I/O
                        barrier                                         # 同步
                    )

            # ========================== Squat ==========================
            elif sport == 'Squat':                                      # 深蹲模式
                if i == 0:                                              # 啟用序 0 → 槓視角（bar，含 original_out）
                    start_time, frame_count, fps, out, original_out, frame_count_for_detect, self.save_sig_1, txt_file = loop.squat_bar_loop(
                        i, frame, label, self.save_sig_1,               # 旗標
                        self.recording_sig,                             # Gate
                        self.folder, start_time, frame_count, fps,      # I/O
                        out, original_out,                              # 疊圖/原始 writer
                        self.models[0],                                 # bar 模型固定用 models[0]
                        txt_file, frame_count_for_detect, barrier       # txt/偵測幀/同步
                    )
                elif i == 1:                                            # 啟用序 1 → 骨架視角（bone，含 original_out）
                    start_time, frame_count, fps, out, original_out, frame_count_for_detect, self.save_sig_2, txt_file = loop.squat_bone_loop(
                        i, frame, label, self.save_sig_2,               # 旗標
                        self.recording_sig,                             # Gate
                        self.folder, start_time, frame_count, fps,      # I/O
                        out, original_out,                              # 疊圖/原始 writer
                        self.models[1],                                 # bone 模型固定用 models[1]
                        txt_file, frame_count_for_detect,               # txt/偵測幀
                        self.skeleton_connections, barrier              # 骨架連線/同步
                    )
                else:                                                   # 其餘 → 一般錄影
                    start_time, frame_count, fps, out, self.save_sig_3 = loop.squat_general_loop(
                        i, frame, label, self.save_sig_3,               # 旗標
                        self.recording_sig,                             # Gate
                        self.folder, start_time, frame_count, fps, out, # I/O
                        barrier                                         # 同步
                    )

            # （需要時可在此擴充其他運動類型）                             # 擴充點

        cap.__del__()                                                   # 離開主迴圈時釋放相機


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

    def recording_ctrl_btn_clicked(self, sport, button_1, button_2, button_3):
        if not self.recording_sig:
            self.start_recording(sport)
            button_1.setEnabled(False)
            button_2.setEnabled(False)
            button_3.setEnabled(False)
        else:
            self.stop_recording()
            button_1.setEnabled(True)
            button_2.setEnabled(True)
            button_3.setEnabled(True)
            
    def player_reset(self, name):
        self.player = name.text()

    def start_recording(self, sport):
        self.stop_event.clear()  # Clear the stop event before starting threads
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        if self.player:
            self.folder = os.path.join(self.save_path[sport], f"recording_{timestamp}_{self.player}")
        else:
            self.folder = os.path.join(self.save_path[sport], f"recording_{timestamp}")
        os.makedirs(self.folder, exist_ok=True)
        self.out_1 = None  # 確保 `out` 變數重置
        self.out_2 = None
        self.out_3 = None
        
        self.recording_sig = True
        with self.shared_lock:                                                        # 進入臨界區
            self.shared_state["recording_sig"] = True                                 # UI 開啟錄影 Gate
            self.shared_state["body_detected"] = False                                # 重置：開錄時重新蒐集
            self.shared_state["bar_y_changed"] = False                                # 重置：等待槓位移觸發
            self.shared_state["prev_bar_y"] = None                                    # 重置：上一幀 y 清空

        print("Recording started")
            
    def stop_recording(self):                                                                # 停止錄影                    # 函式：停止錄影
        if self.recording_sig:                                                                # 若目前在錄影                 # 判斷是否錄影中
            with self.shared_lock:                                                            # 進入臨界區                   # 加鎖保護
                self.shared_state["recording_sig"] = False                                    # 關閉 UI Gate                # 關閉錄影門檻
                # 其他旗標保留最近狀態即可（可選清空）                                           # 可選清空
            self.recording_sig = False                                                        # 關閉舊布林（沿用）           # 關閉舊旗標
            self.save_sig_1 = True                                                            # 允許 loop 收尾（可保留）     # 觸發保存收尾
            self.save_sig_2 = True                                                            # 同上                         # 觸發保存收尾
            self.save_sig_3 = True                                                            # 同上                         # 觸發保存收尾

        
    def data_produce_btn_clicked(self, sport):
        # self.folder = 'C:/Users/92A27/MOCAP/recordings/recording_20250324_145044_BVT'
        if sport == 'Deadlift':
            # 對槓端及骨架做內插
            os.system(f'python ./tools/Deadlift_tool/interpolate.py {self.folder}')
            # bar
            os.system(f'python ./tools/Benchpress_tool/bar_data_produce.py {self.folder} --out ./config --sport deadlift')
            # angle
            os.system(f'python ./tools/Deadlift_tool/data_produce.py {self.folder} --out ./config')
            # split data
            os.system(f'python ./tools/Deadlift_tool/data_split.py {self.folder}')
            # modle predict
            os.system(f'python ./tools/Deadlift_tool/predict.py {self.folder} --out ./config')
            
        if sport == 'Benchpress':
            # 
            os.system(f'python ./tools/Benchpress_tool/step0_hampel_bar.py {self.folder}')
            # 
            os.system(f'python ./tools/Benchpress_tool/step0_hampel_yolo_ske_rear.py {self.folder}')
            #
            os.system(f'python ./tools/Benchpress_tool/step0_hampel_yolo_ske_top.py {self.folder} ')
            # 
            os.system(f'python ./tools/Benchpress_tool/step1_interpolate_bar.py {self.folder}')
            # 
            os.system(f'python ./tools/Benchpress_tool/step2_interpolate_yolo_ske.py {self.folder}')
            #
            os.system(f'python ./tools/Benchpress_tool/step3_autocutting_0801.py {self.folder}')
            # 
            os.system(f'python ./tools/Benchpress_tool/step5_calculate_angle_new_feature_test.py {self.folder}')
            # 
            os.system(f'python ./tools/Benchpress_tool/step6_cut.py {self.folder} ')
            # 
            os.system(f'python ./tools/Benchpress_tool/step7_length_100.py {self.folder}')
            # 
            os.system(f'python ./tools/Benchpress_tool/step8_normalize.py {self.folder}')
        
        if sport == 'Squat':
            pass
            # # 對槓端及骨架做內插
            # os.system(f'python ./tools/Deadlift_tool/interpolate.py {self.folder}')
            # # bar
            # os.system(f'python ./tools/Benchpress_tool/bar_data_produce.py {self.folder} --out ./config --sport deadlift')
            # # angle
            # os.system(f'python ./tools/Deadlift_tool/data_produce.py {self.folder} --out ./config')
            # # split data
            # os.system(f'python ./tools/Deadlift_tool/data_split.py {self.folder}')
            # # modle predict
            # os.system(f'python ./tools/Deadlift_tool/predict.py {self.folder} --out ./config')
            
        # 後製軌跡影片
        os.system(f'python ./tools/trajectory.py {self.folder}')
        print('後製已完成')