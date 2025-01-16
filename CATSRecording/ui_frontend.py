from ui import Ui_MainWindow
from rcfunc import Recordingbackend
from rpfunc import Replaybackend
from PyQt5 import QtCore, QtGui, QtWidgets
from qt_material import apply_stylesheet
import os, glob, sys
from PyQt5.QtCore import QTimer

# frontend logic
class Mainwindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(Mainwindow, self).__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.rcbf = Recordingbackend()
        self.rpbf = Replaybackend()
        self.ui.setupUi(self)
        
        self.ui.rc_Squat_btn.setEnabled(False)
        self.icons = []
        self.isclicked = False
        icon_srcs = glob.glob(self.resource_path('ui_src/*.png'))
        for icon in icon_srcs:
            self.icons.append(QtGui.QIcon(icon))
        self.ui.Play_btn.setIcon(self.icons[1])
        self.ui.Stop_btn.setIcon(self.icons[3])

        self.ui.rc_Deadlift_btn.clicked.connect(self.rc_Deadlift_clicked)
        self.ui.rc_Benchpress_btn.clicked.connect(self.rc_Benchpress_clicked)
        
        # Initialize QTimer for updating frames
        # self.timer = QTimer(self)
        # # self.timer.timeout.connect(self.update_frames)
        # self.timer.start(30)  # Update frames every 30ms
        
        self.Vision_labels = []  # Store QLabel for camera frames
        
        # replay top ctrl connection
        self.ui.rp_Deadlift_btn.clicked.connect(lambda: self.rpbf.Deadlift_btn_pressed(
            self.ui.rp_Deadlift_btn, self.ui.rp_Benchpress_btn, self.ui.rp_Squat_btn, self.ui.Play_btn, self.icons, self.ui.Stop_btn, 
            self.ui.Frameslider, self.ui.fast_forward_combobox, self.ui.File_comboBox, self.ui.Replay_tab, self.ui.play_layout))
        self.ui.rp_Benchpress_btn.clicked.connect(lambda: self.rpbf.Benchpress_btn_pressed(
            self.ui.rp_Deadlift_btn, self.ui.rp_Benchpress_btn, self.ui.rp_Squat_btn, self.ui.Play_btn, self.icons, self.ui.Stop_btn, 
            self.ui.Frameslider, self.ui.fast_forward_combobox, self.ui.File_comboBox, self.ui.Replay_tab, self.ui.play_layout))
        self.ui.rp_Squat_btn.clicked.connect(lambda: self.rpbf.Squat_btn_pressed(
            self.ui.rp_Deadlift_btn, self.ui.rp_Benchpress_btn, self.ui.rp_Squat_btn, self.ui.Play_btn, self.icons, self.ui.Stop_btn, 
            self.ui.Frameslider, self.ui.fast_forward_combobox, self.ui.File_comboBox, self.ui.Replay_tab, self.ui.play_layout))
        self.ui.File_comboBox.currentTextChanged.connect(lambda: self.rpbf.File_combobox_TextChanged(
            self.ui.File_comboBox, self.ui.Play_btn, self.icons, self.ui.Frameslider))
        self.ui.Stop_btn.clicked.connect(lambda: self.rpbf.stop(self.ui.Frameslider, self.ui.Play_btn, self.icons))
        
        # replay bottom ctrl connection
        rates = [1, 1.5, 0.8, 0.5]
        for rate in rates:
            self.ui.fast_forward_combobox.addItems([str(rate)])
        self.ui.Play_btn.clicked.connect(lambda: self.rpbf.play_btn_clicked(
            self.ui.fast_forward_combobox, self.ui.Play_btn, self.icons, self.ui.Frameslider))
        self.ui.Frameslider.valueChanged.connect(lambda: self.rpbf.sliding(self.ui.Frameslider, self.ui.TimeCount_LineEdit))
        self.ui.Frameslider.sliderPressed.connect(self.rpbf.slider_Pressed)
        self.ui.Frameslider.sliderReleased.connect(self.rpbf.slider_released)
        self.ui.Frameslider.valueChanged.connect(lambda: self.rpbf.slider_changed(self.ui.Frameslider, self.ui.Play_btn, self.icons))

    def rc_Deadlift_clicked(self):
        self.Deadlift_layout_set()
        self.rcbf.creat_threads('Deadlift', self.rc_Vision_labels)

    def update_frames(self):
        # Update the camera frames every 30ms
        for i, label in enumerate(self.Vision_labels):
            frame = self.rcbf.get_frame(i)
            if frame is not None:
                height, width, channels = frame.shape
                bytes_per_line = channels * width
                q_image = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_BGR888)
                pixmap = QtGui.QPixmap.fromImage(q_image)
                label.setPixmap(pixmap)

    def rc_Benchpress_clicked(self):
        self.Benchpress_layout_set()
        self.rcbf.creat_threads('Benchpress', self.rc_Vision_labels)

    def back_toolbtn_clicked(self):
        # 清空 recording_layout
        self.rpbf.clear_layout(self.ui.recording_layout)
        # 重新添加原本的控件
        self.add_original_recording_tab_content()

    def add_original_recording_tab_content(self):
        # 恢復錄製按鈕的原本佈局
        grid_layout = self.ui.grid_Layout_recording

        # 添加手動錄製選項
        self.ui.manual_checkbox = QtWidgets.QCheckBox(self.ui.Recording_tab)
        self.ui.manual_checkbox.setObjectName("manual_checkbox")
        self.ui.manual_checkbox.setText("manual recording")
        # self.isclicked = self.ui.manual_checkbox.stateChanged.connect(self.rcbf.manual_checkbox_isclicked)
        grid_layout.addWidget(self.ui.manual_checkbox, 0, 0, 1, 1)

        # 添加 Deadlift 按鈕
        self.ui.rc_Deadlift_btn = QtWidgets.QPushButton(self.ui.Recording_tab)
        self.ui.rc_Deadlift_btn.setFont(QtGui.QFont("Times New Roman", 26))
        self.ui.rc_Deadlift_btn.setText("Deadlift")
        self.ui.rc_Deadlift_btn.setObjectName("rc_Deadlift_btn")
        self.ui.rc_Deadlift_btn.clicked.connect(self.rc_Deadlift_clicked)
        grid_layout.addWidget(self.ui.rc_Deadlift_btn, 1, 0, 1, 1)

        # 添加 Benchpress 按鈕
        self.ui.rc_Benchpress_btn = QtWidgets.QPushButton(self.ui.Recording_tab)
        self.ui.rc_Benchpress_btn.setFont(QtGui.QFont("Times New Roman", 26))
        self.ui.rc_Benchpress_btn.setText("Benchpress")
        self.ui.rc_Benchpress_btn.setObjectName("rc_Benchpress_btn")
        self.ui.rc_Benchpress_btn.clicked.connect(self.rc_Benchpress_clicked)
        grid_layout.addWidget(self.ui.rc_Benchpress_btn, 2, 0, 1, 1)

        # 添加 Squat 按鈕
        self.ui.rc_Squat_btn = QtWidgets.QPushButton(self.ui.Recording_tab)
        self.ui.rc_Squat_btn.setFont(QtGui.QFont("Times New Roman", 26))
        self.ui.rc_Squat_btn.setText("Squat")
        self.ui.rc_Squat_btn.setObjectName("rc_Squat_btn")
        self.ui.rc_Squat_btn.setEnabled(False)
        grid_layout.addWidget(self.ui.rc_Squat_btn, 3, 0, 1, 1)

        # 確保佈局刷新
        self.ui.recording_layout.addLayout(grid_layout)

    def Deadlift_layout_set(self):
        # clear recording layout
        grid_layout = self.ui.grid_Layout_recording
        for i in reversed(range(grid_layout.count())):
            widget = grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # set recording layout
        self.ctrl_layout = QtWidgets.QHBoxLayout()
        self.ctrl_layout.setContentsMargins(0, 0, 0, 0)
        self.ctrl_layout.setSpacing(1800)
        self.ui.recording_layout.addLayout(self.ctrl_layout)

        self.recording_ctrl_btn = QtWidgets.QToolButton(self.ui.Recording_tab)
        self.recording_ctrl_btn.setIcon(self.icons[2])
        self.recording_ctrl_btn.setIconSize(QtCore.QSize(64, 64))
        self.ctrl_layout.addWidget(self.recording_ctrl_btn)

        self.back_toolbtn = QtWidgets.QToolButton(self.ui.Recording_tab)
        self.back_toolbtn.setIcon(self.icons[4])
        self.back_toolbtn.setIconSize(QtCore.QSize(64, 64))
        self.back_toolbtn.clicked.connect(self.back_toolbtn_clicked)
        self.ctrl_layout.addWidget(self.back_toolbtn)

        self.Deadlift_vision_layout = QtWidgets.QHBoxLayout()
        self.ui.recording_layout.addLayout(self.Deadlift_vision_layout)

        labelsize = [420, 560]
        self.rc_Vision_labels, self.rc_qpixmaps = self.rpbf.creat_vision_labels_pixmaps(labelsize, self.ui.Recording_tab, self.Deadlift_vision_layout, 5)
        self.recording_ctrl_btn.clicked.connect(lambda: self.rcbf.recording_ctrl_btn_clicked(self.Vision_labels))
        
    def Benchpress_layout_set(self):
        # clear recording layout
        grid_layout = self.ui.grid_Layout_recording
        for i in reversed(range(grid_layout.count())):
            widget = grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # set recording layout
        self.ctrl_layout = QtWidgets.QHBoxLayout()
        self.ctrl_layout.setContentsMargins(0, 0, 0, 0)
        self.ctrl_layout.setSpacing(1800)
        self.ui.recording_layout.addLayout(self.ctrl_layout)

        self.recording_ctrl_btn = QtWidgets.QToolButton(self.ui.Recording_tab)
        self.recording_ctrl_btn.setIcon(self.icons[2])
        self.recording_ctrl_btn.setIconSize(QtCore.QSize(64, 64))
        self.ctrl_layout.addWidget(self.recording_ctrl_btn)

        self.back_toolbtn = QtWidgets.QToolButton(self.ui.Recording_tab)
        self.back_toolbtn.setIcon(self.icons[4])
        self.back_toolbtn.setIconSize(QtCore.QSize(64, 64))
        self.back_toolbtn.clicked.connect(self.back_toolbtn_clicked)
        self.ctrl_layout.addWidget(self.back_toolbtn)

        self.Benchpress_vision_layout = QtWidgets.QHBoxLayout()
        self.ui.recording_layout.addLayout(self.Benchpress_vision_layout)

        labelsize = [640, 480]
        self.rc_Vision_labels, self.rc_qpixmaps = self.rpbf.creat_vision_labels_pixmaps(labelsize, self.ui.Recording_tab, self.Benchpress_vision_layout, 3)
        self.recording_ctrl_btn.clicked.connect(lambda: self.rcbf.recording_ctrl_btn_clicked(self.rc_Vision_labels))


    def resource_path(self, relative_path):
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        return os.path.join(base_path, relative_path)

