from ui import Ui_MainWindow
from rcfunc import Recordingbackend
from rpfunc import Replaybackend
from PyQt5 import QtCore, QtGui, QtWidgets
import os, glob, sys

# frontend logic
class Mainwindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(Mainwindow, self).__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.rcbf = Recordingbackend()
        self.rpbf = Replaybackend()
        self.ui.setupUi(self)
        
        #self.ui.rc_Squat_btn.setEnabled(True)
        self.icons = []
        self.names = []
        self.graghs = []
        self.player_btn = []
        self.data_layouts = []
        self.D_layout_inited = False
        self.B_layout_inited = False
        self.S_layout_inited = False
        
        icon_srcs = glob.glob('./ui_src/*.png')
        for icon in icon_srcs:
            self.icons.append(QtGui.QIcon(icon))
        self.ui.Play_btn.setIcon(self.icons[1])
        self.ui.Stop_btn.setIcon(self.icons[3])

        self.ui.tabs.currentChanged.connect(self.tab_changed)
        self.ui.rc_Deadlift_btn.clicked.connect(self.rc_Deadlift_clicked)
        self.ui.rc_Benchpress_btn.clicked.connect(self.rc_Benchpress_clicked)
        self.ui.rc_Squat_btn.clicked.connect(self.rc_Squat_clicked)
        self.ui.rp_Deadlift_btn.clicked.connect(lambda :self.rp_layout_set('Deadlift'))
        self.ui.rp_Benchpress_btn.clicked.connect(lambda :self.rp_layout_set('Benchpress'))
        self.ui.rp_Squat_btn.clicked.connect(lambda :self.rp_layout_set('Squat'))
        
        self.Vision_labels = []  # Store QLabel for camera frames
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
        self.ui.search_LineEdit.textChanged.connect(lambda: self.rpbf.search_text_changed(self.ui.File_comboBox, self.ui.search_LineEdit.text()))

    def tab_changed(self, index):
        if index == 0:
            self.layout_clear(self.ui.head_vis_layout)
            self.layout_clear(self.ui.bottom_vis_layout)
            self.layout_clear(self.ui.data_ctrl_layout_V)
            if self.data_layouts:
                for layout in self.data_layouts:
                    self.layout_clear(layout)
            if self.ui.bottom_controls_layout in [self.ui.data_ctrl_layout_V.itemAt(i).layout() for i in range(self.ui.data_ctrl_layout_V.count())]:
                self.ui.data_ctrl_layout_V.removeItem(self.ui.bottom_controls_layout)
            self.data_layouts = []
            self.rpbf.tab_changed()

    def rc_Deadlift_clicked(self):
        self.names.clear()
        self.rc_Deadlift_layout_set()
        self.rcbf.init_rc_backend('Deadlift', self.rc_Vision_labels)

    def rc_Squat_clicked(self):
        self.names.clear()
        self.rc_Squat_layout_set()
        self.rcbf.init_rc_backend('Squat', self.rc_Vision_labels)

    def rc_Benchpress_clicked(self):
        self.names.clear()
        self.rc_Benchpress_layout_set()
        self.rcbf.init_rc_backend('Benchpress', self.rc_Vision_labels)

    def back_toolbtn_clicked(self):
        # 清空 recording_layout
        self.rpbf.clear_layout(self.ui.recording_layout)
        self.rcbf.stop_event.set()
        # 重新添加原本的控件
        self.add_original_recording_tab_content()

    def add_original_recording_tab_content(self):
        # 恢復錄製按鈕的原本佈局
        grid_layout = self.ui.grid_Layout_recording

        # 添加手動錄製選項
        self.ui.manual_checkbox = QtWidgets.QCheckBox(self.ui.Recording_tab)
        self.ui.manual_checkbox.setObjectName("manual_checkbox")
        self.ui.manual_checkbox.setText("manual recording")
        self.ui.manual_checkbox.setChecked(True)
        self.ui.manual_checkbox.setDisabled(True)
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
        self.ui.rc_Squat_btn.clicked.connect(self.rc_Squat_clicked)
        grid_layout.addWidget(self.ui.rc_Squat_btn, 3, 0, 1, 1)

        # 確保佈局刷新
        self.ui.recording_layout.addLayout(grid_layout)

    def rc_Deadlift_layout_set(self):
        # clear recording layout
        grid_layout = self.ui.grid_Layout_recording
        for i in reversed(range(grid_layout.count())):
            widget = grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # set recording layout
        self.ctrl_layout = QtWidgets.QHBoxLayout()
        self.ctrl_layout.setContentsMargins(0, 0, 0, 0)
        self.ctrl_layout.setSpacing(400)
        self.ui.recording_layout.addLayout(self.ctrl_layout)

        self.recording_ctrl_btn = QtWidgets.QToolButton(self.ui.Recording_tab)
        self.recording_ctrl_btn.setIcon(self.icons[2])
        self.recording_ctrl_btn.setIconSize(QtCore.QSize(128, 128))
        self.ctrl_layout.addWidget(self.recording_ctrl_btn)

        self.auto_recording_btn = QtWidgets.QPushButton(self.ui.Recording_tab)
        self.auto_recording_btn.setFont(QtGui.QFont("Times New Roman", 64))
        self.auto_recording_btn.setText("Auto Recording")
        self.auto_recording_btn.setEnabled(False)
        self.ctrl_layout.addWidget(self.auto_recording_btn)
        
        self.data_produce_btn = QtWidgets.QPushButton(self.ui.Recording_tab)
        self.data_produce_btn.setFont(QtGui.QFont("Times New Roman", 64))
        self.data_produce_btn.setText("Data Produce")
        self.ctrl_layout.addWidget(self.data_produce_btn)
        
        self.source_ctrl_btn = QtWidgets.QPushButton(self.ui.Recording_tab)
        self.source_ctrl_btn.setFont(QtGui.QFont("Times New Roman", 64))
        self.source_ctrl_btn.setText("Source change")
        self.ctrl_layout.addWidget(self.source_ctrl_btn)

        self.back_toolbtn = QtWidgets.QToolButton(self.ui.Recording_tab)
        self.back_toolbtn.setIcon(self.icons[4])
        self.back_toolbtn.setIconSize(QtCore.QSize(128, 128))
        self.back_toolbtn.clicked.connect(self.back_toolbtn_clicked)
        self.ctrl_layout.addWidget(self.back_toolbtn)

        self.Deadlift_vision_layout = QtWidgets.QHBoxLayout()
        self.ui.recording_layout.addLayout(self.Deadlift_vision_layout)

        self.subject_layout = QtWidgets.QGridLayout()
        self.subject_layout.setContentsMargins(0, 0, 0, 0)
        for x in range(8):
            for y in range(2):
                if y == 0:
                    text = QtWidgets.QLineEdit()
                    text.setFocus(True)
                    text.setAlignment(QtCore.Qt.AlignCenter)
                    text.setText(f'Name {x+1}')
                    text.setStyleSheet("font-size:20px; color:yellow;")
                    self.names.append(text)
                    self.subject_layout.addWidget(text, y, x)    
                if y == 1:
                    btn = QtWidgets.QPushButton(self.ui.Recording_tab)
                    btn.setFont(QtGui.QFont('Times New Roman', 32))
                    btn.setText(f'Player {x+1}')
                    btn.clicked.connect(lambda checked, i=x: self.rcbf.player_reset(self.names[i]))
                    self.player_btn.append(btn)
                    self.subject_layout.addWidget(btn, y, x)
                    
        self.ui.recording_layout.addLayout(self.subject_layout)

        labelsize = [480, 640]
        self.rc_Vision_labels, self.rc_qpixmaps = self.rpbf.creat_vision_labels_pixmaps([x * 1.2 for x in labelsize], self.ui.Recording_tab, self.Deadlift_vision_layout, 'Deadlift', 5)
        self.data_produce_btn.clicked.connect(lambda: self.rcbf.data_produce_btn_clicked('Deadlift'))
        self.source_ctrl_btn.clicked.connect(lambda: self.rcbf.source_ctrl_btn_clicked('Deadlift', self.rc_Vision_labels))
        self.recording_ctrl_btn.clicked.connect(lambda: self.rcbf.recording_ctrl_btn_clicked('Deadlift', self.data_produce_btn, self.source_ctrl_btn, self.back_toolbtn))
        
    def rc_Benchpress_layout_set(self):
        # clear recording layout
        grid_layout = self.ui.grid_Layout_recording
        for i in reversed(range(grid_layout.count())):
            widget = grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # set recording layout
        self.ctrl_layout = QtWidgets.QHBoxLayout()
        self.ctrl_layout.setContentsMargins(0, 0, 0, 0)
        self.ctrl_layout.setSpacing(400)
        self.ui.recording_layout.addLayout(self.ctrl_layout)

        self.recording_ctrl_btn = QtWidgets.QToolButton(self.ui.Recording_tab)
        self.recording_ctrl_btn.setIcon(self.icons[2])
        self.recording_ctrl_btn.setIconSize(QtCore.QSize(64, 64))
        self.ctrl_layout.addWidget(self.recording_ctrl_btn)
        
        self.auto_recording_btn = QtWidgets.QPushButton(self.ui.Recording_tab)
        self.auto_recording_btn.setFont(QtGui.QFont("Times New Roman", 32))
        self.auto_recording_btn.setText("Auto Recording")
        self.auto_recording_btn.setEnabled(False)
        self.ctrl_layout.addWidget(self.auto_recording_btn)
        
        self.data_produce_btn = QtWidgets.QPushButton(self.ui.Recording_tab)
        self.data_produce_btn.setFont(QtGui.QFont("Times New Roman", 32))
        self.data_produce_btn.setText("Data Produce")
        self.ctrl_layout.addWidget(self.data_produce_btn)
        
        self.source_ctrl_btn = QtWidgets.QPushButton(self.ui.Recording_tab)
        self.source_ctrl_btn.setFont(QtGui.QFont("Times New Roman", 32))
        self.source_ctrl_btn.setText("Source change")
        self.ctrl_layout.addWidget(self.source_ctrl_btn)

        self.back_toolbtn = QtWidgets.QToolButton(self.ui.Recording_tab)
        self.back_toolbtn.setIcon(self.icons[4])
        self.back_toolbtn.setIconSize(QtCore.QSize(64, 64))
        self.back_toolbtn.clicked.connect(self.back_toolbtn_clicked)
        self.ctrl_layout.addWidget(self.back_toolbtn)

        self.Benchpress_vision_layout = QtWidgets.QHBoxLayout()
        self.ui.recording_layout.addLayout(self.Benchpress_vision_layout)
        
        self.subject_layout = QtWidgets.QGridLayout()
        self.subject_layout.setContentsMargins(0, 0, 0, 0)
        for x in range(8):
            for y in range(2):
                if y == 0:
                    text = QtWidgets.QLineEdit()
                    text.setFocus(True)
                    text.setAlignment(QtCore.Qt.AlignCenter)
                    text.setText(f'Name {x+1}')
                    text.setStyleSheet("font-size:20px; color:yellow;")
                    self.names.append(text)
                    self.subject_layout.addWidget(text, y, x)    
                if y == 1:
                    btn = QtWidgets.QPushButton(self.ui.Recording_tab)
                    btn.setFont(QtGui.QFont('Times New Roman', 32))
                    btn.setText(f'Player {x+1}')
                    btn.clicked.connect(lambda checked, i=x: self.rcbf.player_reset(self.names[i]))
                    self.player_btn.append(btn)
                    self.subject_layout.addWidget(btn, y, x)
                    
        self.ui.recording_layout.addLayout(self.subject_layout)

        labelsize = [640, 480]
        self.rc_Vision_labels, self.rc_qpixmaps = self.rpbf.creat_vision_labels_pixmaps([x * 1.5 for x in labelsize], self.ui.Recording_tab, self.Benchpress_vision_layout, 'Benchpress', 3)
        self.data_produce_btn.clicked.connect(lambda: self.rcbf.data_produce_btn_clicked('Benchpress'))
        self.source_ctrl_btn.clicked.connect(lambda: self.rcbf.source_ctrl_btn_clicked('Benchpress', self.rc_Vision_labels))
        self.recording_ctrl_btn.clicked.connect(lambda: self.rcbf.recording_ctrl_btn_clicked('Benchpress', self.data_produce_btn, self.source_ctrl_btn, self.back_toolbtn))

    def rc_Squat_layout_set(self):
        # clear recording layout
        grid_layout = self.ui.grid_Layout_recording
        for i in reversed(range(grid_layout.count())):
            widget = grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # set recording layout
        self.ctrl_layout = QtWidgets.QHBoxLayout()
        self.ctrl_layout.setContentsMargins(0, 0, 0, 0)
        self.ctrl_layout.setSpacing(400)
        self.ui.recording_layout.addLayout(self.ctrl_layout)

        self.recording_ctrl_btn = QtWidgets.QToolButton(self.ui.Recording_tab)
        self.recording_ctrl_btn.setIcon(self.icons[2])
        self.recording_ctrl_btn.setIconSize(QtCore.QSize(128, 128))
        self.ctrl_layout.addWidget(self.recording_ctrl_btn)

        self.auto_recording_btn = QtWidgets.QPushButton(self.ui.Recording_tab)
        self.auto_recording_btn.setFont(QtGui.QFont("Times New Roman", 64))
        self.auto_recording_btn.setText("Auto Recording")
        self.auto_recording_btn.setEnabled(False)
        self.ctrl_layout.addWidget(self.auto_recording_btn)
        
        self.data_produce_btn = QtWidgets.QPushButton(self.ui.Recording_tab)
        self.data_produce_btn.setFont(QtGui.QFont("Times New Roman", 64))
        self.data_produce_btn.setText("Data Produce")
        self.ctrl_layout.addWidget(self.data_produce_btn)
        
        self.source_ctrl_btn = QtWidgets.QPushButton(self.ui.Recording_tab)
        self.source_ctrl_btn.setFont(QtGui.QFont("Times New Roman", 64))
        self.source_ctrl_btn.setText("Source change")
        self.ctrl_layout.addWidget(self.source_ctrl_btn)

        self.back_toolbtn = QtWidgets.QToolButton(self.ui.Recording_tab)
        self.back_toolbtn.setIcon(self.icons[4])
        self.back_toolbtn.setIconSize(QtCore.QSize(128, 128))
        self.back_toolbtn.clicked.connect(self.back_toolbtn_clicked)
        self.ctrl_layout.addWidget(self.back_toolbtn)

        self.Deadlift_vision_layout = QtWidgets.QHBoxLayout()
        self.ui.recording_layout.addLayout(self.Deadlift_vision_layout)

        self.subject_layout = QtWidgets.QGridLayout()
        self.subject_layout.setContentsMargins(0, 0, 0, 0)
        for x in range(8):
            for y in range(2):
                if y == 0:
                    text = QtWidgets.QLineEdit()
                    text.setFocus(True)
                    text.setAlignment(QtCore.Qt.AlignCenter)
                    text.setText(f'Name {x+1}')
                    text.setStyleSheet("font-size:20px; color:yellow;")
                    self.names.append(text)
                    self.subject_layout.addWidget(text, y, x)    
                if y == 1:
                    btn = QtWidgets.QPushButton(self.ui.Recording_tab)
                    btn.setFont(QtGui.QFont('Times New Roman', 32))
                    btn.setText(f'Player {x+1}')
                    btn.clicked.connect(lambda checked, i=x: self.rcbf.player_reset(self.names[i]))
                    self.player_btn.append(btn)
                    self.subject_layout.addWidget(btn, y, x)
                    
        self.ui.recording_layout.addLayout(self.subject_layout)

        labelsize = [480, 640]
        self.rc_Vision_labels, self.rc_qpixmaps = self.rpbf.creat_vision_labels_pixmaps([x * 1.2 for x in labelsize], self.ui.Recording_tab, self.Deadlift_vision_layout, 'Deadlift', 5)
        self.data_produce_btn.clicked.connect(lambda: self.rcbf.data_produce_btn_clicked('Squat'))
        self.source_ctrl_btn.clicked.connect(lambda: self.rcbf.source_ctrl_btn_clicked('Squat', self.rc_Vision_labels))
        self.recording_ctrl_btn.clicked.connect(lambda: self.rcbf.recording_ctrl_btn_clicked('Squat', self.data_produce_btn, self.source_ctrl_btn, self.back_toolbtn))
       

    def rp_layout_set(self, sport):
        self.layout_clear(self.ui.head_vis_layout)
        self.layout_clear(self.ui.bottom_vis_layout)
        self.layout_clear(self.ui.data_ctrl_layout_V)
        if self.data_layouts:
            for layout in self.data_layouts:
                self.layout_clear(layout)
        self.data_layouts = []
        if self.ui.bottom_controls_layout in [self.ui.data_ctrl_layout_V.itemAt(i).layout() for i in range(self.ui.data_ctrl_layout_V.count())]:
            self.ui.data_ctrl_layout_V.removeItem(self.ui.bottom_controls_layout)
        if sport == 'Deadlift':
            label_size = [480, 640]
            # 左半邊labels
            self.head_Vis_label, _ = self.rpbf.creat_vision_labels_pixmaps([x * 1.4 for x in label_size], self.ui.Replay_tab, self.ui.head_vis_layout, sport, 1)
            self.bottom_Vis_labels, _ = self.rpbf.creat_vision_labels_pixmaps([x * 1.1 for x in label_size], self.ui.Replay_tab, self.ui.bottom_vis_layout, sport, 2)
            
            # 右半邊graphic
            data_layout = QtWidgets.QFormLayout()
            self.data_layouts.append(data_layout)
            graphicview, graphicscene, canvas, axes, table = self.rpbf.creat_graphic(self.ui.Replay_tab, data_layout, (27.5,16.5), 4)
            self.graph = {'graphicview' : graphicview, 'graphicscene' : graphicscene, 'canvas' : canvas, 'axes' : axes}
            self.ui.data_ctrl_layout_V.addLayout(data_layout)
                
            self.ui.bottom_vis_layout.setSpacing(50)
            self.ui.bottom_vis_layout.setContentsMargins(0, 0, 10, 10)
        
            self.rpbf.Deadlift_btn_pressed(
                self.ui.rp_Deadlift_btn, self.ui.rp_Benchpress_btn, self.ui.rp_Squat_btn, self.ui.Play_btn, self.icons, self.ui.Stop_btn, 
                self.ui.Frameslider, self.ui.fast_forward_combobox, self.ui.File_comboBox, self.ui.Replay_tab, self.ui.play_layout,
                self.head_Vis_label, self.bottom_Vis_labels, self.graph, table)
            
        elif sport == 'Benchpress':
            label_size = [640, 480]
            # 左半邊labels
            self.ui.head_vis_layout.setContentsMargins(230, 0, 250, 70)
            self.head_Vis_label, vertical_slider, horizontal_slider = self.rpbf.creat_vision_labels_pixmaps([x * 1.5 for x in label_size], self.ui.Replay_tab, self.ui.head_vis_layout, sport, 1, type = 'rp')
            self.bottom_Vis_labels, vertical_sliders, horizontal_sliders = self.rpbf.creat_vision_labels_pixmaps([x * 1.2 for x in label_size], self.ui.Replay_tab, self.ui.bottom_vis_layout, sport, 2, type = 'rp')
            self.V_sliders = vertical_sliders + [vertical_slider]
            self.H_sliders = horizontal_sliders + [horizontal_slider]
            
            # 右半邊labels
            data_layout = QtWidgets.QFormLayout()
            self.data_layouts.append(data_layout)
            graphicview, graphicscene, canvas, axes, table = self.rpbf.creat_graphic(self.ui.Replay_tab, data_layout, (20, 15), 4)
            self.graph = {'graphicview' : graphicview, 'graphicscene' : graphicscene, 'canvas' : canvas, 'axes' : axes}
            self.ui.data_ctrl_layout_V.addLayout(data_layout)
            
            self.ui.bottom_vis_layout.setSpacing(50)
            self.ui.bottom_vis_layout.setContentsMargins(0, 0, 10, 70)
                
            self.rpbf.Benchpress_btn_pressed(
                self.ui.rp_Deadlift_btn, self.ui.rp_Benchpress_btn, self.ui.rp_Squat_btn, self.ui.Play_btn, self.icons, self.ui.Stop_btn, 
                self.ui.Frameslider, self.ui.fast_forward_combobox, self.ui.File_comboBox, self.ui.Replay_tab, self.ui.play_layout,
                self.head_Vis_label, self.bottom_Vis_labels, self.V_sliders, self.H_sliders, self.graph, table)
            
        elif sport == 'Squat':
            label_size = [480, 640]
            # 左半邊labels
            self.head_Vis_label, _ = self.rpbf.creat_vision_labels_pixmaps([x * 1.4 for x in label_size], self.ui.Replay_tab, self.ui.head_vis_layout, sport, 1)
            self.bottom_Vis_labels, _ = self.rpbf.creat_vision_labels_pixmaps([x * 1.1 for x in label_size], self.ui.Replay_tab, self.ui.bottom_vis_layout, sport, 2)
            
            # 右半邊graphic
            data_layout = QtWidgets.QFormLayout()
            self.data_layouts.append(data_layout)
            graphicview, graphicscene, canvas, axes, table = self.rpbf.creat_graphic(self.ui.Replay_tab, data_layout, (27.5,16.5), 4)
            self.graph = {'graphicview' : graphicview, 'graphicscene' : graphicscene, 'canvas' : canvas, 'axes' : axes}
            self.ui.data_ctrl_layout_V.addLayout(data_layout)
                
            self.ui.bottom_vis_layout.setSpacing(50)
            self.ui.bottom_vis_layout.setContentsMargins(0, 0, 10, 10)
        
            self.rpbf.Squat_btn_pressed(
                self.ui.rp_Deadlift_btn, self.ui.rp_Benchpress_btn, self.ui.rp_Squat_btn, self.ui.Play_btn, self.icons, self.ui.Stop_btn, 
                self.ui.Frameslider, self.ui.fast_forward_combobox, self.ui.File_comboBox, self.ui.Replay_tab, self.ui.play_layout,
                self.head_Vis_label, self.bottom_Vis_labels, self.graph, table)








        self.ui.data_ctrl_layout_V.addLayout(self.ui.bottom_controls_layout)
            
            
    def layout_clear(self, layout):
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

