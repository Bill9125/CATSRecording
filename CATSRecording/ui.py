# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\V1_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from qt_material import apply_stylesheet

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(2171, 843)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1292, 500))

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        self.replay_tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.replay_tabWidget.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.replay_tabWidget.setFont(font)
        self.replay_tabWidget.setObjectName("replay_tabWidget")

        # Recording tab
        self.Recording_tab = QtWidgets.QWidget()
        self.Recording_tab.setObjectName("Recording_tab")
        self.recording_layout = QtWidgets.QVBoxLayout(self.Recording_tab)
        self.recording_layout.setContentsMargins(10, 10, 10, 10)

        self.grid_Layout_recording = QtWidgets.QGridLayout()
        self.grid_Layout_recording.setContentsMargins(0, 0, 0, 0)

        self.manual_checkbox = QtWidgets.QCheckBox(self.Recording_tab)
        self.manual_checkbox.setObjectName("manual_checkbox")
        self.manual_checkbox.setEnabled(False)
        self.manual_checkbox.setChecked(True)
        self.grid_Layout_recording.addWidget(self.manual_checkbox, 0, 0, 1, 1)

        self.rc_Deadlift_btn = QtWidgets.QPushButton(self.Recording_tab)
        font.setPointSize(26)
        self.rc_Deadlift_btn.setFont(font)
        self.rc_Deadlift_btn.setObjectName("Deadlift_btn_3")
        self.grid_Layout_recording.addWidget(self.rc_Deadlift_btn, 1, 0, 1, 1)

        self.rc_Benchpress_btn = QtWidgets.QPushButton(self.Recording_tab)
        self.rc_Benchpress_btn.setFont(font)
        self.rc_Benchpress_btn.setObjectName("Benchpress_btn_3")
        self.grid_Layout_recording.addWidget(self.rc_Benchpress_btn, 2, 0, 1, 1)

        self.rc_Squat_btn = QtWidgets.QPushButton(self.Recording_tab)
        self.rc_Squat_btn.setFont(font)
        self.rc_Squat_btn.setObjectName("Squat_btn_3")
        self.grid_Layout_recording.addWidget(self.rc_Squat_btn, 3, 0, 1, 1)

        self.recording_layout.addLayout(self.grid_Layout_recording)
        self.replay_tabWidget.addTab(self.Recording_tab, "")

        # Replay tab
        font.setPointSize(18)
        self.Replay_tab = QtWidgets.QWidget()
        self.Replay_tab.setObjectName("Replay_tab")
        self.replay_layout = QtWidgets.QVBoxLayout(self.Replay_tab)
        self.replay_layout.setContentsMargins(10, 10, 10, 10)

        self.top_controls_layout = QtWidgets.QHBoxLayout()

        self.rp_Deadlift_btn = QtWidgets.QPushButton(self.Replay_tab)
        self.rp_Deadlift_btn.setStyleSheet("font-size:18px")
        self.rp_Deadlift_btn.setObjectName("Deadlift_play_btn")
        self.top_controls_layout.addWidget(self.rp_Deadlift_btn)

        self.rp_Benchpress_btn = QtWidgets.QPushButton(self.Replay_tab)
        self.rp_Benchpress_btn.setStyleSheet("font-size:18px")
        self.rp_Benchpress_btn.setObjectName("Benchpress_play_btn")
        self.top_controls_layout.addWidget(self.rp_Benchpress_btn)

        self.rp_Squat_btn = QtWidgets.QPushButton(self.Replay_tab)
        self.rp_Squat_btn.setStyleSheet("font-size:18px")
        self.rp_Squat_btn.setObjectName("Squat_play_btn")
        self.top_controls_layout.addWidget(self.rp_Squat_btn)

        self.File_comboBox = QtWidgets.QComboBox(self.Replay_tab)
        self.File_comboBox.setObjectName("File_comboBox")
        self.top_controls_layout.addWidget(self.File_comboBox)
        self.top_controls_layout.setStretch(0, 1)
        self.top_controls_layout.setStretch(1, 1)
        self.top_controls_layout.setStretch(2, 1)
        self.top_controls_layout.setStretch(3, 20)
        self.top_controls_layout.setContentsMargins(0, 0, 500, 0)
        

        self.replay_layout.addLayout(self.top_controls_layout)

        self.play_groupBox = QtWidgets.QGroupBox(self.Replay_tab)
        self.play_groupBox.setObjectName("play_groupBox")
        self.play_layout = QtWidgets.QHBoxLayout(self.play_groupBox)

        self.replay_layout.addWidget(self.play_groupBox)

        self.bottom_controls_layout = QtWidgets.QHBoxLayout()

        self.Play_btn = QtWidgets.QToolButton(self.Replay_tab)
        self.Play_btn.setEnabled(False)
        self.Play_btn.setObjectName("Play_btn")
        self.bottom_controls_layout.addWidget(self.Play_btn)

        self.Stop_btn = QtWidgets.QToolButton(self.Replay_tab)
        self.Stop_btn.setEnabled(False)
        self.Stop_btn.setObjectName("Stop_btn")
        self.bottom_controls_layout.addWidget(self.Stop_btn)

        self.TimeCount_LineEdit = QtWidgets.QLineEdit(self.Replay_tab)
        self.TimeCount_LineEdit.setEnabled(True)
        self.TimeCount_LineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.TimeCount_LineEdit.setReadOnly(True)
        self.TimeCount_LineEdit.setObjectName("TimeCount_LineEdit")

        self.Frameslider = QtWidgets.QSlider(self.Replay_tab)
        self.Frameslider.setEnabled(False)
        self.Frameslider.setOrientation(QtCore.Qt.Horizontal)
        self.Frameslider.setObjectName("Frameslider")

        self.fast_forward_combobox = QtWidgets.QComboBox(self.Replay_tab)
        self.fast_forward_combobox.setEnabled(False)
        self.fast_forward_combobox.setObjectName("fast_forward_combobox")

        self.bottom_controls_layout.addWidget(self.fast_forward_combobox)
        self.bottom_controls_layout.addWidget(self.Frameslider)
        self.bottom_controls_layout.addWidget(self.TimeCount_LineEdit)
        self.bottom_controls_layout.setStretch(0, 0)
        self.bottom_controls_layout.setStretch(1, 0)
        self.bottom_controls_layout.setStretch(2, 0)
        self.bottom_controls_layout.setStretch(3, 90)
        self.bottom_controls_layout.setStretch(4, 5)
        

        self.replay_layout.addLayout(self.bottom_controls_layout)
        self.replay_tabWidget.addTab(self.Replay_tab, "")

        self.main_layout.addWidget(self.replay_tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.replay_tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CATS Webcamviewer"))
        self.rc_Benchpress_btn.setText(_translate("MainWindow", "Benchpress"))
        self.rc_Squat_btn.setText(_translate("MainWindow", "Squat"))
        self.rc_Deadlift_btn.setText(_translate("MainWindow", "Deadlift"))
        self.manual_checkbox.setText(_translate("MainWindow", "manual recording"))
        self.replay_tabWidget.setTabText(self.replay_tabWidget.indexOf(self.Recording_tab), _translate("MainWindow", "Recording"))
        self.TimeCount_LineEdit.setPlaceholderText(_translate("MainWindow", "00:00"))
        self.rp_Deadlift_btn.setText(_translate("MainWindow", "Deadlift"))
        self.rp_Benchpress_btn.setText(_translate("MainWindow", "Benchpress"))
        self.rp_Squat_btn.setText(_translate("MainWindow", "Squat"))
        self.replay_tabWidget.setTabText(self.replay_tabWidget.indexOf(self.Replay_tab), _translate("MainWindow", "Replay"))
