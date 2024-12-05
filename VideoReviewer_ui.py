# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\VideoReviewer_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

# pyinstaller --onefile -n 'CATS VideoReviewer' --add-data "ui_src/*.png;ui_src" .\VideoReviewer_ui.py

from PyQt5 import QtCore, QtGui, QtWidgets
from qt_material import apply_stylesheet
import glob
import os, sys
import cv2, time
import threading

class Ui_myvideoreviewer(object):
    def setupUi(self, myvideoreviewer):
        self.icons = []
        self.cameras = []
        self.qpixmaps = []
        self.Vision_labels = []
        self.threads = []
        self.ocv = True
        self.index = 0

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextframe)

        # Set icon
        icon_srcs = glob.glob(self.resource_path('ui_src/*.png'))
        for icon in icon_srcs:
            self.icons.append(QtGui.QIcon(icon))
        
        myvideoreviewer.setObjectName("myvideoreviewer")
        myvideoreviewer.setWindowModality(QtCore.Qt.NonModal)
        myvideoreviewer.resize(2172, 784)
        myvideoreviewer.setMouseTracking(False)
        myvideoreviewer.setFocusPolicy(QtCore.Qt.WheelFocus)
        myvideoreviewer.setToolTipDuration(-2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(myvideoreviewer)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.groupBox = QtWidgets.QGroupBox(myvideoreviewer)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 680, 2131, 73))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setContentsMargins(10, 0, 10, 0)
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.Play_btn = QtWidgets.QToolButton(self.horizontalLayoutWidget)
        self.Play_btn.setEnabled(False)
        self.Play_btn.setText("")
        self.Play_btn.setAutoRaise(False)
        self.Play_btn.setObjectName("Play_btn")
        self.Play_btn.setIcon(self.icons[3])
        self.Play_btn.clicked.connect(self.autoplay)
        self.horizontalLayout.addWidget(self.Play_btn)

        self.Stop_btn = QtWidgets.QToolButton(self.horizontalLayoutWidget)
        self.Stop_btn.setEnabled(False)
        self.Stop_btn.setText("")
        self.Stop_btn.setAutoRaise(False)
        self.Stop_btn.setObjectName("Stop_btn")
        self.Stop_btn.setIcon(self.icons[4])
        self.Stop_btn.clicked.connect(self.stop)
        self.horizontalLayout.addWidget(self.Stop_btn)

        self.Frameslider = QtWidgets.QSlider(self.horizontalLayoutWidget)
        self.Frameslider.setEnabled(False)
        self.Frameslider.setSingleStep(1)
        self.Frameslider.setProperty("value", 0)
        self.Frameslider.setTracking(True)
        self.Frameslider.setOrientation(QtCore.Qt.Horizontal)
        self.Frameslider.setObjectName("Frameslider")
        self.Frameslider.valueChanged.connect(self.sliding)
        self.Frameslider.sliderMoved.connect(self.interupsliding)
        self.horizontalLayout.addWidget(self.Frameslider)

        self.TimeCount_LineEdit = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.TimeCount_LineEdit.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(50)
        font.setBold(True)
        font.setWeight(75)
        self.TimeCount_LineEdit.setFont(font)
        self.TimeCount_LineEdit.setCursor(QtGui.QCursor(QtCore.Qt.IBeamCursor))
        self.TimeCount_LineEdit.setMouseTracking(False)
        self.TimeCount_LineEdit.setAcceptDrops(True)
        self.TimeCount_LineEdit.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.TimeCount_LineEdit.setStyleSheet("background-color: rgba(255, 255, 255, 0); font-size:15px; color:yellow;")
        self.TimeCount_LineEdit.setFrame(False)
        self.TimeCount_LineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.TimeCount_LineEdit.setReadOnly(True)
        self.TimeCount_LineEdit.setObjectName("TimeCount_LineEdit")
        self.horizontalLayout.addWidget(self.TimeCount_LineEdit)
        self.horizontalLayout.setStretch(0,0)
        self.horizontalLayout.setStretch(1,0)
        self.horizontalLayout.setStretch(2,90)
        self.horizontalLayout.setStretch(3,5)

        self.selectFile_btn = QtWidgets.QToolButton(self.groupBox)
        self.selectFile_btn.setGeometry(QtCore.QRect(20, 10, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        self.selectFile_btn.setFont(font)
        self.selectFile_btn.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.selectFile_btn.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.selectFile_btn.setObjectName("selectFile_btn")
        self.selectFile_btn.clicked.connect(self.loadfolder)

        self.Filelist_comboBox = QtWidgets.QComboBox(self.groupBox)
        self.Filelist_comboBox.setEnabled(True)
        self.Filelist_comboBox.setGeometry(QtCore.QRect(110, 10, 781, 22))
        self.Filelist_comboBox.setEditable(True)
        self.Filelist_comboBox.setObjectName("Filelist_comboBox")
        self.Filelist_comboBox.setEditable(False)
        self.Filelist_comboBox.setStyleSheet("font-size:15px; color:yellow;")
        self.Filelist_comboBox.currentIndexChanged.connect(self.readvideofile)

        self.gridLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 40, 2131, 631))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.gridLayout.setSpacing(20)
        self.gridLayout.setObjectName("gridLayout")

        self.Vision_lineEdit_2 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        self.Vision_lineEdit_2.setFont(font)
        self.Vision_lineEdit_2.setStyleSheet("background-color: rgba(255, 255, 255, 0); font-size:20px; color:yellow;")
        self.Vision_lineEdit_2.setFrame(False)
        self.Vision_lineEdit_2.setReadOnly(True)
        self.Vision_lineEdit_2.setObjectName("Vision_lineEdit_2")
        self.Vision_lineEdit_2.setText('vision 2')
        self.Vision_lineEdit_2.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.Vision_lineEdit_2, 1, 1, 1, 1)

        self.Vision_lineEdit_1 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(24)
        self.Vision_lineEdit_1.setFont(font)
        self.Vision_lineEdit_1.setStyleSheet("background-color: rgba(255, 255, 255, 0); font-size:20px; color:yellow;")
        self.Vision_lineEdit_1.setFrame(False)
        self.Vision_lineEdit_1.setDragEnabled(False)
        self.Vision_lineEdit_1.setReadOnly(True)
        self.Vision_lineEdit_1.setClearButtonEnabled(False)
        self.Vision_lineEdit_1.setObjectName("Vision_lineEdit_1")
        self.Vision_lineEdit_1.setText('vision 1')
        self.Vision_lineEdit_1.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.Vision_lineEdit_1, 1, 0, 1, 1)

        self.Vision_lineEdit_4 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(24)
        self.Vision_lineEdit_4.setFont(font)
        self.Vision_lineEdit_4.setStyleSheet("background-color: rgba(255, 255, 255, 0); font-size:20px; color:yellow;")
        self.Vision_lineEdit_4.setFrame(False)
        self.Vision_lineEdit_4.setReadOnly(True)
        self.Vision_lineEdit_4.setObjectName("Vision_lineEdit_4")
        self.Vision_lineEdit_4.setText('vision 4')
        self.Vision_lineEdit_4.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.Vision_lineEdit_4, 1, 3, 1, 1)

        self.Vision_lineEdit_5 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(24)
        self.Vision_lineEdit_5.setFont(font)
        self.Vision_lineEdit_5.setStyleSheet("background-color: rgba(255, 255, 255, 0); font-size:20px; color:yellow;")
        self.Vision_lineEdit_5.setFrame(False)
        self.Vision_lineEdit_5.setReadOnly(True)
        self.Vision_lineEdit_5.setObjectName("Vision_lineEdit_5")
        self.Vision_lineEdit_5.setText('vision 5')
        self.Vision_lineEdit_5.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.Vision_lineEdit_5, 1, 4, 1, 1)

        self.Vision_lineEdit_3 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(24)
        self.Vision_lineEdit_3.setFont(font)
        self.Vision_lineEdit_3.setStyleSheet("background-color: rgba(255, 255, 255, 0); font-size:20px; color:yellow;")
        self.Vision_lineEdit_3.setFrame(False)
        self.Vision_lineEdit_3.setReadOnly(True)
        self.Vision_lineEdit_3.setObjectName("Vision_lineEdit_3")
        self.Vision_lineEdit_3.setText('vision 3')
        self.Vision_lineEdit_3.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.Vision_lineEdit_3, 1, 2, 1, 1)

        self.Vision_label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Vision_label_5.setFrameShape(QtWidgets.QFrame.Panel)
        self.Vision_label_5.setText("")
        self.Vision_label_5.setScaledContents(True)
        self.Vision_label_5.setObjectName("Vision_label_5")
        self.gridLayout.addWidget(self.Vision_label_5, 0, 4, 1, 1)
        self.Vision_label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Vision_label_2.setFrameShape(QtWidgets.QFrame.Panel)
        self.Vision_label_2.setText("")
        self.Vision_label_2.setScaledContents(True)
        self.Vision_label_2.setObjectName("Vision_label_2")
        self.gridLayout.addWidget(self.Vision_label_2, 0, 1, 1, 1)
        self.Vision_label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Vision_label_4.setFrameShape(QtWidgets.QFrame.Panel)
        self.Vision_label_4.setText("")
        self.Vision_label_4.setScaledContents(True)
        self.Vision_label_4.setObjectName("Vision_label_4")
        self.gridLayout.addWidget(self.Vision_label_4, 0, 3, 1, 1)
        self.Vision_label_1 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Vision_label_1.setFrameShape(QtWidgets.QFrame.Panel)
        self.Vision_label_1.setText("")
        self.Vision_label_1.setScaledContents(True)
        self.Vision_label_1.setObjectName("Vision_label_1")
        self.gridLayout.addWidget(self.Vision_label_1, 0, 0, 1, 1)
        self.Vision_label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Vision_label_3.setFrameShape(QtWidgets.QFrame.Panel)
        self.Vision_label_3.setText("")
        self.Vision_label_3.setScaledContents(True)
        self.Vision_label_3.setObjectName("Vision_label_3")
        self.gridLayout.addWidget(self.Vision_label_3, 0, 2, 1, 1)
        self.horizontalLayout_3.addWidget(self.groupBox)
        self.Vision_labels = [self.Vision_label_1, self.Vision_label_2, self.Vision_label_3, self.Vision_label_4, self.Vision_label_5]

        self.qpixmap_1 = QtGui.QPixmap()
        self.qpixmap_2 = QtGui.QPixmap()
        self.qpixmap_3 = QtGui.QPixmap()
        self.qpixmap_4 = QtGui.QPixmap()
        self.qpixmap_5 = QtGui.QPixmap()
        self.qpixmaps = [self.qpixmap_1, self.qpixmap_2, self.qpixmap_3, self.qpixmap_4, self.qpixmap_5]

        self.retranslateUi(myvideoreviewer)
        QtCore.QMetaObject.connectSlotsByName(myvideoreviewer)

    def resource_path(self, relative_path):
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        return os.path.join(base_path, relative_path)

    def loadfolder(self):
        self.folderPath = QtWidgets.QFileDialog.getExistingDirectory(None ,'Open folder', self.resource_path('./'))
        self.Frameslider.setEnabled(True)
        self.Play_btn.setEnabled(True)
        self.Stop_btn.setEnabled(True)
        self.Filelist_comboBox.clear()
        if self.folderPath:
            for folder in os.listdir(self.folderPath):
                self.Filelist_comboBox.addItems([folder])

    # 讀取combobox內的資料夾
    def readvideofile(self):
        videofolder = self.Filelist_comboBox.currentText()
        self.index = 0
        self.Play_btn.setIcon(self.icons[3])
        self.videos = glob.glob(f'{self.folderPath}/{videofolder}/*.avi')
        self.stop()
        # Clear all the Qpixmap
        for i in range(len(self.Vision_labels)):
            self.Vision_labels[i].setPixmap(QtGui.QPixmap())

        # 臥推有六部影片，要抽取三部
        if len(self.videos) >= 6:
            self.videos = [video for video in self.videos 
                           if os.path.basename(video) in ('original_vision1.avi', 'vision2.avi', 'original_vision3.avi')
                        ]
            self.videos[1], self.videos[2] = self.videos[2], self.videos[1]

        # show the first frame of videos
        for i in range(len(self.videos)):
            thread = threading.Thread(target=self.showprevision, args=(self.videos[i], self.qpixmaps[i], self.Vision_labels[i]))
            self.threads.append(thread)
            thread.start()

    def showprevision(self, video, qpixmap, label):
        cap = cv2.VideoCapture(video)
        if self.ocv:
            _ , frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QtGui.QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QtGui.QImage.Format_RGB888)
            qpixmap = QtGui.QPixmap.fromImage(image)
            scaled_pixmap = qpixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio)
            label.setPixmap(scaled_pixmap)

    def nextframe(self):
        # slider value +1
        val = self.Frameslider.value()
        self.Frameslider.setSliderPosition(val+1)

    def autoplay(self):
        self.start = time.time()
        self.index += 1
        # play
        if self.index % 2 == 1: 
            self.Play_btn.setIcon(self.icons[2])
            f_num = []
            for video in self.videos:
                cap = cv2.VideoCapture(video)
                f_num.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            framenumber = min(f_num)
            # FPS = 30
            self.timer.start(1)
            self.Frameslider.setMaximum(int(framenumber))

        # pause
        elif self.index % 2 == 0:
            self.Play_btn.setIcon(self.icons[3])
            self.timer.stop()
        
    def stop(self):
        self.timer.stop()
        self.Frameslider.setSliderPosition(0)
        self.Play_btn.setIcon(self.icons[3])
        self.index = 0

    def interupsliding(self):
        self.timer.stop()
        self.Play_btn.setIcon(self.icons[3])
        self.index = 0

    def sliding(self):
        val = self.Frameslider.value()
        sec = val/30
        minute = "%02d" % int(sec / 60)
        second = "%02d" % int(sec % 60)
        self.TimeCount_LineEdit.setText(f'{minute}:{second}')

        if val < self.Frameslider.maximum():
            for i in range(len(self.videos)):
                cap = cv2.VideoCapture(self.videos[i])
                cap.set(cv2.CAP_PROP_POS_FRAMES, val)
                _ , frame = cap.read()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QtGui.QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QtGui.QImage.Format_RGB888)
                self.qpixmaps[i] = QtGui.QPixmap.fromImage(image)
                scaled_pixmap = self.qpixmaps[i].scaled(self.Vision_labels[i].size(), QtCore.Qt.KeepAspectRatio)
                self.Vision_labels[i].setPixmap(scaled_pixmap)
        else:
            self.Play_btn.setIcon(self.icons[3])
            self.index = 0
            print(f'It actually took {time.time() - self.start} seconds.')

    def retranslateUi(self, myvideoreviewer):
        _translate = QtCore.QCoreApplication.translate
        myvideoreviewer.setWindowTitle(_translate("myvideoreviewer", "VideoReviewer CATS"))
        self.TimeCount_LineEdit.setPlaceholderText(_translate("myvideoreviewer", "00:00"))
        self.selectFile_btn.setText(_translate("myvideoreviewer", "File"))

    def closeEvent(self):
        self.ocv = False

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_amber.xml')
    myvideoreviewer = QtWidgets.QWidget()
    ui = Ui_myvideoreviewer()
    ui.setupUi(myvideoreviewer)
    myvideoreviewer.show()
    sys.exit(app.exec_())
