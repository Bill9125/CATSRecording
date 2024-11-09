# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
import sys, os
import glob
import json
from os.path import join
from avi_to_mp4 import avi_2_mp4

class Ui_camcalib(object):
    def setupUi(self, camcalib):
        camcalib.setObjectName("camcalib")
        camcalib.resize(810, 741)
        self.tabWidget = QtWidgets.QTabWidget(camcalib)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 551, 711))
        self.tabWidget.setObjectName("tabWidget")
        self.tabs = []
        self.labels = []
        self.qpixmaps = []
        self.zoom_factors = [1.0, 1.0, 1.0, 1.0]  # Zoom factors
        self.offsets = [(0, 0), (0, 0), (0, 0), (0, 0)]  # Offset for each image
        self.drag_start_position = None  # Drag start position
        self.checkbox_states = {i: False for i in range(4)}  # 初始化每個 Tab 的 QCheckBox 狀態為 False

        # SpinBox to select keypoint index (initialize it first)
        self.spinBox = QtWidgets.QSpinBox(camcalib)
        self.spinBox.setGeometry(QtCore.QRect(660, 250, 70, 25))
        self.spinBox.setObjectName("spinBox")
        self.spinBox.setRange(1, 15)  # Default range, will be updated per JSON

        # Initialize 4 tabs and corresponding QLabel and QPixmap
        for i in range(4):
            tab = QtWidgets.QWidget()
            tab.setObjectName(f"tab_{i+1}")
            label = DraggableLabel(self, tab, i)
            label.setGeometry(QtCore.QRect(30, 20, 480, 640))
            label.setObjectName(f"label_{i+1}")
            self.tabWidget.addTab(tab, f"cam_{i+1}")
            self.tabs.append(tab)
            self.labels.append(label)
            qpixmap = QtGui.QPixmap()
            self.qpixmaps.append(qpixmap)
        self.tabWidget.currentChanged.connect(self.on_tab_changed)

        # sequence inverse checkbox
        self.reverseCheckBox = QtWidgets.QCheckBox(camcalib)
        self.reverseCheckBox.setGeometry(QtCore.QRect(590, 273, 190, 31))
        self.reverseCheckBox.setText('Inverse Sequence')
        self.reverseCheckBox.stateChanged.connect(self.toggle_reverse_keypoints)
        
        # build extri.yml button
        self.buildextriBtn = QtWidgets.QPushButton(camcalib)
        self.buildextriBtn.setGeometry(QtCore.QRect(590, 600, 190, 31))
        self.buildextriBtn.setText("Build extri.yml")
        self.buildextriBtn.clicked.connect(self.buildextri)

        # show cube button
        self.showcubeBtn = QtWidgets.QPushButton(camcalib)
        self.showcubeBtn.setGeometry(QtCore.QRect(590, 650, 190, 31))
        self.showcubeBtn.setText("Show Cube")
        self.showcubeBtn.clicked.connect(self.showcube)
        
        # Load data button
        self.LoaddataBtn = QtWidgets.QPushButton(camcalib)
        self.LoaddataBtn.setGeometry(QtCore.QRect(590, 40, 190, 31))
        self.LoaddataBtn.setText("Load extri_data")
        self.LoaddataBtn.clicked.connect(self.load_path)

        # extri_calib button
        self.loadImage1Btn = QtWidgets.QPushButton(camcalib)
        self.loadImage1Btn.setGeometry(QtCore.QRect(590, 90, 190, 31))
        self.loadImage1Btn.setText("Chessboard_Detect")
        self.loadImage1Btn.clicked.connect(self.chessboard_detect)

        # Zoom buttons
        self.pushButton = QtWidgets.QPushButton(camcalib)
        self.pushButton.setGeometry(QtCore.QRect(590, 140, 81, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Zoom in")
        self.pushButton.clicked.connect(self.Zoomin)

        self.pushButton_2 = QtWidgets.QPushButton(camcalib)
        self.pushButton_2.setGeometry(QtCore.QRect(700, 140, 81, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText("Zoom out")
        self.pushButton_2.clicked.connect(self.Zoomout)

        # Keypoint input area
        self.label = QtWidgets.QLabel(camcalib)
        self.label.setGeometry(QtCore.QRect(590, 195, 51, 21))
        self.label.setText("Keypoints")
        self.label.setObjectName("label")

        self.lineEdit_2 = QtWidgets.QLineEdit(camcalib)
        self.lineEdit_2.setGeometry(QtCore.QRect(640, 220, 41, 21))
        self.lineEdit_2.setObjectName("lineEdit_2")

        self.label_5 = QtWidgets.QLabel(camcalib)
        self.label_5.setGeometry(QtCore.QRect(620, 222, 16, 16))
        self.label_5.setText("X")
        self.label_5.setObjectName("label_5")

        self.label_6 = QtWidgets.QLabel(camcalib)
        self.label_6.setGeometry(QtCore.QRect(700, 222, 16, 16))
        self.label_6.setText("Y")
        self.label_6.setObjectName("label_6")

        self.label_7 = QtWidgets.QLabel(camcalib)
        self.label_7.setGeometry(QtCore.QRect(620, 249, 70, 25))
        self.label_7.setText("index :")
        self.label_7.setObjectName("label_7")

        self.lineEdit_3 = QtWidgets.QLineEdit(camcalib)
        self.lineEdit_3.setGeometry(QtCore.QRect(720, 220, 41, 21))
        self.lineEdit_3.setObjectName("lineEdit_3")

        self.retranslateUi(camcalib)
        QtCore.QMetaObject.connectSlotsByName(camcalib)

    def retranslateUi(self, camcalib):
        _translate = QtCore.QCoreApplication.translate
        camcalib.setWindowTitle(_translate("camcalib", "Camera Calibration"))

    def chessboard_detect(self):
        if os.path.isdir(self.path):
            extri_videos_path = join(self.path, 'videos')
            for filename in os.listdir(extri_videos_path):
                if filename.endswith(".avi"):
                    avi_2_mp4(extri_videos_path)
                    break

        cmd_1 = f'python C:/MOCAP/EasyMocap/scripts/preprocess/extract_video.py {self.path} --no2d'
        os.system(cmd_1)
        cmd_2 = f'python C:/MOCAP/EasyMocap/scripts/preprocess/random_process.py {self.path}'
        os.system(cmd_2)
        chessboard = join(self.path, 'chessboard')
        cmd_3 = f'python C:/MOCAP/EasyMocap/apps/calibration/detect_chessboard.py {self.path} --out {self.path}/{chessboard} --pattern 5,3 --grid 0.135'
        os.system(cmd_3)
        self.load_image('images')

    def load_path(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(directory='C:/MOCAP/EasyMocap/')
        if path:
            self.path = path  # 儲存路徑

    def load_image(self, type):
        # Load images and set initial display
        if type == 'images':
            for x in range(1, 5):
                imagelist = glob.glob(f'{self.path}/{type}/{x}/*.jpg')
                if imagelist:
                    self.qpixmaps[x-1].load(imagelist[0])  # Load the first image in the directory
                    self.labels[x-1].setPixmap(self.qpixmaps[x-1])
                    self.labels[x-1].image_loaded = True  # 圖片加載後設置為 True
                    # 傳遞選擇的路徑給 DraggableLabel
                    self.labels[x-1].set_json_path(self.path)
        elif type == 'cube':
            imagelist = glob.glob(f'{self.path}/{type}/*.jpg')
            for label in self.labels:
                label.clear_circles()
            for x in range(1, 5):
                if x-1 < len(imagelist):
                    if not self.qpixmaps[x-1].load(imagelist[x-1]):
                        print(f"Failed to load image for cam_{x}")
                    else:
                        self.labels[x-1].setPixmap(self.qpixmaps[x-1])
                else:
                    print(f"No image available for cam_{x}")
            
        

    def Zoomin(self):
        self.zoomImage(1.2)

    def Zoomout(self):
        self.zoomImage(1 / 1.2)

    def showcube(self):
        for x in range (1, 5):
            os.rename(glob.glob(f'{self.path}/images/{x}/*.jpg')[0],f'{self.path}/images/{x}/00000000.jpg')
        cmd_5 = f'python C:/MOCAP/EasyMocap/apps/calibration/check_calib.py {self.path} --out {self.path} --mode cube --write'
        os.system(cmd_5)
        self.load_image('cube')

    def buildextri(self):
        cmd_4 = f'python C:/MOCAP/EasyMocap/apps/calibration/calib_extri.py {self.path} --intri {self.path}/intri.yml'
        os.system(cmd_4)

    def zoomImage(self, scale_factor):
        current_tab = self.tabWidget.currentIndex()
        label = self.labels[current_tab]
        self.zoom_factors[current_tab] *= scale_factor

        # Calculate zoom center as the current image display center
        label_center_x = label.width() // 2
        label_center_y = label.height() // 2

        # Adjust offset to keep the zoom center in the same position
        offset_x = label_center_x - (label_center_x - label.pixmap_offset.x()) * scale_factor
        offset_y = label_center_y - (label_center_y - label.pixmap_offset.y()) * scale_factor
        label.pixmap_offset = QtCore.QPoint(int(offset_x), int(offset_y))

        # Update the display of the zoomed image
        self.updateZoom(current_tab)

    def updateZoom(self, tab_index):
        label = self.labels[tab_index]
        pixmap = self.qpixmaps[tab_index]
        width = int(pixmap.width() * self.zoom_factors[tab_index])
        height = int(pixmap.height() * self.zoom_factors[tab_index])
        scaled_pixmap = pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio)
        label.setScaledPixmap(scaled_pixmap)  # Use DraggableLabel's setScaledPixmap

    def toggle_reverse_keypoints(self, state):
        current_tab = self.tabWidget.currentIndex()
        self.checkbox_states[current_tab] = (state == QtCore.Qt.Checked)  # 更新當前頁籤的 QCheckBox 狀態

        label = self.labels[current_tab]
        if state == QtCore.Qt.Checked:
            label.reverse_keypoints_order()
        else:
            label.restore_original_keypoints_order()

    def on_tab_changed(self, index):
        # 根據當前頁籤的狀態更新 QCheckBox
        self.reverseCheckBox.blockSignals(True)  # 避免觸發 stateChanged 信號
        self.reverseCheckBox.setChecked(self.checkbox_states[index])
        self.reverseCheckBox.blockSignals(False)

    def updateKeypoint(self, x, y):
        """Update keypoint display coordinates"""
        self.lineEdit_2.setText(str(x))
        self.lineEdit_3.setText(str(y))
    
    def updateCircleDisplay(self):
        """Update circle display when spinBox value changes"""
        current_tab = self.tabWidget.currentIndex()
        self.labels[current_tab].update()  # Trigger repaint in current tab

class DraggableLabel(QtWidgets.QLabel):
    def __init__(self, main_window, parent, tab_index):
        super().__init__(parent)
        self.main_window = main_window
        self.tab_index = tab_index
        self.setMouseTracking(True)
        self.is_dragging = False
        self.pixmap_offset = QtCore.QPoint(0, 0)
        self.circles = {}
        self.json_file = None
        self.keypoints_data = None
        self.image_loaded = False
        self.original_keypoints_order = None  # 用來保存原始順序

    def clear_circles(self):
        """Clear circles visually without modifying JSON data"""
        self.circles.clear()
        self.update()  # 重新繪製來清除顯示的圓圈

    def set_json_path(self, path):
        """根據選擇的路徑來設定 JSON 檔案路徑並讀取 keypoints"""
        self.json_file = glob.glob(f'{path}/chessboard/{self.tab_index + 1}/*.json')[0]
        with open(self.json_file, 'r') as f:
            self.keypoints_data = json.load(f)
            # 保存一份 keypoints2d 的原始順序
            self.original_keypoints_order = self.keypoints_data["keypoints2d"].copy()

    def reverse_keypoints_order(self):
        """顛倒 keypoints2d 順序並更新 JSON 檔案"""
        self.keypoints_data["keypoints2d"].reverse()
        self.update_json_file()

    def restore_original_keypoints_order(self):
        """恢復 keypoints2d 的原始順序並更新 JSON 檔案"""
        if self.original_keypoints_order is not None:
            self.keypoints_data["keypoints2d"] = self.original_keypoints_order.copy()
            self.update_json_file()

    def update_json_file(self):
        """將當前的 keypoints2d 寫回 JSON 檔案"""
        with open(self.json_file, 'w') as f:
            json.dump(self.keypoints_data, f, indent=4)
    def setScaledPixmap(self, pixmap):
        """Set scaled QPixmap while preserving offset"""
        self.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.is_dragging = True
            self.drag_start_position = event.pos()

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            # Calculate new offset
            offset_x = event.pos().x() - self.drag_start_position.x()
            offset_y = event.pos().y() - self.drag_start_position.y()
            self.pixmap_offset += QtCore.QPoint(offset_x, offset_y)
            self.update()
            self.drag_start_position = event.pos()

        # Update mouse coordinates in the image
        scaled_x = (event.x() - self.pixmap_offset.x()) / self.main_window.zoom_factors[self.tab_index]
        scaled_y = (event.y() - self.pixmap_offset.y()) / self.main_window.zoom_factors[self.tab_index]
        self.main_window.updateKeypoint(int(scaled_x), int(scaled_y))

    def mouseDoubleClickEvent(self, event):
        if self.image_loaded:  # 檢查圖片是否已加載
            if event.button() == QtCore.Qt.LeftButton:
                # 取得目前鼠標位置，並扣除偏移量得到圖片內相對位置
                relative_x = (event.pos().x() - self.pixmap_offset.x()) / self.main_window.zoom_factors[self.tab_index]
                relative_y = (event.pos().y() - self.pixmap_offset.y()) / self.main_window.zoom_factors[self.tab_index]
                x_value = int(relative_x)
                y_value = int(relative_y)
                
                # 取得當前的索引值
                index = self.main_window.spinBox.value()
                if 1 <= index <= len(self.keypoints_data["keypoints2d"]):
                    self.keypoints_data["keypoints2d"][index-1] = [x_value, y_value, 1]

                # 儲存相對圖片的圓圈座標
                self.circles[index] = (x_value, y_value)  
                
                # 更新 JSON 文件
                with open(self.json_file, 'w') as f:
                    json.dump(self.keypoints_data, f, indent=4)

                # 將 spinBox 設置為下一個索引
                next_index = (index + 1) % len(self.keypoints_data["keypoints2d"])
                self.main_window.spinBox.setValue(next_index)
                
                self.update()  # 重新繪製

                if index < len(self.keypoints_data["keypoints2d"]):
                    self.main_window.spinBox.setValue(index + 1)
                else:
                    self.main_window.spinBox.setValue(1)
                

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.is_dragging = False

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        if self.pixmap():
            painter.drawPixmap(self.pixmap_offset, self.pixmap())
        
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 2))
        painter.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))

        # 將每個儲存的相對位置的圓圈，考慮偏移量後繪製在圖片上
        for index, (relative_x, relative_y) in self.circles.items():
            adjusted_x = int(relative_x * self.main_window.zoom_factors[self.tab_index] + self.pixmap_offset.x())
            adjusted_y = int(relative_y * self.main_window.zoom_factors[self.tab_index] + self.pixmap_offset.y())
            
            painter.drawEllipse(QtCore.QPoint(adjusted_x, adjusted_y), 10, 10)
            painter.drawText(adjusted_x - 5, adjusted_y + 5, str(index))

        painter.end()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    camcalib = QtWidgets.QWidget()
    ui = Ui_camcalib()
    ui.setupUi(camcalib)
    camcalib.show()
    sys.exit(app.exec_())
