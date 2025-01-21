import json, os
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets

class ButtonClickApp(QMainWindow):
    ok_clicked = pyqtSignal()

    def __init__(self, n, sport):
        super().__init__()
        self.click_order = []  # 用於記錄點擊順序
        self.sport = sport
        self.init_ui(n)

    def init_ui(self, n):
        self.setWindowTitle("Vision source")
        self.setGeometry(100, 100, 300, 200)

        central_widget = QWidget()
        main_layout = QVBoxLayout()

        self.info_label = QLabel("Click the visions in right order.")
        self.info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.info_label)

        # Create a horizontal layout for the vision buttons
        vision_layout = QHBoxLayout()

        self.buttons = []
        for i in range(1, n + 1):
            button = QPushButton(f"Vision{i}")
            button.clicked.connect(lambda _, b=button: self.record_click(b))
            self.buttons.append(button)
            vision_layout.addWidget(button)

        main_layout.addLayout(vision_layout)  # Add the vision button layout to the main layout

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.check_order)
        main_layout.addWidget(self.ok_button)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def record_click(self, button):
        text = button.text()
        button.setEnabled(False)
        if text not in self.click_order:
            self.click_order.append(int(text[-1]) - 1)

    def check_order(self):
        self.save_to_json()
        self.ok_clicked.emit()
        self.close()

    def save_to_json(self):
        # 將點擊順序寫入 JSON 文件
        file_path = '../config/click_order.json'
        try:
            # 如果文件存在且不是空的，讀取現有數據
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                with open(file_path, mode='r', newline='', encoding='utf-8') as file:
                    existing_data = json.load(file)
            else:
                existing_data = {}  # 如果文件不存在或為空，初始化為空字典

            # 更新或新增鍵
            existing_data[self.sport] = self.click_order
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                json.dump(existing_data, file, indent=4, ensure_ascii=False)
            print(f"Click order saved to {file_path}")
        except Exception as e:
            print(f"Failed to save JSON file: {e}")
            