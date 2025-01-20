import csv
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt

class ButtonClickApp(QMainWindow):
    def __init__(self, n):
        super().__init__()
        self.click_order = []  # 用於記錄點擊順序
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
        self.save_to_csv()
        self.close()
        
    def save_to_csv(self):
        # 將點擊順序寫入 CSV 文件
        with open('../config/click_order.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.click_order)  # 寫入一行順序
