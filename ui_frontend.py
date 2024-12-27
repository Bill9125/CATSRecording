from ui import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from qt_material import apply_stylesheet

# frontend logic
class Mainwindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(Mainwindow, self).__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.rc_Deadlift_btn.clicked.connect(self.rc_Deadlift_clicked)
        # self.ui.rc_Benchpress_btn.clicked.connect(self.rc_Benchpress_clicked())
        # self.ui.rc_Squat_btn.clicked.connect(self.rc_Squat_clicked())

    def rc_Deadlift_clicked(self):
        self.Deadlift_layout_set()

    def Deadlift_layout_set(self):
        # clear recording layout
        grid_layout = self.ui.grid_Layout_recording
        for i in reversed(range(grid_layout.count())):
            widget = grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # set recording layout
        self.Deadlift_vision_layout = QtWidgets.QHBoxLayout()
        self.Deadlift_vision_layout.setContentsMargins(0, 0, 0, 0)
        self.ui.recording_layout.addLayout(self.Deadlift_vision_layout)

        self.recording_ctrl = QtWidgets.QPushButton(self.ui.Recording_tab)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.recording_ctrl.setFont(font)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_amber.xml')
    win = Mainwindow()
    win.show()
    sys.exit(app.exec_())