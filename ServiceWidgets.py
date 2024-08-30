"""Fun ction to use useful service widgets, like progressbars etc...

"""

from PyQt5 import QtWidgets


class ProgressBar(QtWidgets.QWidget):
    """Simple progressbar widget."""
    def __init__(self, parent=None, total1=20):
        super().__init__(parent)
        self.name_line1  =  QtWidgets.QLineEdit()

        self.progressbar1  =  QtWidgets.QProgressBar()
        self.progressbar1.setMinimum(1)
        self.progressbar1.setMaximum(total1)

        main_layout  =  QtWidgets.QGridLayout()
        main_layout.addWidget(self.progressbar1, 0, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Progress")
        self.setGeometry(500, 300, 300, 50)

    def update_progressbar1(self, val1):
        """Update progressbar."""
        self.progressbar1.setValue(val1)
        QtWidgets.qApp.processEvents()


class ProgressBarDouble(QtWidgets.QWidget):
    """Double Progressbar widget."""
    def __init__(self, parent=None, total1=20, total2=20):
        super().__init__(parent)
        self.name_line1  =  QtWidgets.QLineEdit()

        self.progressbar1  =  QtWidgets.QProgressBar()
        self.progressbar1.setMinimum(1)
        self.progressbar1.setMaximum(total1)

        self.progressbar2  =  QtWidgets.QProgressBar()
        self.progressbar2.setMinimum(1)
        self.progressbar2.setMaximum(total2)

        main_layout  =  QtWidgets.QGridLayout()
        main_layout.addWidget(self.progressbar1, 0, 0)
        main_layout.addWidget(self.progressbar2, 1, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Progress")
        self.setGeometry(500, 300, 300, 50)


    def update_progressbar1(self, val1):
        """Update progressbar 1."""
        self.progressbar1.setValue(val1)
        QtWidgets.qApp.processEvents()


    def update_progressbar2(self, val2):
        """Update progressbar 2."""
        self.progressbar2.setValue(val2)
        QtWidgets.qApp.processEvents()


    def pbar2_setmax(self, total2):
        self.progressbar2.setMaximum(total2)
