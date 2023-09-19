from UI_Components.MainApplicationWindow import MainApplicationWindow
import sys

from PySide6 import (QtCore, QtWidgets)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    mainWindow = MainApplicationWindow()

    mainWindow.show()

    app.exec()   
