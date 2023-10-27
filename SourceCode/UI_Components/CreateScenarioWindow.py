from PySide6 import QtWidgets
from .CreateScenario import Ui_CreateScenario
import pathlib
from DataTypes.Scenario import Scenario
from Database import Database
class CreateScenarioWindow(QtWidgets.QDialog, Ui_CreateScenario):
    def __init__(self):
        super(CreateScenarioWindow, self).__init__()
        self.setupUi(self)
        self.button_searchFiles.pressed.connect(self.openFileBrowser)
        self.buttonBox.accepted.connect(self.createScenario)
        self.buttonBox.rejected.connect( lambda : self.close())
    def openFileBrowser(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/santiago/Documents/Trabajo/Workspace/GLOMIM/glomim_v1/AuxImages/',"Image files (*.jpg *.gif *.svg *.png)")
        self.plainTextEdit_refImage.clear()
        self.plainTextEdit_refImage.insertPlainText(fname[0])

    def createScenario(self):
        dirPath = pathlib.Path().resolve().parent.parent.parent.absolute() / 'InputScenarios'
        filePath = dirPath / self.plainTextEdit_scenarioName.toPlainText()

        if (filePath.exists()):
            return
        
        scenarioName: str           = self.plainTextEdit_scenarioName.toPlainText()

        if (not self.plainTextEdit_height.toPlainText().isdigit() or not self.plainTextEdit_width.toPlainText().isdigit()): return
        scenarioShape: list[int]    = [int(self.plainTextEdit_height.toPlainText()), int(self.plainTextEdit_width.toPlainText())]
        
        scenarioReferenceImage: str = self.plainTextEdit_refImage.toPlainText()
        if (not pathlib.Path(scenarioReferenceImage).exists()): return
        
        scenario = None
        if ( scenarioName != '' and scenarioShape[0] > 0 and scenarioShape[1] > 0):
            scenario = Scenario(scenarioName,
                                scenarioShape,    
                                scenarioReferenceImage, 
                                [], 
                                [])    
                 
            try:
                Database().scenario = scenario 
            except:
                print(scenario)
                Database(scenario) # First time (initialize singleton)
            finally:
                self.close()


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication, QMainWindow
    import sys
    class TestingWindow(QMainWindow, Ui_CreateScenario):
        def __init__(self):
            super(TestingWindow, self).__init__()
            self.setupUi(self)


    app = QApplication(sys.argv)

    window = TestingWindow()
    window.show()

    app.exec()
