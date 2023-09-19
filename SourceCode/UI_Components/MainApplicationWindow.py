# This Python file uses the following encoding: utf-8
from PySide6 import QtCore
from PySide6 import QtWidgets
from PySide6 import QtGui
import sys
from .MainApplication import Ui_MainApplication
from .CreateScenarioWindow import CreateScenarioWindow
from .AddMicroserviceWindow import AddMicroserviceWindow
from Database import Database
from .UAVLabel import UAVLabel
from DataTypes.Scenario import Scenario
from DataTypes.Microservice import Microservice
from .MicroserviceLabel import MicroserviceLabel
import json
import pathlib
from functools import partial

class MainApplicationWindow(QtWidgets.QMainWindow, Ui_MainApplication):
    def __init__(self):
        super(MainApplicationWindow, self).__init__()
        self.setupUi(self)
        self.actionCreate_new_Scenario.triggered.connect(self.createNewScenario)
        self.actionLoad_existing_Scenario.triggered.connect(self.loadExistingScenario)
        self.actionSave_current_Scenario.triggered.connect(self.saveScenario)
        self.actionScenario_View.triggered.connect(self.displayScenario)
        self.actionUAV_View.triggered.connect(self.displayUAVs)
        self.actionAdd_Microservice.triggered.connect(self.addMicroservice)
        self.actionGLOSIP.triggered.connect(self.solveWithGLOSIP)
        self.actionGLOMIP.triggered.connect(self.solveWithGLOMIP)
        
    def createNewScenario(self, MainWindow):
        # Open the dialog to create the scenario
        self.createScenarioWindow = CreateScenarioWindow()
        self.createScenarioWindow.exec()
        
        if (Database().scenario is not None):
            self.displayScenario()
            self.reloadMicroservices()
        else:
            print('Could not create new scenario')
            
    def addMicroservice(self):
        self.addMicroserviceWindow = AddMicroserviceWindow()
        self.addMicroserviceWindow.exec()
        self.reloadMicroservices()
        
    def loadExistingScenario(self, MainWindow):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/santiago/Documents/Trabajo/Workspace/GLOMIM/glomim_v1/InputScenarios/',"JSON files (*.json)")

        Database(Scenario.loadJSON(fname[0]))
        
        if (Database().scenario is not None):
            self.displayScenario()
            self.reloadMicroservices()
        else:
            print('Could not load existing scenario')

    def saveScenario(self):
            scenariosDir : pathlib.Path = pathlib.Path().resolve().parent.parent.parent.absolute() / 'InputScenarios'
            scenarioToSave : Scenario   = Database().scenario 
            fileName : str              = scenarioToSave.scenarioName + '.json'
            with open(scenariosDir / fileName, 'wt') as outputFile:
                json.dump(scenarioToSave.toJSON(), outputFile, indent=4)
            
            
    def reloadMicroservices(self) -> None:
        #Retrieve all actions except the first one
        _translate = QtCore.QCoreApplication.translate
        actions : list[QtCore.QAction] = self.menuMicroserviceView.actions()
        [self.menuMicroserviceView.removeAction(action) for action in actions[2:]]
        
        for ms in self.scenario.microserviceList:
            auxAction = QtGui.QAction(parent=self)
            auxAction.setObjectName(f'action{ms.id}')
            self.menuMicroserviceView.addAction(auxAction)
            auxAction.setText(_translate("MainWindow", f'{ms.id}'))
            auxAction.triggered.connect(partial(self.displayMicroservice, ms))
        
    def clearDisplay(self) -> None:
        for i in reversed(range(self.gridLayout.count())): 
            item = self.gridLayout.takeAt(i)
            item.widget().setParent(None)
            self.gridLayout.removeItem(item)
    
    def displayScenario(self) -> None:

        self.clearDisplay()
         # Retrieve the size of the background image and scale the layout accordingly
        self.scenario = Database().scenario
        pixmap: QtGui.QPixmap = QtGui.QPixmap(self.scenario.backgroundImg)
        shape: QtCore.QSize = QtGui.QPixmap(self.scenario.backgroundImg).size()
        self.gridLayoutWidget.setGeometry(QtCore.QRect(9, 9, shape.width(), shape.height()))
        
        # Calculate the size of each Pixmap
        croppingSize: QtCore.QSize = QtCore.QSize(shape.width() / self.scenario.shape[1], shape.height() / self.scenario.shape[0])
        
        for row in range(self.scenario.shape[0]):
            for column in range(self.scenario.shape[1]):
                label = QtWidgets.QLabel()
                croppedPixmap = QtGui.QPixmap(self.scenario.backgroundImg).copy(column * croppingSize.width(), row * croppingSize.height(), croppingSize.width(), croppingSize.height())
                label.resize(croppingSize)
                label.setPixmap(croppedPixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio))
                self.gridLayout.addWidget(label, row, column)
      
    def displayUAVs(self) -> None:

        self.clearDisplay()
         # Retrieve the size of the background image and scale the layout accordingly
        self.scenario = Database().scenario
        pixmap: QtGui.QPixmap = QtGui.QPixmap(self.scenario.backgroundImg)
        shape: QtCore.QSize = QtGui.QPixmap(self.scenario.backgroundImg).size()
        self.gridLayoutWidget.setGeometry(QtCore.QRect(9, 9, shape.width(), shape.height()))
        
        # Calculate the size of each Pixmap
        croppingSize: QtCore.QSize = QtCore.QSize(shape.width() / self.scenario.shape[1], shape.height() / self.scenario.shape[0])
        
        for row in range(self.scenario.shape[0]):
            for column in range(self.scenario.shape[1]):
                label = UAVLabel([row, column])
                croppedPixmap = QtGui.QPixmap(self.scenario.backgroundImg).copy(column * croppingSize.width(), row * croppingSize.height(), croppingSize.width(), croppingSize.height())
                # Check if there exists an UAV in the current tile
                if (self.thereIsUAV([row, column])):
                    backgroundImg : QtGui.QImage = croppedPixmap.toImage()
                    uavImg : QtGui.QImage = QtGui.QPixmap('/home/santiago/Documents/Trabajo/Workspace/GLOMIM/glomim_v1/AuxImages/uav.png').toImage()
                    painter: QtGui.QPainter = QtGui.QPainter(backgroundImg)
                    painter.drawImage(backgroundImg.rect(), uavImg)
                    painter.end()
                    croppedPixmap = QtGui.QPixmap.fromImage(backgroundImg)

                img = croppedPixmap.toImage()
                label.resize(croppingSize)
                label.setPixmap(croppedPixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio))
                self.gridLayout.addWidget(label, row, column)
        
        
    def displayMicroservice(self, microservice: Microservice) -> None:
        print(microservice)
        self.clearDisplay()
         # Retrieve the size of the background image and scale the layout accordingly
        self.scenario = Database().scenario
        pixmap: QtGui.QPixmap = QtGui.QPixmap(self.scenario.backgroundImg)
        shape: QtCore.QSize = QtGui.QPixmap(self.scenario.backgroundImg).size()
        self.gridLayoutWidget.setGeometry(QtCore.QRect(9, 9, shape.width(), shape.height()))
        
        # Calculate the size of each Pixmap
        croppingSize: QtCore.QSize = QtCore.QSize(shape.width() / self.scenario.shape[1], shape.height() / self.scenario.shape[0])
        
        for row in range(self.scenario.shape[0]):
            for column in range(self.scenario.shape[1]):
                label = MicroserviceLabel([row, column], microservice)
                croppedPixmap = QtGui.QPixmap(self.scenario.backgroundImg).copy(column * croppingSize.width(), row * croppingSize.height(), croppingSize.width(), croppingSize.height())
                # Check if there exists an UAV in the current tile
                backgroundImg : QtGui.QImage = croppedPixmap.toImage()
                heatImg : QtGui.QImage = QtGui.QPixmap(f'/home/santiago/Documents/Trabajo/Workspace/GLOMIM/glomim_v1/AuxImages/heatmaps/heatmap{int(microservice.heatmap[row][column])}.png').toImage()
                painter: QtGui.QPainter = QtGui.QPainter(backgroundImg)
                painter.drawImage(backgroundImg.rect(), heatImg)
                painter.end()
                croppedPixmap = QtGui.QPixmap.fromImage(backgroundImg)
                img = croppedPixmap.toImage()
                label.resize(croppingSize)
                label.setPixmap(croppedPixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio))
                self.gridLayout.addWidget(label, row, column)
                

    def thereIsUAV(self, position: list[int]) -> bool:
        for uav in Database().scenario.uavList:
            if (uav.position[0] == position[0] and uav.position[1] == position[1]):
                return True
        return False
        
        
    def clearCurrentDeployment(self) -> None:
        uavList : list[UAV] = self.scenario.uavList
        for uav in uavList:
            uav.microservices = []

    def solveWithGLOSIP(self) -> None:
        print('Solving with GLOSIP')
        self.clearCurrentDeployment()

        # Setup the solver
        
        # Solve scenario
        
        # Show Result

        
    def solveWithGLOMIP(self) -> None:
        print('Solving with GLOMIP')
        self.clearCurrentDeployment()
        
        # Setup the solver
        
        # Solve scenario
        
        # Show Result

