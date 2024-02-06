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
from .DeploymentLabel import DeploymentLabel
from DataTypes.Scenario import Scenario
from DataTypes.Microservice import Microservice
from .MicroserviceLabel import MicroserviceLabel
from Solvers.GLOMIP import GLOMIP
from Solvers.MANETOptiServ import MANETOptiServ
import json
import pathlib
from functools import partial

class MainApplicationWindow(QtWidgets.QMainWindow, Ui_MainApplication):
    
    
    def __init__(self):
        super(MainApplicationWindow, self).__init__()
        self.setupUi(self)
        self.actionCreate_new_Scenario.triggered.connect(
            self.createNewScenario)
        self.actionLoad_existing_Scenario.triggered.connect(
            self.loadExistingScenario)
        self.actionSave_current_Scenario.triggered.connect(self.saveScenario)
        self.actionScenario_View.triggered.connect(self.displayScenario)
        self.actionUAV_View.triggered.connect(self.displayUAVs)
        self.actionAdd_Microservice.triggered.connect(self.addMicroservice)
        self.actionGLOSIP.triggered.connect(self.solveWithGLOSIP)
        self.actionGLOMIP.triggered.connect(self.solveWithGLOMIP)
        self.actionMANETOptiServGlobLat.triggered.connect(
            self.solveWithMANETOptiServGlobLat)
        self.actionMANETOptiServFairness.triggered.connect(
            self.solveWithMANETOptiServFairness) 
        self.actionDeployment_View.triggered.connect(self.displayDeployment)
        
        
        
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
            scenariosDir : pathlib.Path = pathlib.Path().resolve().parent.absolute() / 'InputScenarios'
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
        print(shape)
        # Scale the image to fit the window size
        shape = self.scaleImageToFitGrid(shape)
        print(shape)
        
        # Calculate the size of each Pixmap
        croppingSize: QtCore.QSize = QtCore.QSize(shape.width() / self.scenario.shape[1], shape.height() / self.scenario.shape[0])

        for row in range(self.scenario.shape[0]):
            for column in range(self.scenario.shape[1]):
                label = QtWidgets.QLabel()
                croppedPixmap = pixmap.scaled(shape).copy(column * croppingSize.width(), row * croppingSize.height(), croppingSize.width(), croppingSize.height())

                label.resize(croppingSize)
                label.setPixmap(croppedPixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio))
                self.gridLayout.addWidget(label, row, column)
      
    def displayUAVs(self) -> None:

        self.clearDisplay()
         # Retrieve the size of the background image and scale the layout accordingly
        self.scenario = Database().scenario
        pixmap: QtGui.QPixmap = QtGui.QPixmap(self.scenario.backgroundImg)
        shape: QtCore.QSize = QtGui.QPixmap(self.scenario.backgroundImg).size()
        #self.gridLayoutWidget.setGeometry(QtCore.QRect(9, 9, shape.width(), shape.height()))
        
        # Scale the grid to fit the image
        shape = self.scaleImageToFitGrid(shape)
        # Calculate the size of each Pixmap
        croppingSize: QtCore.QSize = QtCore.QSize(shape.width() / self.scenario.shape[1], shape.height() / self.scenario.shape[0])
        
        for row in range(self.scenario.shape[0]):
            for column in range(self.scenario.shape[1]):
                label = UAVLabel([row, column], shape)
                croppedPixmap = pixmap.scaled(shape).copy(column * croppingSize.width(), row * croppingSize.height(), croppingSize.width(), croppingSize.height())
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
        # print(microservice)
        self.clearDisplay()
         # Retrieve the size of the background image and scale the layout accordingly
        self.scenario = Database().scenario
        pixmap: QtGui.QPixmap = QtGui.QPixmap(self.scenario.backgroundImg)
        shape: QtCore.QSize = QtGui.QPixmap(self.scenario.backgroundImg).size()
        #self.gridLayoutWidget.setGeometry(QtCore.QRect(9, 9, shape.width(), shape.height()))
        
        # Scale the grid to fit the image
        shape = self.scaleImageToFitGrid(shape)
        
        # Calculate the size of each Pixmap
        croppingSize: QtCore.QSize = QtCore.QSize(shape.width() / self.scenario.shape[1], shape.height() / self.scenario.shape[0])
        
        for row in range(self.scenario.shape[0]):
            for column in range(self.scenario.shape[1]):
                label = MicroserviceLabel([row, column], microservice, shape)
                croppedPixmap = pixmap.scaled(shape).copy(column * croppingSize.width(), row * croppingSize.height(), croppingSize.width(), croppingSize.height())
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
                
    def displayDeployment(self) -> None:
        self.clearDisplay()
        # Retrieve the size of the background image and scale the 
        # layout accordingly
        self.scenario = Database().scenario
        pixmap: QtGui.QPixmap = QtGui.QPixmap(self.scenario.backgroundImg)
        shape: QtCore.QSize = QtGui.QPixmap(
            self.scenario.backgroundImg).size()
        # Scale the grid to fit the image
        shape = self.scaleImageToFitGrid(shape)
        # Calculate the size of each Pixmap
        croppingSize: QtCore.QSize = QtCore.QSize(
            shape.width() / self.scenario.shape[1],
            shape.height() / self.scenario.shape[0])
        msList = self.scenario.microserviceList
        for row in range(self.scenario.shape[0]):
            for column in range(self.scenario.shape[1]):
                label = DeploymentLabel([row, column], shape)
                croppedPixmap = pixmap.scaled(shape).copy(
                    column * croppingSize.width(),
                    row * croppingSize.height(),
                    croppingSize.width(),
                    croppingSize.height())
                backgroundImg : QtGui.QImage = croppedPixmap.toImage()     
                painter: QtGui.QPainter = QtGui.QPainter(backgroundImg)
                painter.setPen(QtGui.QColor('pink'))
                text = label.getTextForLabel()
                painter.drawText(backgroundImg.rect(), text) 
                painter.end()                
                croppedPixmap = QtGui.QPixmap.fromImage(backgroundImg)
                label.resize(croppingSize)
                label.setPixmap(croppedPixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio))
                self.gridLayout.addWidget(label, row, column)
            
            
            
    def thereIsUAV(self, position: list[int]) -> bool:
        for uav in Database().scenario.uavList:
            if (uav.position[0] == position[0] and uav.position[1] == position[1]):
                return True
        return False
        
    def scaleImageToFitGrid(self, shape : QtCore.QSize) -> QtCore.QSize:
        
        windowsShape: QtCore.QSize = self.frameGeometry()
        widgetShape: QtCore.QSize = self.gridLayoutWidget.frameGeometry()
        
        # First force the grid to be smaller than the windows size
        padding: int = 20
        
        ratio = max((widgetShape.width()) / windowsShape.width(), (widgetShape.height()) / windowsShape.height())

        # If the grid is greater (ratio > 1), then scale down the grid
        if ratio > 1.0:

            self.gridLayoutWidget.setGeometry(QtCore.QRect(padding, padding, (widgetShape.width() * (1/ratio)) - padding, (widgetShape.height() * (1/ratio)) - padding))
            widgetShape = self.gridLayoutWidget.frameGeometry()


        # Find the ratio between the grid and the image
        ratio = min((widgetShape.width()) / shape.width(), (widgetShape.height()) / shape.height())
        # If the image is greater (ratio < 1), then scale down the image
        newShape = shape

        if (ratio < 1.0): 

            newShape = QtCore.QSize(shape.width() * ratio - padding , shape.height() * ratio - padding)
        self.gridLayoutWidget.setGeometry(padding, padding, newShape.width() + padding, newShape.height() + padding)
        return newShape
        
    def solveWithGLOSIP(self) -> None:
        print('Solving with GLOSIP')
        self.clearCurrentDeployment()

        # Setup the solver

        # Solve scenario
        solver.initializeModel()
        # Show Result
        solver.solve()
        
    def solveWithGLOMIP(self) -> None:
        print('Solving with GLOMIP')
        self.scenario.clearUAVs();
        # Setup the solver
        solver = GLOMIP()        
        # Solve scenario
        solver.initializeModel()        
        # Show Result
        solver.solve()
        
    def solveWithMANETOptiServGlobLat(self) -> None:
        print('Solving with MANETOptiServe')
        self.scenario.clearUAVs();
        # Setup the solver
        solver = MANETOptiServ('globalLatency')        
        # Solve scenario
        # Show Result
        solver.solve()
                
    def solveWithMANETOptiServFairness(self) -> None:
        print('Solving with MANETOptiServe')
        self.scenario.clearUAVs();
        # Setup the solver
        solver = MANETOptiServ('fairness')        
        # Solve scenario
        # Show Result
        solver.solve() 
