from PySide6 import QtCore, QtGui, QtWidgets
from .AddMicroservice import Ui_AddMicroservice
from Database import Database
from DataTypes.Scenario import Scenario
from DataTypes.UAV import UAV
from DataTypes.Microservice import Microservice
import numpy as np

class AddMicroserviceWindow(QtWidgets.QDialog, Ui_AddMicroservice):
    def __init__(self):
        super(AddMicroserviceWindow, self).__init__()
        self.setupUi(self)
        self.buttonBox.accepted.connect(self.createMicroservice)
        self.buttonBox.rejected.connect( lambda : self.close())
        self.__scenario = Database().scenario
        
    def createMicroservice(self):
        msId  : str   = self.plainTextEdit_microserviceId.toPlainText()
        msRam : float = self.plainTextEdit_ramRequirement.toPlainText()
        msCpu : float = self.plainTextEdit_cpuRequirement.toPlainText()
        msHeatmap: np.array = np.zeros((self.__scenario.shape[0], self.__scenario.shape[1]))
        self.__scenario.microserviceList.append(Microservice(msId, msRam, msCpu, msHeatmap))
        self.close()

