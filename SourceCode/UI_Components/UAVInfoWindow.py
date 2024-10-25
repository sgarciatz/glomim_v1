from PySide6 import QtCore, QtGui, QtWidgets
from .UAVInfo import Ui_UAVInfo
from Database import Database
from DataTypes.Scenario import Scenario
from DataTypes.UAV import UAV

class UAVInfoWindow(QtWidgets.QDialog, Ui_UAVInfo):

    def __init__(self, uav: UAV = None):
        super(UAVInfoWindow, self).__init__()
        self.setupUi(self)
        self.__uav: UAV = uav

        self.buttonBox.accepted.connect(self.modifyUAV)
        self.buttonBox.rejected.connect(self.closeWindow)

        self.plainTextEdit_uavId.setPlainText(uav.id)
        self.plainTextEdit_row.setPlainText(str(uav.position[0]))
        self.plainTextEdit_column.setPlainText(str(uav.position[1]))
        self.plainTextEdit_ramCapacity.setPlainText(str(uav.ramCapacity))
        self.plainTextEdit_ramAllocated.setPlainText(str(uav.ramAllocated))
        self.plainTextEdit_cpuCapacity.setPlainText(str(uav.cpuCapacity))
        self.plainTextEdit_cpuAllocated.setPlainText(str(uav.cpuAllocated))
        ms: str = ', '.join([ms.id for ms in uav.microservices])
        self.plainTextEdit_microservices.setPlainText(ms)

    def modifyUAV(self) -> None:
        uavId: str               = self.plainTextEdit_uavId.toPlainText()
        uavPosition: list[int]   = \
            [int(self.plainTextEdit_row.toPlainText()),
             int(self.plainTextEdit_column.toPlainText())]
        ramCapacity: float       = float(
            self.plainTextEdit_ramCapacity.toPlainText())
        ramAllocated: float      = float(
            self.plainTextEdit_ramAllocated.toPlainText())
        cpuCapacity: float       = float(
            self.plainTextEdit_cpuCapacity.toPlainText())
        cpuAllocated: float      = float(
            self.plainTextEdit_cpuAllocated.toPlainText())
        microservices: list[str] = \
            [ms for ms in \
                self.plainTextEdit_microservices.toPlainText().split(', ')]

        self.__uav.id            = uavId
        self.__uav.position      = uavPosition
        self.__uav.ramCapacity   = ramCapacity
        self.__uav.ramAllocated  = ramAllocated
        self.__uav.cpuCapacity   = cpuCapacity
        self.__uav.cpuAllocated  = cpuAllocated
        self.__uav.microservices = microservices

        print(Database().scenario.uavList)
        self.closeWindow()

    def closeWindow(self) -> None:
        self.close()
