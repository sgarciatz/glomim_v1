from PySide6 import QtCore, QtGui, QtWidgets
from DataTypes.UAV import UAV
from Database import Database
from DataTypes.Scenario import Scenario
from .UAVInfoWindow import UAVInfoWindow


class DeploymentLabel(QtWidgets.QLabel):


    """A Label to hold information about the microservices deployed in
    an UAV"""

    def __init__(self, position: list[int], shape: QtCore.QSize) -> None:

        """Initialize the label with the info about the microservices
        deployed on it"""
        super().__init__()
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.setupMenu)
        self.__position: list[int] = position
        self.__shape: QtCore.QSize = shape
        self.__scenario: Scenario = Database().scenario
        self.__uav: UAV = next(
            filter(lambda uav: uav.position[0] == self.__position[0]
                               and uav.position[1] == self.__position[1],
                   self.__scenario.uavList), None)

    def setupMenu(self) -> None:

        """Prepare the menu entries and the corresponding actions"""
        if (self.__uav == None): return
        menu = QtWidgets.QMenu()
        showUAV: QtWidgets.QAction = menu.addAction('Show UAV')
        showUAV.triggered.connect(self.showUAV)
        menu.exec_(QtGui.QCursor.pos())

    def showUAV(self) -> None:

        """ Display a new window that shows the UAV with the deployed
        microservices"""

        self.__uavInfoWindow = UAVInfoWindow(self.__uav)
        self.__uavInfoWindow.exec()

    def getTextForLabel(self) -> None:

        """ Returns the formatted string that must be shown in the
        UI"""

        string = ''
        if (self.__uav != None):
            for ms in self.__uav.microservices:
                string += ms.id[:6] + ', '
        return string[:-2]
