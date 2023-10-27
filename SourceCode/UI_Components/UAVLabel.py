from PySide6 import QtCore, QtGui, QtWidgets
from DataTypes.UAV import UAV
from Database import Database
from DataTypes.Scenario import Scenario
from .UAVInfoWindow import UAVInfoWindow

class UAVLabel(QtWidgets.QLabel):
    
    def __init__(self, position: list[int]):
        super().__init__()
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.setupMenu)    
        self.__position: list[int] = [position[0], position[1]]

        self.__scenario: Scenario = Database().scenario
        uavList = self.__scenario.uavList      
        self.__uav: UAV = None        
        for uav in uavList:
            if (uav.position[0] == self.__position[0] and uav.position[1] == self.__position[1]):
                self.__uav = uav

    def setupMenu(self):
        menu = QtWidgets.QMenu()
        if (self.__uav == None):
            addUAV: QtWidgets.QAction = menu.addAction('Add UAV')
            addUAV.triggered.connect(self.addUAV)
        else:
            removeUAV: QtWidgets.QAction = menu.addAction('Remove UAV')
            removeUAV.triggered.connect(self.removeUAV)
            
        showInfo: QtWidgets.QAction = menu.addAction('Show info')
        showInfo.triggered.connect(self.showInfo)
        menu.exec_(QtGui.QCursor.pos())
        
        
    def addUAV(self) -> None:
        uavList = self.__scenario.uavList
        
        uavId = 'uav_'
        if (self.__position[0] < 10): uavId += '0'
        uavId += str(self.__position[0])
        uavId += '_'
        if (self.__position[1] < 10): uavId += '0'
        uavId += str(self.__position[1])
        
        newUAV: UAV = UAV(uavId, self.__position)
        
        uavList.append(newUAV)
        self.__uav = newUAV
        
        pixmap: QtGui.QPixmap = QtGui.QPixmap(self.__scenario.backgroundImg)
        shape: QtCore.QSize = QtGui.QPixmap(self.__scenario.backgroundImg).size()
        croppingSize: QtCore.QSize = QtCore.QSize(shape.width() / self.__scenario.shape[1], shape.height() / self.__scenario.shape[0])
        croppedPixmap = QtGui.QPixmap(self.__scenario.backgroundImg).copy(self.__position[1] * croppingSize.width(), self.__position[0] * croppingSize.height(), croppingSize.width(), croppingSize.height())
        backgroundImg : QtGui.QImage = croppedPixmap.toImage()
        uavImg : QtGui.QImage = QtGui.QPixmap('/home/santiago/Documents/Trabajo/Workspace/GLOMIM/glomim_v1/AuxImages/uav.png').toImage()
        painter: QtGui.QPainter = QtGui.QPainter(backgroundImg)
        painter.drawImage(backgroundImg.rect(), uavImg)
        painter.end()
        croppedPixmap = QtGui.QPixmap.fromImage(backgroundImg)

        self.setPixmap(croppedPixmap)
        
    def removeUAV(self) -> None:
        uavList = self.__scenario.uavList
        print(uavList)
        for uav in uavList:
            if (uav.id == self.__uav.id):
                uavList.remove(uav)
        print(uavList)
        self.__uav = None
        

        pixmap: QtGui.QPixmap = QtGui.QPixmap(self.__scenario.backgroundImg)
        shape: QtCore.QSize = QtGui.QPixmap(self.__scenario.backgroundImg).size()
        croppingSize: QtCore.QSize = QtCore.QSize(shape.width() / self.__scenario.shape[1], shape.height() / self.__scenario.shape[0])
        croppedPixmap = QtGui.QPixmap(self.__scenario.backgroundImg).copy(self.__position[1] * croppingSize.width(), self.__position[0] * croppingSize.height(), croppingSize.width(), croppingSize.height())
        self.setPixmap(croppedPixmap)
        
    def showInfo(self) -> None:
        for uav in self.__scenario.uavList:
            if (uav.position[0] == self.__position[0] and uav.position[1] == self.__position[1]):
                self.uavInfoWindow = UAVInfoWindow(uav)
                self.uavInfoWindow.exec()
                          
                
                
                
                
                
                
                
                
    
