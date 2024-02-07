from PySide6 import QtCore, QtGui, QtWidgets
from DataTypes.UAV import UAV
from Database import Database
from DataTypes.Scenario import Scenario
from .UAVInfoWindow import UAVInfoWindow
from DataTypes.Microservice import Microservice

class MicroserviceLabel(QtWidgets.QLabel):
    
    def __init__(self, position: list[int], microservice: Microservice, shape: QtCore.QSize):
        super().__init__()
        self.__position: list[int] = [position[0], position[1]]
        self.__scenario: Scenario = Database().scenario
        uavList = self.__scenario.uavList        
        self.__microservice: Microservice = microservice       
         
        self.__shape: QtCore.QSize = shape
        self.mousePressEvent = self.modifyHeatValue

        
    def modifyHeatValue(self, event) -> None:
        heatValue : int = self.__microservice.heatmap[self.__position[0]][self.__position[1]]
        if (event.button() == QtCore.Qt.LeftButton):
            self.__microservice.heatmap[self.__position[0]][self.__position[1]] = (heatValue + 1) % 6
            
            pixmap: QtGui.QPixmap = QtGui.QPixmap(self.__scenario.backgroundImg)
            croppingSize: QtCore.QSize = QtCore.QSize(self.__shape.width() / self.__scenario.shape[1], self.__shape.height() / self.__scenario.shape[0])
            croppedPixmap = QtGui.QPixmap(self.__scenario.backgroundImg).scaled(self.__shape).copy(self.__position[1] * croppingSize.width(), self.__position[0] * croppingSize.height(), croppingSize.width(), croppingSize.height())
            backgroundImg : QtGui.QImage = croppedPixmap.toImage()
            heatImg : QtGui.QImage = QtGui.QPixmap(f'../AuxImages/heatmaps/heatmap{int(self.__microservice.heatmap[self.__position[0]][self.__position[1]])}.png').toImage()
            painter: QtGui.QPainter = QtGui.QPainter(backgroundImg)
            painter.drawImage(backgroundImg.rect(), heatImg)
            painter.end()
            croppedPixmap = QtGui.QPixmap.fromImage(backgroundImg)
            self.setPixmap(croppedPixmap)            
                
                
                
                
                
                
                
                
    
