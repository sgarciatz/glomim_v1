from pathlib import Path
import numpy as np
import json
from .UAV import UAV
from .Microservice import Microservice

class Scenario(object):

    def __init__(self, scenarioName: str,
                 shape: list[int],
                 backgroundImg: str = None, 
                 uavList: list[UAV] = [],
                 microserviceList: list[Microservice] = []) -> None:
                 
        self.__scenarioName      = scenarioName 
        self.__shape             = shape
        self.__backgroundImg     = backgroundImg
        self.__uavList           = uavList
        self.__microserviceList  = microserviceList

    @classmethod
    def loadJSON(cls, inputFile: Path) -> None:
        inputData = json.load(open(inputFile))
        uavList: list[UAV] = []
        for uav in inputData['uavList']:
                    uavList.append(UAV(uav['uavId'], uav['position'], uav['ramCapacity'], uav['ramAllocated'], 
                                       uav['cpuCapacity'], uav['cpuAllocated'], uav['microservices']))
        msList: list[Microservice] = []
        for ms in inputData['microserviceList']:
                msList.append(Microservice(ms['microserviceId'], ms['ramRequirement'], ms['cpuRequirement'], ms['replicationIndex'], np.array(ms['heatmap'])))

        return cls(inputData['scenarioName'], inputData['shape'], inputData['backgroundImg'], uavList, msList) 

    def uavPlacement(self) -> np.array:
        uavPlacementMatrix = np.zeros((self.shape[0], self.shape[1]), dtype=int)
        for uav in self.uavList:
            uavPlacementMatrix[uav.position[0]][uav.position[1]] = 1
        
        return uavPlacementMatrix
        
    @property
    def scenarioName(self) -> str:
        return self.__scenarioName
        
    @scenarioName.setter
    def scenarioName(self, newScenarioName: str) -> None:
        self.__scenarioName = newScenarioName
           
    @property
    def backgroundImg(self) -> str:
        return self.__backgroundImg
        
    @property
    def shape(self) -> str:
        return self.__shape
        
    @property
    def scenarioName(self) -> str:
        return self.__scenarioName
    
    
    @property
    def uavList(self) -> list[UAV]:
        return self.__uavList
        
    @uavList.setter
    def uavList(self, newUAVList: list[UAV]) -> None:
        self.__uavList = newUAVList
        
    @property
    def microserviceList(self) -> list[UAV]:
        return self.__microserviceList
        
    @microserviceList.setter
    def microserviceList(self, newMicroserviceList: list[UAV]) -> None:
        self.__microserviceList = newMicroserviceList
        
    def toJSON(self) -> str:
        
        json: dict = {
                'scenarioName'     : self.__scenarioName,
                'shape'            : [self.__shape[0], self.__shape[1]],
                'backgroundImg'    : self.__backgroundImg,
                'uavList'          : [uav.toJSON() for uav in self.__uavList],
                'microserviceList' : [ms.toJSON() for ms in self.__microserviceList]
            }
        return json
