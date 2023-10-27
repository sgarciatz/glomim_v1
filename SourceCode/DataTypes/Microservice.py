import numpy as np

class Microservice(object):

    def __init__(self, msId: str, ramRequirement: float = 1, cpuRequirement: float = 1, replicationIndex: int = 1,  heatmap: np.array = None) -> None:
        self.__id: str               = msId
        self.__ramRequirement: float = ramRequirement
        self.__cpuRequirement: float = cpuRequirement
        self.__replicationIndex: int = replicationIndex
        self.__heatmap: np.array     = heatmap

    @property
    def id(self) -> str:
        return self.__id

    @id.setter
    def id(self, newId) -> None:
        self.__id = newId
        
    @property
    def ramRequirement(self) -> float:
        return self.__ramRequirement
    
    @ramRequirement.setter
    def ramRequirement(self, newRamRequirement: float) -> None:
        self.__ramRequirement = newRamRequirement
    
    @property
    def cpuRequirement(self) -> float:
        return self.__cpuRequirement
        
    @cpuRequirement.setter
    def cpuRequirement(self, newCpuRequirement: float) -> None:
        self.__cpuRequirement = newCpuRequirement

    @property
    def replicationIndex(self) -> float:
        return self.__replicationIndex
        
    @replicationIndex.setter
    def replicationIndex(self, newReplicationIndex: float) -> None:
        self.__replicationIndex = newReplicationIndex
       
    @property 
    def heatmap(self) -> np.array:
        return self.__heatmap
        
    @heatmap.setter
    def heatmap(self, newHeatmap: np.array) -> None:
        self.__heatmap = newHeatmap

    def __str__(self) -> str:
        heatmapString = ''
        if (self.__heatmap is not None):
            for row in self.__heatmap:
                heatmapString += f'\n\t{row}'
              
        return f'Microservice: {self.__id}\n\t-RAM requirement: {self.__ramRequirement}\n\t-CPU requirement: {self.__cpuRequirement}\n\t-Heatmap:{heatmapString}'
        
    def toJSON(self) -> dict:
        json = {
            'microserviceId' : self.__id,
            'ramRequirement' : self.__ramRequirement,
            'cpuRequirement' : self.__cpuRequirement,
            'replicationIndex' : self.__replicationIndex,            
            'heatmap'        :  [[heatValue for heatValue in row] for row in self.__heatmap]
            }
        return json
