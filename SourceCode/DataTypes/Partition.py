import numpy as np
from .UAV import UAV
from .Microservice import Microservice
from Database import Database

class Partition(object):


    """A subdivision of the scenario that is served by one ms instance.
    
    This class represents a Partition of the set of partitions of a ms.
    If any ms have a replication index greater than 1, then a partition
    for each instance is created. 
    """
    
    def __init__(self, 
                 ms: Microservice,
                 uavList: list[UAV],
                 metric: str = 'fairness') -> None:
    
        """Initializes the UAVs and the ms of the partition""" 
    
        self.__uavList: list[UAV] = uavList
        self.__ms: Microservice = ms
        self.__metric = metric
    def calculateBestUAVtoDeploy(self, spMatrix: np.ndarray) -> None:
        
        """Calculates the cost of deploying the instance on each UAV
        and returns a ascending ordered list of UAVs"""
        
        costs_list: list = []
        for srcUAV in self.__uavList:
            cost: int = 0
            indexSrcUAV: int = Database().scenario.uavList.index(srcUAV)
            if (self.__metric == 'fairness'):
                costs_list.append((srcUAV, 
                                   self.calcMaxCost(srcUAV, spMatrix)))
            elif (self.__metric == 'globalLatency'):
                costs_list.append((srcUAV,
                                   self.calcCostSum(srcUAV, spMatrix)))
            else:
                costs_list.append(srcUAV, 
                                  self.calcMaxCost(srcUAV, spMatrix))
        sortedUAVs = sorted(costs_list, key= lambda cost: cost[1])
        self.__uavList = [uav for uav, _ in sortedUAVs]
        return sortedUAVs
    
    def calcMaxCost(self, srcUAV: UAV, spMatrix: np.ndarray) -> int:

        """Calculates the worst cost of serving the ms to any UAV of 
        the partition"""

        cost: int = 0
        maxCost: int = 0
        indexSrcUAV: int = Database().scenario.uavList.index(srcUAV)        
        for dstUAV in self.__uavList:
            indexDstUAV: int = Database().scenario.uavList.index(dstUAV)
            path_cost: int = spMatrix[indexSrcUAV][indexDstUAV]
            heat = self.__ms.heatmap[dstUAV.position[0]][dstUAV.position[1]]
            if heat != 0:
                heat = 5 - heat
            cost = path_cost * heat
            if (cost > maxCost):
                maxCost = cost
        return maxCost
        
    def calcCostSum(self, srcUAV: UAV, spMatrix: np.ndarray) -> int:

        """Calculates the sum of the costs of serving the ms to all the
        UAVs of the partition"""

        cost: int = 0
        indexSrcUAV: int = Database().scenario.uavList.index(srcUAV)        
        for dstUAV in self.__uavList:
            indexDstUAV: int = Database().scenario.uavList.index(dstUAV)
            path_cost: int = spMatrix[indexSrcUAV][indexDstUAV]
            heat = self.__ms.heatmap[dstUAV.position[0]][dstUAV.position[1]]
            if heat != 0:
                heat = 5 - heat
            cost += path_cost * heat
        return cost
    def deployMicroservice(self) -> None:
    
        """ Deploy the microservice in the best UAV Possible"""
        
        isDeployed: bool = False
        for uav in self.__uavList:
            isDeployed = uav.deployMicroservice(self.__ms)
            if (isDeployed): break
    
    def __str__(self) -> str:

        """Returns a string representation of the partition"""
        string = f"Partition of {self.__ms.id} composed of:"
        for uav in self.__uavList:
            string += f'\n\t{uav.id}'
        return string
