from time import time_ns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import igraph as ig
from Database import Database
from DataTypes.Partition import Partition
from DataTypes.Microservice import Microservice
import os


class MANETOptiServ(object):


    """An heuristic method for deploying services in a MANET network

    This class contains the implementation of the heuristic MANETOptiServe.
    The algorithm needs the list of microservices (RAM, CPU and replication
    index), the heatmap of each microservice and the computing capabilities
    of each UAV. It returns the best possible deployment scheme that 
    minimices the global latency and that satisfies the restrictions. 
    """

    def __init__(self, variation: str='globalLatency') -> None:
    
        """This constructor loads the configuration from the Database 
        singleton"""
        
        self.__scenario = Database().scenario
        self.__variation = variation
        self.calcShortestPathMatrix()
        
    def calcShortestPathMatrix(self) -> None:

        """Create an auxiliar data structure to hold the info about the
         paths"""
         
        self.__adjMatrix : np.ndarray = np.zeros(
            (len(self.__scenario.uavList), len(self.__scenario.uavList)))
                    
        uav : UAV = None
        neighbour  : UAV = None
        for row in range(self.__adjMatrix.shape[0]):
            for value in range(self.__adjMatrix.shape[1]):
                uav = self.__scenario.uavList[row]
                neighbour = self.__scenario.uavList[value]

                if (abs(uav.position[0] - neighbour.position[0]) <= 1
                    and abs(uav.position[1] - neighbour.position[1]) <= 1):
                    # Is an adjacent matrix element 
                    self.__adjMatrix[row][value] = 1
        
        # Calculate shortest path Matrix
        graphNodes = [uav.id for uav in self.__scenario.uavList]
        a = pd.DataFrame(self.__adjMatrix, 
                         index=graphNodes,
                         columns=graphNodes)
        g = ig.Graph.Adjacency(a)
        
        
        self.__spMatrix : np.ndarray = \
            [g.get_shortest_paths(v, output = "epath") for v in g.vs]
        self.__spMatrix = np.array(
            [[ len(column) for column in row] for row in self.__spMatrix])
     
    def solve(self) -> None:
        
        """Given the scenario provide a solution
        
        Steps:
        1. Sort microservices by replication index in ascending order.
        2. Calculate all the final partitions (in case that there are
           ms with multiple instances). Uses density based clustering.
        3. Measure the cost of deploying the ms in each UAV
           (Path cost * heat).  
        4. Deploy the ms instance in each partition (they are already
           ordered). In the case of collision, choose the second best
           option.
        5. Return the deployment scheme.
        """
        
        t = time_ns()  # Save the start time to measure run time
        partitions: list = []
        costs: list = []
        bestCost: list = []
        # Sort microservices by replication index in ascending order
        microservices = self.__scenario.microserviceList
        microservices = sorted(
            microservices, 
            key=lambda ms: ms.replicationIndex)
        # Make Partitions for the microservices with R > 1
#        [print(uav) for uav in self.__scenario.uavList]
        for microservice in microservices:
#            print(f'Microservice {microservice.id}')
            if (microservice.replicationIndex > 1):
                X = []
                weights = []
                for i, row in enumerate(microservice.heatmap):
                    for j, cell in enumerate(row):
                         if (cell > 0.):
                            X.append([i, j])
                            weights.append(cell)
                kmeans = KMeans(
                    n_clusters=microservice.replicationIndex,
                    random_state=42,
                    init='k-means++',
                    n_init=100)
                labels: list = kmeans.fit_predict(X, weights)
                for i in range(microservice.replicationIndex):
                    filterFunction = lambda uav: uav[1] == i
                    uavList: list = list(filter(filterFunction,
                                                zip(X,labels)))
#                    print(f'\tPartition {uavList}')                    
                    partitions.append(self.__buildPartition(
                        microservice,
                        uavList))
            else:
                partitions.append(Partition(microservice, 
                                            self.__scenario.uavList,
                                            self.__variation))
        deployList = []
        for partition in partitions:
            deployList.append(
                partition.calculateBestUAVtoDeploy(self.__spMatrix))
        # TODO for costList in auxCosts: deploy instance
        #print(len(deployList))
        isDeployed: bool = False
        
        for partition in partitions:
            partition.deployMicroservice()
        
        #[print(uav) for uav in self.__scenario.uavList]               


        elapsed_time = time_ns() - t 
        print(f'Run time (solving time): {elapsed_time} ns')
        self.solutionToCSV()
        
    def __buildPartition(self, ms: Microservice, uavList: list) -> Partition:
        
        """Creates a partition given the cluster positions and the
        microservice"""
        
        finalUavList = []

        for pseudoUAV in uavList:
            filterFunc = lambda uav: uav.position[0] == pseudoUAV[0][0] \
                                     and uav.position[1] == pseudoUAV[0][1]

            uav: UAV = list(filter(filterFunc,
                                   self.__scenario.uavList))[0]

            finalUavList.append(uav)
        if (len(finalUavList) != len(uavList)): print('Error') 
        return Partition(ms, finalUavList, self.__variation)
        
    def solutionToCSV(self, output_file: str = 'out.csv'):
        
        """ Stores the solution in a .csv file
        
        The file containes the following information about the UAVs:
        1. The id.
        2. The microservices deployed.
        3. The number of hops to the closest server of each 
           microservice.
        4. The adjacency list
        """

        csvList = []
        for i, uav in enumerate(self.__scenario.uavList):
            csvRow = []
            csvRow.append(uav.id)
            for ms in self.__scenario.microserviceList:
                if ms in (uav.microservices):
                    csvRow.append(1)
                else:
                    csvRow.append(0)
            for ms in self.__scenario.microserviceList:
                heat: int = ms.heatmap[uav.position[0]][uav.position[1]]
                csvRow.append(heat)
            jumpsList = []
            for j, ms in enumerate(self.__scenario.microserviceList):
                if (ms.heatmap[uav.position[0], uav.position[1]] == 0 ):
                    jumpsList.append(-1)
                else:
                    best_neigh = 999
                    
                    for k, uav2 in enumerate(self.__scenario.uavList):
                        if ms in (uav2.microservices):
                            current_neigh = self.__spMatrix[i][k]


                            if (current_neigh < best_neigh):
                                best_neigh = current_neigh

                    jumpsList.append(best_neigh)

            [csvRow.append(jump) for jump in jumpsList]
            [csvRow.append(adj) for adj in self.__adjMatrix[i]]
            csvList.append(csvRow)
        columnList = ['drone']
        for m in self.__scenario.microserviceList:
            columnList.append(m.id)
        for m in self.__scenario.microserviceList:
            columnList.append(f'heat_{m.id}')
        for m in self.__scenario.microserviceList:
            columnList.append(f'jumps_{m.id}')
        for d in self.__scenario.uavList:
            columnList.append(f'adj_{d.id}')
        result_df = pd.DataFrame(csvList, columns=columnList)
        if (not os.path.exists('../Solutions')): os.makedirs('../Solutions')
        result_df.to_csv(
            (
                f'../Solutions/{self.__scenario.scenarioName}'
                f'_MANETOptiServ_'
                f'{self.__variation}.csv'
            ),
            sep=',',
            index=False)
        
         
