import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from mip import *
import pandas as pd
import json
from pathlib import Path
from Database import Database
import igraph as ig
from time import time_ns


class GLOMIP(object):


    """The modelization of the problem in a ILP that can be solved by
    GUROBI.
    
    This class loads the information of the scenarios and constructs an
    ILP that solves the problem of multiple instance deployment of
    various microservices over a swarm of resource constrained UAVs. It
    uses GUROBI as the solver.    
    """

    def __init__(self):

        """Load the scenario from the Database singleton"""
            
        #Load configuration from Database 
        self.__scenario = Database().scenario        
        self.calcShortestPathMatrix()
        self.__maxDistance = np.max(self.__spMatrix)
        self.__M = self.__maxDistance + 1
        self.__maxHeat =  np.max(
            [ms.heatmap for ms in self.__scenario.microserviceList])
        # Initialize the model
        self.__model = Model(sense=MINIMIZE, solver_name=GRB)
        self.__model.verbose = 1
        self.__model.threads = -1

    def initializeModel(self):

        """ Create all the decision variables and restrictions
        
        X is subindexed by the microservice and by the UAV. It 
        specifies wether a ms is deployed on a given UAV or not.
        
        Z holds the lowest value to serve a specific ms to a given UAV.
        
        Y is an activation variable to correctly give a value to Z. It
        is used so GUROBI can choose the best ms instance for each Z.
        """

        msList = self.__scenario.microserviceList
        uavList = self.__scenario.uavList
        upperLimit = self.__maxDistance
        lowerLimit = 0
        for uav in uavList:
            for microservice in msList:
                    self.__model.add_var(
                        f'x_{microservice.id},uav{uav.id}',
                        var_type=BINARY)
                    self.__model.add_var(
                        f'z_{microservice.id},uav{uav.id}',
                        var_type=INTEGER,
                        lb=lowerLimit,
                        ub=upperLimit)
                    for uav2 in uavList:
                        self.__model.add_var(
                            f'y_{microservice.id},uav{uav.id},uav{uav2.id}',
                            var_type=BINARY)
        
        decisionVariables = np.array(self.__model.vars)
        X = decisionVariables[0::2+len(uavList)].reshape(
            (len(uavList),
             len(msList)))
        Z = decisionVariables[1::2+len(uavList)].reshape(
            (len(uavList), 
             len(msList)))
        Y = decisionVariables.reshape(
            (len(uavList),
             len(msList), -1))[:,:,2:]
        # Add the replication index contraints
        X_groupedByMicroservice = X.T
        zippedMsX = zip(msList, X_groupedByMicroservice)
        for ms, xAux in zippedMsX:
            constraint = ms.replicationIndex - xsum(xAux) == 0
            constraint = self.__model.add_constr(
                constraint,
                name=f'All instances of {ms.id} are deployed')
        # Add the UAVs CPU constraints
        X_groupedByUav = X
        for uav, xAux in zippedMsX:
            cpuReqs = xsum([x_ms * msList[i].cpuRequirement \
                            for i, x_ms in enumerate(xAux)])
            constraint = uav.cpuCapacity - cpuReqs >= 0
            self.__model.add_constr(
                constraint, 
                name=f'CPU Capacity of {uav.id} must be not surpased')
        # Add the UAVs RAM constraints
        for uav, xAux in zip(uavList, X_groupedByUav):
            ramReqs = xsum([x_ms * msList[i].ramRequirement \
                            for i, x_ms in enumerate(xAux)])
            constraint = uav.ramCapacity - ramReqs >= 0
            self.__model.add_constr(
                constraint,
                name=f'RAM Capacity of {uav.id} must be not surpased')
        # Add the top boundaries for z (i,j,i') constraint to ensure
        # tightness
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for ii in range(X.shape[0]):
                    undeploydPenalty = (1 - X[ii][j]) * self.__M
                    pathCost = self.__spMatrix[i][ii] * X[ii][j]
                    yActivation = (9999 * Y[i][j][ii])
                    constraint = \
                        Z[i][j] >= undeploydPenalty + pathCost - yActivation
                    constraint = self.__model.add_constr(
                        constraint,
                        name=(f'Ensure tightness: uav '
                              f'{uavList[i].id}, microservice'
                              f' {msList[j].id},'
                              f' uav {uavList[ii].id}'))                         
        # Add constraints to deactivate all y except one for each pair
        # of x and z
        for i, uav in enumerate(uavList):
                for j, ms in enumerate(msList):
                    constraint = xsum(Y[i][j]) - (len(uavList)) + 1 == 0
                    self.__model.add_constr(
                        constraint,
                        name=(f'Ensure one y for uav {uav.id}'
                              f' and microservice {ms.id}'))

        # Add the objetive function       
        self.__model.objective = minimize(
            xsum([ms.heatmap[uav.position[0]][uav.position[1]] * Z[i][j]\
                  for j, ms in enumerate(msList)\
                  for i, uav in enumerate(uavList)]))
    def solve(self):

        """Execute the solver and return the results"""
        
        t = time_ns()
        status = self.__model.optimize()
        print(status)
        elapsed_time = time_ns() - t 
        print(f'Run time (solving time): {elapsed_time} ns')
        # Retrieve the UAV's where a ms is deployed
        results : np.ndarray = np.array([ var for var in self.__model.vars])
        results = results.reshape((len(self.__scenario.uavList), -1))
        df_data = [{
                        'Decision Variable': var.name, 
                        'Value': var.x
                    } for index, var in enumerate(self.__model.vars)]
        self.solutionToCSV(df_data)
        self.deploy()
        return pd.DataFrame(df_data)

    def deploy(self):

        """Deploy the microservices on the UAVs using the solution
        provided by the solver"""

        msList = self.__scenario.microserviceList
        uavList = self.__scenario.uavList
        result =  [int(decision_variable.x)\
                    for decision_variable in self.__model.vars[:]]
        result = np.array(result)[::2+len(uavList)]
        result = result.reshape((len(uavList), len(msList)))
        for i, row in enumerate(result):
            for j, value in enumerate(row):
                if (value == 1):
                    uavList[i].deployMicroservice(msList[j])
        
        pass
        
    def calcShortestPathMatrix(self) -> None:
        # Create an auxiliar data structure to hold the info about the paths
        # Its size is proportional to the number of uav in the scenario
        # spMatrix : np.ndarray  = np.zeroes((len(self.__scenario.uavList), len(self.__scenario.uavList)))
        
        # Calculate adjacencyMatrix
        # For each UAV check if it has any neighbours 
        self.__adjMatrix : np.ndarray = np.zeros((len(self.__scenario.uavList), len(self.__scenario.uavList)))

        uav : UAV = None
        neighbour  : UAV = None
        for row in range(self.__adjMatrix.shape[0]):
            for value in range(self.__adjMatrix.shape[1]):
                uav = self.__scenario.uavList[row]
                neighbour = self.__scenario.uavList[value]

                if (abs(uav.position[0] - neighbour.position[0]) <= 1 and abs(uav.position[1] - neighbour.position[1]) <= 1):
                    self.__adjMatrix[row][value] = 1
        
        # Calculate shortest path Matrix
        graphNodes = [uav.id for uav in self.__scenario.uavList]
        a = pd.DataFrame(self.__adjMatrix, index=graphNodes, columns=graphNodes)
        g = ig.Graph.Adjacency(a)
        
        
        self.__spMatrix : np.ndarray = [g.get_shortest_paths(v, output = "epath") for v in g.vs]
        self.__spMatrix = np.array([[ len(column) for column in row] for row in self.__spMatrix])
    
    
    
    def solutionToCSV(self, output_file: str = 'out.csv'):
    
        # Prepare a list that for each uav it contains the name of the UAV microservices deployed, the heat of each ms, the number of jumps to the closest UAV to serve each ms and its the adjacency list
        csv_list = []
        decision_variables = np.array([(decision_variable.name, int(decision_variable.x)) for decision_variable in self.__model.vars[:]])[::2+len(self.__scenario.uavList)] #drone -> Xms1, Xms2, Xms3, ..., Xmsn
        

        for i, uav in enumerate(self.__scenario.uavList):

            uav_name = uav.id
            csv_row = [uav.id]
            ms_deployed = [int(decision_variable[1]) for decision_variable in decision_variables[i*len(self.__scenario.microserviceList):i*len(self.__scenario.microserviceList) + len(self.__scenario.microserviceList)]]
            [csv_row.append(ms) for ms in ms_deployed]
            heats = [ms.heatmap[uav.position[0], uav.position[1]] for ms in self.__scenario.microserviceList]
            [csv_row.append(heat) for heat in heats]            
            jumps_list = []
            # For each microservice
            for j, ms in enumerate(self.__scenario.microserviceList):
                if (ms.heatmap[uav.position[0], uav.position[1]] == 0 ):
                    jumps_list.append(-1)
                else:
                    best_neigh = 999
                    for k, dec_var in enumerate(decision_variables[j::len(self.__scenario.microserviceList)]):
                        if (int(dec_var[1]) == 1):
                            current_neigh = self.__spMatrix[i][k]


                            if (current_neigh < best_neigh):
                                best_neigh = current_neigh

                    jumps_list.append(best_neigh)

            [csv_row.append(jump) for jump in jumps_list]                             
            adj_list = self.__adjMatrix[i]
            [csv_row.append(adj) for adj in adj_list]                             
            csv_list.append(csv_row)
        
        # [print(csv_row) for csv_row in csv_list]
        
        column_list = ['drone']
        [column_list.append(m.id) for m in self.__scenario.microserviceList]
        [column_list.append(f'heat_{m.id}') for m in self.__scenario.microserviceList]
        [column_list.append(f'jumps_{m.id}') for m in self.__scenario.microserviceList]
        [column_list.append(f'adj_{d.id}') for d in self.__scenario.uavList]
        # print(len(csv_list[0]), len(column_list)) 
        result_df = pd.DataFrame(csv_list, columns=column_list)
        result_df.to_csv(f'../Solutions/{self.__scenario.scenarioName}.csv', sep=',', index=False)
        
