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

class GLOMIP(object):

    def __init__(self):
    
        #Load configuration from Database 
        self.__scenario     = Database().scenario        
        
        # MaxDistanceConst is the Big-M constant, it is a value outside the domain of all parameters
        self.__maxDistance  = len(self.__scenario.uavList) - 1
     
        self.__M            = len(self.__scenario.uavList)
        # MaxHeat is a convenience variable that stores the maximum value of heat of the scenario
        self.__maxHeat      =  max([heatValue for ms in self.__scenario.microserviceList for heatRow in ms.heatmap for heatValue in heatRow])
        # Calculate the shortest path matrix
        self.calcShortestPathMatrix()

        # Initialize the model
        self.__model = Model(sense=MINIMIZE, solver_name=GRB)
        self.__model.verbose = 1
        self.__model.threads = -1
        self.__model.preprocess = 0

        # self.model.max_mip_gap = 0.001
        # self.__model.emphasis = 1
    

    def initializeModel(self):
        # Add the decision variables
        #   x (i,j,k)       -> says if m(k) is deployed in d(i,j)
        #   z (i,j,k)       -> the lowest number of jumps needed to serve d(i,j) the microservice m(k)
        #   y (i,j,k,ii,jj) -> activation/desactivation variable to select the lowest posibility for z(i,j,k)   
        #   U               -> maximun value of Z
        upperLimit = self.__maxHeat * self.__maxDistance
        lowerLimit = 0
        for uav in self.__scenario.uavList:
            for microservice in self.__scenario.microserviceList:
                    self.__model.add_var(f'x_{microservice.id},uav{uav.id}', var_type=BINARY)
                    self.__model.add_var(f'z_{microservice.id},uav{uav.id}', var_type=CONTINUOUS, lb=lowerLimit, ub=upperLimit)
                    for uav2 in self.__scenario.uavList:
                        self.__model.add_var(f'y_{microservice.id},uav{uav.id},uav{uav2.id}', var_type=BINARY)
        
        decisionVariables = np.array(self.__model.vars)#.reshape((len(self.__scenario.uavList), len(self.__scenario.microserviceList), len(self.__scenario.uavList)))
        # X is a matrix where each row represent all the microservices of an UAV 
        X = decisionVariables[0::2+len(self.__scenario.uavList)].reshape((len(self.__scenario.uavList), len(self.__scenario.microserviceList)))

        # Z is a matrix where each row represent the shortests path to reach each microservice of an UAV 
        Z = decisionVariables[1::2+len(self.__scenario.uavList)].reshape((len(self.__scenario.uavList), len(self.__scenario.microserviceList)))

        Y = decisionVariables.reshape((len(self.__scenario.uavList), len(self.__scenario.microserviceList), -1))[:,:,2:]

        # Add the replication index contraints
        X_groupedByMicroservice = X.T
        for ms, xAux in zip(self.__scenario.microserviceList, X_groupedByMicroservice):
            constraint = ms.replicationIndex - xsum(xAux) == 0
            constraint = self.__model.add_constr(constraint, name=f'All instances of {ms.id} are deployed')

        # Add the UAVs CPU constraints
        X_groupedByUav = X
        for uav, xAux in zip(self.__scenario.uavList, X_groupedByUav):
            constraint = uav.cpuCapacity - xsum([x_ms * self.__scenario.microserviceList[i].cpuRequirement for i, x_ms in enumerate(xAux)]) >= 0
            self.__model.add_constr(constraint, name=f'CPU Capacity of {uav.id} must be not surpased')

        # Add the UAVs RAM constraints
        for uav, xAux in zip(self.__scenario.uavList, X_groupedByUav):
            constraint = uav.ramCapacity - xsum([x_ms * self.__scenario.microserviceList[i].ramRequirement for i, x_ms in enumerate(xAux)]) >= 0
            self.__model.add_constr(constraint, name=f'RAM Capacity of {uav.id} must be not surpased')


        # Add the top boundaries for z (i,j,i') constraint to ensure tightness
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for ii in range(X.shape[0]):
                    constraint = Z[i][j] >= (((1-X[ii][j])* self.__M + self.__spMatrix[i][ii] * X[ii][j]) - (9999 * Y[i][j][ii]))
                    constraint = self.__model.add_constr(constraint, name=f'Ensure tightness: uav {self.__scenario.uavList[i].id}, microservice {self.__scenario.microserviceList[j].id}, uav {self.__scenario.uavList[ii].id}')        

                    
        # Add constraints to deactivate all y except one for each pair of x and z
        for i, uav in enumerate(self.__scenario.uavList):
                for j, ms in enumerate(self.__scenario.microserviceList):
                    constraint = xsum(Y[i][j]) - (len(self.__scenario.uavList)) + 1 == 0
                    self.__model.add_constr(constraint, name=f'Ensure one y for uav {uav.id} and microservice {ms.id}')

        # Add the objetive function
        # Z_groupedByUAV = Z.reshape((len(self.__scenario.microserviceList), -1))
        
        self.__model.objective = minimize(xsum([ms.heatmap[uav.position[0]][uav.position[1]] * Z[i][j] for j, ms in enumerate(self.__scenario.microserviceList) for i, uav in enumerate(self.__scenario.uavList)]))
	
      
    def solve(self):

        status = self.__model.optimize()
        print(status)
        
        # Retrieve the UAV's where a ms is deployed   
        results : np.ndarray = np.array([ var for var in self.__model.vars]).reshape((len(self.__scenario.uavList), -1))

        
        df_data = [{'Decision Variable': var.name, 'Value': var.x} for index, var in enumerate(self.__model.vars)]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
            df = pd.DataFrame(df_data)
            dfToShow = df[::2+len(self.__scenario.uavList)]
            print(dfToShow.loc[dfToShow['Value'] == 1])
        self.solutionToCSV(df_data)
        return pd.DataFrame(df_data)


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
        
        print('Deployment Scheme:')
        [print(decision_variables[1]) for decision_variable in decision_variables if int(decision_variable[1]) == 1]
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
                        print('Hula', dec_var)
                        if (int(dec_var[1]) == 1):
                            print('Entré')
                            current_neigh = self.__spMatrix[i][k]


                            if (current_neigh < best_neigh):
                                best_neigh = current_neigh

                    jumps_list.append(best_neigh)

            [csv_row.append(jump) for jump in jumps_list]                             
            adj_list = self.__adjMatrix[i]
            [csv_row.append(adj) for adj in adj_list]                             
            csv_list.append(csv_row)
        
        [print(csv_row) for csv_row in csv_list]
        
        column_list = ['drone']
        [column_list.append(m.id) for m in self.__scenario.microserviceList]
        [column_list.append(f'heat_{m.id}') for m in self.__scenario.microserviceList]
        [column_list.append(f'jumps_{m.id}') for m in self.__scenario.microserviceList]
        [column_list.append(f'adj_{d.id}') for d in self.__scenario.uavList]
        print(len(csv_list[0]), len(column_list)) 
        result_df = pd.DataFrame(csv_list, columns=column_list)
        result_df.to_csv(f'../Solutions/{self.__scenario.scenarioName}.csv', sep=',', index=False)
        
