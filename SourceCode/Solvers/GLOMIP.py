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

class GLOMIP(object):

    def __init__(self):
    
        #Load configuration from Database 
        self.__scenario = Database().scenario        
        # C is the Big-M constant, it is a value outside the domain of all parameters
        self.__C      = (np.array(self.P).max() * np.array(self.H).max()) + 1
        
        # Initialize the model
        self.model = Model(sense=MINIMIZE, solver_name=GRB)
        self.model.verbose = 1
        self.model.threads = -1
        #self.model.max_mip_gap = 0.001
        self.model.emphasis = 1
    

    def initializeModel(self):
        # Add the decision variables
        #   x (i,j,k)       -> says if m(k) is deployed in d(i,j)
        #   z (i,j,k)       -> the lowest number of jumps needed to serve d(i,j) the microservice m(k)
        #   y (i,j,k,ii,jj) -> activation/desactivation variable to select the lowest posibility for z(i,j,k)   
        #   U               -> maximun value of Z
        max_z = max(np.array(self.H).max(), np.array(self.P).max())
        min_z = 0
        for microservice in self.M:
            for drone in self.D:
                    self.model.add_var(f'x_{microservice},d{drone["position"]}', var_type=BINARY)
                    self.model.add_var(f'z_{microservice},d{drone["position"]}', var_type=CONTINUOUS, lb=min_z, ub=max_z)
                    for drone2 in self.D:
                        self.model.add_var(f'y_{microservice},d({drone["position"]}),d({drone2["position"]})', var_type=BINARY)
        
        decision_variables = np.array(self.model.vars)
        #U = self.model.add_var(f'U', var_type=CONTINUOUS, lb= -self.C, ub= (np.array(self.P).max() * np.array(self.H).max()))


        # Add the drones limitations constraints
        #TODO The constraint must check against the limitations of the drones, not the deployment matrix. With this configuration a drone can hold a maximun of 1 microservices
        decision_variables_aux = decision_variables.reshape((len(self.M), len(self.D), -1))
        decision_variables_x = decision_variables_aux[:,:,0].reshape((len(self.M),-1)).T
        for i, x in enumerate(decision_variables_x):
            constraint = xsum(x) <= 1
            self.model.add_constr(constraint, name=f'Capacity of {self.D[i]["id"]} not surpased')


        # Add the mandatory deployment of all microservices constraint
        decision_variables_x = decision_variables_x.T
        for i, x in enumerate(decision_variables_x):
            constraint = xsum(x) >= 1
            self.model.add_constr(constraint, name=f'Microservice {self.M[i]} deployed')


        # Add the top boundaries for z (i,j,k) constraint to ensure tightness
        decision_variables_x = decision_variables.reshape((len(self.M), len(self.D), -1))[:,:,0]      
        decision_variables_z = decision_variables.reshape((len(self.M), len(self.D), -1))[:,:,1]
        decision_variables_y = decision_variables.reshape((len(self.M), len(self.D), -1))[:,:,2:].reshape((len(self.M), len(self.D), len(self.D)))

        ph_ratio = np.array(self.P).max() / np.array(self.H).max()


        for microservice in range(len(self.M)):
            for drone in range(len(self.D)):
                for drone2 in range(len(self.D)):
                    #constraint = decision_variables_z[microservice][drone] >= (decision_variables_x[microservice][drone2] * self.P[drone][drone2] * self.H[microservice][drone])  - self.C * decision_variables_y[microservice][drone][drone2]
                    x = decision_variables_x[microservice][drone2]
                    h = self.H[microservice][drone]
                    h_binary = 0 if h == 0 else 1
                    p = self.P[self.D[drone]['position'][0]*self.__shape[1] + self.D[drone]['position'][1]][self.D[drone2]['position'][0]*self.__shapeT[1] + self.D[drone2]['position'][1]]
                    y = decision_variables_y[microservice][drone][drone2]
                    z = decision_variables_z[microservice][drone]
                    constraint = z >= h_binary * ((max_z ) - ( x * h * ph_ratio  / p)) - (max_z + 1) * y                  
                    self.model.add_constr(constraint, name=f'Ensure Tightness of top boundary for  yz({self.M[microservice]}{self.D[drone]["id"]}{self.D[drone2]["id"]}) z({microservice}{self.D[drone]["id"]}{self.D[drone2]["id"]})')

        # Add the constraints that ensures that only one of the previous constraint is active for each z(k,i,j)
        for microservice in range(len(self.M)):
            for drone in range(len(self.D)):
                constraint = xsum([decision_variables_y[microservice][drone][drone2] for drone2 in range(len(self.D))]) == len(self.D) - 1
                self.model.add_constr(constraint, name=f'Check that only one top boundary constraint is active for y({microservice},{self.D[drone]["id"]},{self.D[drone2]["id"]})')

        #for i, zetas in enumerate(decision_variables_z):
            #for j,z in enumerate(zetas):
                #constraint = U >= z 
                # self.model.add_constr(constraint, name=f'U greater than z{i},{j}')


        # Add the objetive function
        decision_variables_z = decision_variables.reshape((len(self.M), len(self.D), -1))[:,:,1]
        self.model.objective = minimize(xsum([z for zetas in decision_variables_z for z in zetas]))
        #self.model.objective = minimize(U)
      
    def solve(self):

        status = self.model.optimize()
        print(status)

        df_data = [{'Decision Variable': var.name, 'Value': var.x} for index, var in enumerate(self.model.vars)]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
            df = pd.DataFrame(df_data)
        return pd.DataFrame(df_data)


    def flatten(self, xs):
        result = []
        if isinstance(xs, (list, tuple)):
            for x in xs:
                result.extend(self.flatten(x))
        else:
            result.append(xs)
        return result
    
    def printSolution(self, background_img: str = None):
        solution_df = pd.DataFrame([{"Decision Variable": var.name, "Value": var.x} for var in self.model.vars[:]])

        print(solution_df)
        solution_ndarray = solution_df.to_numpy()[:,1]
        print(solution_ndarray.shape)
        solution_ndarray = solution_ndarray.reshape(len(self.M), len(self.D), -1)[:,:,0]
        deployment_matrix = np.zeros((len(self.M), self.__shape[0], self.__shape[1]))
        for microservice in range(len(self.M)):
            for i, drone in enumerate(self.D):
               deployment_matrix[microservice][drone['position'][0]][drone['position'][1]] =  solution_ndarray[microservice][i]


        deployment_matrix = [m *(i+1) for m, i in zip(deployment_matrix[:], range(deployment_matrix.shape[0]))]
        deployment_matrix = np.add.reduce(deployment_matrix[:] )

        colors = ['#FFFFFF', '#FF2D01','#0184FF', '#FFBA01','#B601FF', '#FFF701', '#9BFF01', '#01FFDC','#010DFF', '#FF01E0']
        cmap = ListedColormap(colors[0:len(self.M)+1], name='from_list', N=None)

        plt.figure(figsize = (24,8))
        ax = sns.heatmap(deployment_matrix.astype(np.int8), cmap=cmap, linecolor='gainsboro', linewidths=.1, alpha=0.6)
        plt.title('Deployment Matrix')

        colorbar = ax.collections[0].colorbar
        n = len(self.M) + 1


        r = colorbar.vmax - colorbar.vmin
        colorbar.set_ticks([colorbar.vmin + 0.5 * r / (n) + r * i / (n) for i in range(len(self.M) + 1)])
        colorbar.set_ticklabels(['Empty'] + self.M)
        try:
            plt.imshow(plt.imread(background_img), extent=[0, self.__shape[1], self.__shape[0], 0])
        except:
            pass
        plt.show()
    
    def getMin(self, microservice_index, src_drone, proposed_solution):
        hops_to_dsts = []
        i  = src_drone['position'][0]
        j  = src_drone['position'][1]
        for dst_drone_index, dst_drone in enumerate(self.D):
                if (proposed_solution[microservice_index][dst_drone_index] == 1 ):
                    ii = dst_drone['position'][0]
                    jj = dst_drone['position'][1]
                    hops_to_dsts.append(self.P[i*self.__shape[1]+j][ii*self.__shape[1]+jj] - 1)
        print(f'{self.M[microservice_index]} - d({i},{j}) ->', hops_to_dsts)
        return min(hops_to_dsts)

    def solutionLongFormat(self, scenario_name: str):

        proposed_solution = np.array([decision_variable.x for decision_variable in self.model.vars[:]])

        # print(len(proposed_solution))
        proposed_solution_aux = proposed_solution.reshape((len(self.M), len(self.D), -1))[:,:,0]

        # Calcular el número de saltos mínimo que hay que dar para, desde un dron de origen, poder comsumir un servicio
        # alojado en el dron de destino más cercano
        proposed_solution = []
        for ms in range(len(self.M)):
            for i ,d in enumerate(self.D): 
                if (self.H[ms][i] > 0):
                   proposed_solution.append(self.getMin(ms, d, proposed_solution_aux))
                else:
                    proposed_solution.append(-1)
                        
                       
        proposed_solution = np.array(proposed_solution).reshape((len(self.M), -1))
        proposed_solution = proposed_solution.tolist()
        
        for index, sublist in enumerate(proposed_solution):
            proposed_solution[index] = [element for element in sublist if element != -1]
          
        dataset = []
        for ms, sublist in zip(self.M, proposed_solution):
                for cost in sublist:
                    dataset.append([scenario_name, ms, cost])

        return dataset
    
    def solutionToCSV(self):
        decision_variables = np.array([decision_variable.x for decision_variable in self.model.vars[:]])
        decision_variables = decision_variables.reshape((len(self.M), len(self.D), -1))[:,:,0].T #drone -> Xms1, Xms2, Xms3, ..., Xmsn
        heatmaps = self.H.T # drone -> Hms1, Hms2, Hms3 ... Hmsn
        jumps = np.array([[ self.getMin(ms, d, decision_variables.T) if (self.H[ms][i] > 0) else -1 for i, d in enumerate(self.D)] for ms in range(len(self.M))]).T #drone -> ms1, ms2, ms3, ..., msn
        adjacencyList = np.array([[ 1 if max( abs(d2['position'][0] - d['position'][0]), abs(d2['position'][1] - d['position'][1])) < 2 else 0  for j, d2 in enumerate(self.D)] for i, d in enumerate(self.D)]) #drone -> drone, drone2, drone3, ..., drone4
        print(decision_variables.shape, heatmaps.shape, jumps.shape, adjacencyList.shape)
        allLists = zip(decision_variables, heatmaps, jumps, adjacencyList)
        result_list = []        
        for i, (Xms, Hms, Jd2, Ad2) in enumerate(allLists):
            result_list.append(
                {
                   'drone': f"drone_{self.D[i]['position'][0]}-{self.D[i]['position'][1]}",
                })
            print(Xms.shape, Hms.shape, Jd2.shape, Ad2.shape)
            for j, (x, h, ju) in enumerate(zip(Xms, Hms, Jd2)):
                result_list[i][self.M[j]] = x
                result_list[i][f'heat_{self.M[j]}'] = h
                result_list[i][f'jumps_{self.M[j]}'] = ju
            for k, drone in enumerate(Ad2):
                result_list[i][f'adj_drone{k}'] = drone
        
        column_list = ['drone']
        [column_list.append(m) for m in self.M]
        [column_list.append(f'heat_{ms}') for ms in self.M]
        [column_list.append(f'jumps_{self.M[i]}') for i in range(len(self.M))]
        [column_list.append(f'adj_drone{i}') for i in range(len(self.D))]
        result_df = pd.DataFrame(result_list, columns=column_list)
        result_df.to_csv('out.csv', sep=',', index=False)
    @staticmethod
    def compareSolutions(solutions_df):
        medianprops = {'linestyle':'-', 'linewidth':2, 'color':'lightcoral'}
        meanpointprops = {'marker':'D', 'markeredgecolor':'forestgreen','markerfacecolor':'forestgreen'}

        sns.set_theme(style="ticks", palette="pastel")


        fig, ax = plt.subplots(figsize=(20, 10))

        sns.boxplot(data= solutions_df, x="Microservice", y="Hops", hue="Scenario",
                    meanprops = meanpointprops, showmeans=True, 
                    flierprops={"marker": "x"},
                    medianprops=medianprops)

        plt.xticks(range(solutions_df['Microservice'].nunique()), solutions_df['Microservice'].unique(), rotation=45)
        # plt.yticks(range(0, int(df.loc[df['Hops'].idxmax()]['Hops'])+2))
        plt.yticks(range(0, 30))
        plt.xlabel('Microservices', fontsize=16)
        plt.ylabel('Latency (nº hops)', fontsize=16)
        plt.title('Medium sized Scenario', pad=30, fontsize=24)
        plt.legend(fontsize=14)
        plt.show()


auxImagesDir       = Path('../AuxImages')
inputScenariosDir  = Path('../InputScenarios')

configuration = json.load(open(inputScenariosDir.joinpath('test_010.json')))

my_model = GLOMIP(configuration=configuration)
my_model.showDeploymentMatrix(auxImagesDir.joinpath('scenario3.png'))
my_model.showHeatMaps(auxImagesDir.joinpath('scenario3.png'))
my_model.initializeModel()
my_model.solve()#.to_csv('test_010_2_result.csv')
my_model.printSolution(auxImagesDir.joinpath('scenario3.png'))

scenarios = my_model.solutionLongFormat('Test_010')
#my_model.solutionToCSV()


configuration = json.load(open(inputScenariosDir.joinpath('test_011.json')))

my_model = GLOMIP(configuration=configuration)
my_model.showHeatMaps(auxImagesDir.joinpath('scenario3.png'))
my_model.initializeModel()
my_model.solve()#.to_csv('test_011_2_result.csv')
my_model.printSolution(auxImagesDir.joinpath('scenario3.png'))

scenarios += my_model.solutionLongFormat('Test_011')

configuration = json.load(open(inputScenariosDir.joinpath('test_012.json')))

my_model = GLOMIP(configuration=configuration)
my_model.showHeatMaps(auxImagesDir.joinpath('scenario3.png'))
my_model.initializeModel()
my_model.solve()#.to_csv('test_012_2_result.csv')
my_model.printSolution(auxImagesDir.joinpath('scenario3.png'))

scenarios += my_model.solutionLongFormat('Test_012')



df = pd.DataFrame(data = scenarios,
                  columns = ['Scenario','Microservice', 'Hops'])
print(df)
GLOMIP.compareSolutions(df)
