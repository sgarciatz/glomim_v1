# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PySide6 import QtCore, QtGui, QtWidgets


class Ui_MainApplication(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1500, 800)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(parent=self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(9, 9, 1471, 741))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1500, 22))
        self.menubar.setObjectName("menubar")
        self.menuArchive = QtWidgets.QMenu(parent=self.menubar)
        self.menuArchive.setObjectName("menuArchive")
        self.menuViews = QtWidgets.QMenu(parent=self.menubar)
        self.menuViews.setObjectName("menuViews")
        self.menuMicroserviceView = QtWidgets.QMenu(parent=self.menuViews)
        self.menuMicroserviceView.setObjectName("menuMicroserviceView")
        self.menuSolvers = QtWidgets.QMenu(parent=self.menubar)
        self.menuSolvers.setObjectName("menuSolvers")


        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)


        self.actionCreate_new_Scenario = QtGui.QAction(parent=MainWindow)
        self.actionCreate_new_Scenario.setObjectName("actionCreate_new_Scenario")


        self.actionScenario_View = QtGui.QAction(parent=MainWindow)
        self.actionScenario_View.setObjectName("actionScenario_View")

        self.actionUAV_View = QtGui.QAction(parent=MainWindow)
        self.actionUAV_View.setObjectName("actionUAV_View")
        

        
        self.actionDeployment_View = QtGui.QAction(parent=MainWindow)
        self.actionDeployment_View.setObjectName("actionDeployment_View")
        
        self.actionSave_current_Scenario = QtGui.QAction(parent=MainWindow)
        self.actionSave_current_Scenario.setObjectName("actionSave_current_Scenario")
        
        self.actionLoad_existing_Scenario = QtGui.QAction(parent=MainWindow)
        self.actionLoad_existing_Scenario.setObjectName("actionLoad_existing_Scenario")
        
        self.actionAdd_Microservice = QtGui.QAction(parent=MainWindow)
        self.actionAdd_Microservice.setObjectName("actionAdd_Microservice")
        
        self.actionGLOSIP = QtGui.QAction(parent=MainWindow)
        self.actionGLOSIP.setObjectName("actionGLOSIP")

        self.actionGLOMIP = QtGui.QAction(parent=MainWindow)
        self.actionGLOMIP.setObjectName("actionGLOMIP")
        
        self.actionMANETOptiServ = QtGui.QAction(parent=MainWindow)
        self.actionMANETOptiServ.setObjectName("actionMANETOptiServ")
        
        
        self.menuMicroserviceView.addAction(self.actionAdd_Microservice)
        self.menuMicroserviceView.addSeparator()
        self.menuArchive.addAction(self.actionCreate_new_Scenario)
        self.menuArchive.addSeparator()
        self.menuArchive.addAction(self.actionSave_current_Scenario)
        self.menuArchive.addSeparator()
        self.menuArchive.addAction(self.actionLoad_existing_Scenario)
        self.menuViews.addAction(self.actionScenario_View)
        self.menuViews.addAction(self.actionUAV_View)
        self.menuViews.addAction(self.actionDeployment_View)
        self.menuViews.addAction(self.menuMicroserviceView.menuAction())
        
        self.menuSolvers.addAction(self.actionGLOSIP)
        self.menuSolvers.addAction(self.actionGLOMIP)
        self.menuSolvers.addAction(self.actionMANETOptiServ)
        
        self.menubar.addAction(self.menuArchive.menuAction())
        self.menubar.addAction(self.menuViews.menuAction())
        self.menubar.addAction(self.menuSolvers.menuAction())
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuArchive.setTitle(_translate("MainWindow", "Archive"))
        self.menuViews.setTitle(_translate("MainWindow", "Views"))
        self.menuMicroserviceView.setTitle(_translate("MainWindow", "Microservice View"))
        self.menuSolvers.setTitle(_translate("MainWindow", "Solvers"))        
        
        self.actionCreate_new_Scenario.setText(_translate("MainWindow", "Create new Scenario"))
        self.actionScenario_View.setText(_translate("MainWindow", "Scenario View"))
        self.actionUAV_View.setText(_translate("MainWindow", "UAV View"))
        self.actionAdd_Microservice.setText(_translate("MainWindow", "Add Microservice"))
        self.actionDeployment_View.setText(_translate("MainWindow", "Deployment View"))
        self.actionSave_current_Scenario.setText(_translate("MainWindow", "Save current Scenario"))
        self.actionLoad_existing_Scenario.setText(_translate("MainWindow", "Load existing Scenario"))
        self.actionGLOSIP.setText(_translate("MainWindow", "Solve with GLOSIP")) 
        self.actionGLOMIP.setText(_translate("MainWindow", "Solve with GLOMIP"))
        self.actionMANETOptiServ.setText(_translate("MainWindow", "Solve with MANETOptiServ"))


