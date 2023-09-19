# Form implementation generated from reading ui file 'addmicroservice.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PySide6 import QtCore, QtGui, QtWidgets


class Ui_AddMicroservice(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 243)
        self.verticalLayoutWidget = QtWidgets.QWidget(parent=Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(9, 9, 381, 191))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_microserviceId = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_microserviceId.setFont(font)
        self.label_microserviceId.setObjectName("label_microserviceId")
        self.verticalLayout.addWidget(self.label_microserviceId)
        self.plainTextEdit_microserviceId = QtWidgets.QPlainTextEdit(parent=self.verticalLayoutWidget)
        self.plainTextEdit_microserviceId.setObjectName("plainTextEdit_microserviceId")
        self.verticalLayout.addWidget(self.plainTextEdit_microserviceId)
        self.line = QtWidgets.QFrame(parent=self.verticalLayoutWidget)
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.label_ramRequirement = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_ramRequirement.setFont(font)
        self.label_ramRequirement.setObjectName("label_ramRequirement")
        self.verticalLayout.addWidget(self.label_ramRequirement)
        self.plainTextEdit_ramRequirement = QtWidgets.QPlainTextEdit(parent=self.verticalLayoutWidget)
        self.plainTextEdit_ramRequirement.setObjectName("plainTextEdit_ramRequirement")
        self.verticalLayout.addWidget(self.plainTextEdit_ramRequirement)
        self.line_2 = QtWidgets.QFrame(parent=self.verticalLayoutWidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        self.label_cpuRequirement = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_cpuRequirement.setFont(font)
        self.label_cpuRequirement.setObjectName("label_cpuRequirement")
        self.verticalLayout.addWidget(self.label_cpuRequirement)
        self.plainTextEdit_cpuRequirement = QtWidgets.QPlainTextEdit(parent=self.verticalLayoutWidget)
        self.plainTextEdit_cpuRequirement.setObjectName("plainTextEdit_cpuRequirement")
        self.verticalLayout.addWidget(self.plainTextEdit_cpuRequirement)
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=Form)
        self.buttonBox.setGeometry(QtCore.QRect(220, 210, 166, 25))
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setObjectName("buttonBox")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_microserviceId.setText(_translate("Form", "Specify Microservice ID"))
        self.label_ramRequirement.setText(_translate("Form", "Specify the RAM Requirement (in GB)"))
        self.label_cpuRequirement.setText(_translate("Form", "Specify the CPU Requirement (in Cores)"))
