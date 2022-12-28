from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.ui_assets import UiDimensions, UiColors, FontStyle1, ButtonStyle1
from ui.screen import SensorValueCanvas

class Ui_MainWindow(object):

    def uploadMethod(self, method):
        self.method = method

    def setupUi(self, MainWindow):
        self.dimensions = UiDimensions()
        self.colors = UiColors()
        self.mainWindow = MainWindow 

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(self.dimensions.mainWindowWidth, self.dimensions.mainWindowHeight)
        MainWindow.setStyleSheet("background: " + self.colors.darkBackground + ";")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        self.sensorGraph = QtWidgets.QWidget(self.centralwidget)
        self.sensorGraph.setStyleSheet("background: " + self.colors.lightBackground + ";")
        self.sensorGraph.setObjectName("sensorWidget")
        self.sensorGraph.setGeometry(QtCore.QRect(self.dimensions.width40 + self.dimensions.widgetWidth + self.dimensions.width40, self.dimensions.height80, self.dimensions.widgetWidth, self.dimensions.widgetHeight))
        self.sensorCanvas = SensorValueCanvas(self.sensorGraph, self.dimensions.width340, self.dimensions.height210)
        self.sensorCanvas.startAni()

        self.resultLabel = QtWidgets.QLabel(self.centralwidget)
        self.resultLabel.setGeometry(QtCore.QRect(self.dimensions.width40, self.dimensions.height80, self.dimensions.widgetWidth, self.dimensions.widgetHeight))
        self.resultLabel.setAlignment(Qt.AlignCenter)
        self.resultLabel.setObjectName("StatusWidget")
        self.resultLabel.setText("Result")
        self.resultLabel.setStyleSheet("color: white;" "background: {};".format(self.colors.lightBackground))
        self.resultLabel.setFont(FontStyle1(self.dimensions.font20))

        self.button_data_collection = ButtonStyle1(self.centralwidget, "Collect\nStart", FontStyle1(self.dimensions.font12))
        self.button_data_collection.clicked.connect(self.stateDataCollection)
        self.button_data_collection.setGeometry(QtCore.QRect(self.dimensions.width40 + self.dimensions.width110, self.dimensions.height80 + self.dimensions.widgetHeight +  self.dimensions.height80, self.dimensions.buttonWidth, self.dimensions.buttonHeight))
        
        self.button_start_prediction = ButtonStyle1(self.centralwidget, "Start\nPrediction", FontStyle1(self.dimensions.font12))
        self.button_start_prediction.clicked.connect(self.stateStartPrediction)
        self.button_start_prediction.setGeometry(QtCore.QRect(self.dimensions.width40 + self.dimensions.width110 + self.dimensions.buttonWidth + self.dimensions.width95 + self.dimensions.radius70 + self.dimensions.width95, self.dimensions.height80 + self.dimensions.widgetHeight +  self.dimensions.height80, self.dimensions.buttonWidth, self.dimensions.buttonHeight))

        self.timer = QTimer() 
        self.timer.setInterval(50)

        self.timer.timeout.connect(self.updateFigures)
        self.timer.start()

    def updateResult(self, result):
        self.resultLabel.setText(result)

    def updateFigures(self):
        sensor_vis_window = self.method.getVisWindow('sensor')
        self.sensorCanvas.setSensorValueWindow(sensor_vis_window)

    def stateDataCollection(self):
        print("demo start")
        self.method.changeState('demo')

    def stateStartPrediction(self):
        print("demo stop")
        self.method.changeState('train')

    def setPrediction(self, prediction):
        self.resultLabel.setText(prediction)





  


