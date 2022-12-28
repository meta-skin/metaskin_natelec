from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas



class SensorValueCanvas(FigureCanvas):
    def __init__(self, parent = None, width = 1000, height = 400, dpi = 100):
        plt.tight_layout()

        self.fig = Figure(figsize = (width/100,height/100), dpi = dpi)
        self.axes = self.fig.add_subplot(111)
        self.fig.patch.set_alpha(0)
        self.axes.patch.set_alpha(0)
        self.axes.tick_params(axis = 'x', colors = 'white')
        self.axes.tick_params(axis = 'y', colors = 'white')

        for direction in ["left", "right", "bottom", "top"]:
            self.axes.spines[direction].set_visible(False)
        self.axes.grid(color = [24/255,27/255,46/255])
        self.sensor_values = [0] * 100 
        self.computeInitialFigure()

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

    def setSensorValueWindow(self, sensor_value_window):
        self.sensor_values = sensor_value_window

    def computeInitialFigure(self):
        x = np.arange(100)
        y = np.array(self.sensor_values)
        self.line = self.axes.plot(x, y, animated = True, lw= 2)[0]
        
    def updatePlot(self, i):
        x = np.arange(100)
        y = np.array(self.sensor_values)
        self.line.set_ydata(y)
        tmp_sensor = y.reshape(-1,1)
        cur_min, cur_max = np.min(tmp_sensor), np.max(tmp_sensor)

        self.axes.set_ylim([cur_min-0.1, cur_max+0.1])
        self.draw()
        return [self.line]

    def startAni(self):
        self.ani = animation.FuncAnimation(self.fig, self.updatePlot, blit = True, interval = 50)



class KeyboardPushButton(QtWidgets.QFrame):
    def __init__(self, parent, x, y, dx, dy):
        QtWidgets.QFrame.__init__(self, parent)
        self.setGeometry(QtCore.QRect(x, y, dx, dy))
        # self.setObjectName(_fromUtf8("gridFrame"))
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setSpacing(10)

        # self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.keys = [['Q','W','E','R','T','Y','U','I','O','P'],
                        ['A','S','D','F','G','H','J','K','L',';'],
                            ['Shift','Z','X','C','V','B','N','M',',','.']]
        self.create_keys()
        self.gridLayout.setRowStretch(1,2)
        self.gridLayout.setRowStretch(2,2)
    def create_keys(self):
        self.button = {}
        self.button_str = {}
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        for i, layer in enumerate(self.keys):
            for j, key in enumerate(layer):
                self.button["b{0}".format(key)] = QtWidgets.QPushButton(self)
                self.button["b{0}".format(key)].setStyleSheet("QPushButton {color: white; background-color:qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(57, 65, 80, 255), stop:1 rgba(81, 88, 101, 255));; border: 2px solid rgb(255, 255, 255);}")
                self.gridLayout.addWidget(self.button["b{0}".format(key)], i,j, 1, 1)
                self.button["b{0}".format(key)].setText(str(key))
                self.button["b{0}".format(key)].setFont(font)
                # self.button_str["b{0}".format(key)] = key
                self.button["b{0}".format(key)].clicked.connect(self.button_clicked)

    def button_clicked(self, key):
        print("button pressed: ",key)