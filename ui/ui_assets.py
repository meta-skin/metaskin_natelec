import attr
from PyQt5 import QtGui, QtWidgets

@attr.s(auto_attribs=True)
class UiDimensions:
    mainWindowWidth: int = 800 *2
    mainWindowHeight: int = 600  *2

    # width
    width40: float = mainWindowWidth / 20   
    width80: float = mainWindowWidth / 10
    width95: float = mainWindowWidth / 8.421052
    width110: float = mainWindowWidth / 7.272727
    width120: float = mainWindowWidth / 6.666666
    width340: float = mainWindowWidth / 2.352941


    # height
    height40: float = mainWindowHeight / 15
    height70: float = mainWindowHeight / 8.571429
    height80: float = mainWindowHeight / 7.5
    height210: float = mainWindowHeight / 2.857143

    # widget size
    widgetWidth: float = mainWindowWidth / 2.352941
    widgetHeight: float = mainWindowHeight / 2.857143

    # button size
    buttonWidth: float = mainWindowWidth / 6.666666
    buttonHeight: float = mainWindowHeight / 8.571429

    # radius
    radius70: float = mainWindowHeight / 8.571429

    # font
    font12 = mainWindowWidth / 66.666666
    font20 = mainWindowWidth / 40   

@attr.s(auto_attribs=True)
class UiColors:
    darkBackground: str = "rgb(32,37,64)"
    lightBackground: str = "rgb(45,51,87)"
    
    buttonBackground: str = "white"

class FontStyle1(QtGui.QFont):
    def __init__(self, size):
        super().__init__()
        self.setFamily("Arial")
        self.setPointSize(size)

class ButtonStyle1(QtWidgets.QPushButton):
    def __init__(self, parent, text, font):
        super().__init__(parent)
        self.setStyleSheet("background-color: white")
        self.setText(text)
        self.setFont(font)
