
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.Qt3DCore import *
from PyQt5.QtWidgets import *
from PyQt5.Qt3DExtras import *

class OrbitTransformController(QObject):
    def __init__(self, parent):
        super(OrbitTransformController, self).__init__(parent)
        self.m_target = QTransform()
        self.m_matrix = QMatrix4x4()
        self.m_radius = 1.0
        self.m_angle = 0

    def getTarget(self):
        return self.m_target

    def setTarget(self, target):
        if self.m_target != target:
            self.m_target = target
            self.targetChanged.emit()

    def getRadius(self):
        return self.m_radius

    def setRadius(self, radius):
        if not QtCore.qFuzzyCompare(self.m_radius, radius):
            self.m_radius = radius
            self.updateMatrix()
            self.radiusChanged.emit()

    def getAngle(self):
        return self.m_angle

    def setAngle(self, angle):
        if not QtCore.qFuzzyCompare(angle, self.m_angle):
            self.m_angle = angle
            self.updateMatrix()
            self.angleChanged.emit()

    def updateMatrix(self):
        self.m_matrix.setToIdentity()
        self.m_matrix.rotate(self.m_angle, QVector3D(0, 1, 0))
        self.m_matrix.translate(self.m_radius, 0, 0)
        self.m_target.setMatrix(self.m_matrix)

    # QSignal
    targetChanged = pyqtSignal()
    radiusChanged = pyqtSignal()
    angleChanged = pyqtSignal()

    # Qt properties
    target = pyqtProperty(QTransform, fget=getTarget, fset=setTarget)
    radius = pyqtProperty(float, fget=getRadius, fset=setRadius)
    angle = pyqtProperty(float, fget=getAngle, fset=setAngle)

class DirectionLabel(Qt3DWindow):
    def __init__(self, parent,x, y, dx, dy):
        Qt3DWindow.__init__(self)

        self.contrainer = parent.createWindowContainer(self, parent)
        self.contrainer.setGeometry(QtCore.QRect(x, y, dx, dy))

        self.createScene()

        self.camera = self.camera()
        self.camera.lens().setPerspectiveProjection(45.0, 16.0/9.0, 0.1, 1000)
        self.camera.setPosition(QVector3D(20, 15, 40))
        self.camera.setViewCenter(QVector3D(0, 0, 0))

        self.camController = QOrbitCameraController(self.scene)
        self.camController.setLinearSpeed( 50.0 )
        self.camController.setLookSpeed( 180.0 )
        self.camController.setCamera(self.camera)

        self.setRootEntity(self.scene)

    def createScene(self):
        rootEntity = QEntity()
        self.material = QMorphPhongMaterial(rootEntity)
        self.material.setDiffuse(QColor(15,81,212,0.6))
        self.objectEntity = QEntity(rootEntity)
        self.objectMesh = QSphereMesh()
        self.objectMesh.setRadius(0.5)

        self.objectTransform = QTransform()
        self.controller = OrbitTransformController(self.objectTransform)
        self.controller.setTarget(self.objectTransform)
        self.controller.setRadius(0)

        self.objectRotateTransformAnimation = QPropertyAnimation(self.objectTransform)
        self.objectRotateTransformAnimation.setTargetObject(self.controller)
        self.objectRotateTransformAnimation.setPropertyName(b'angle')
        self.objectRotateTransformAnimation.setStartValue(0)
        self.objectRotateTransformAnimation.setEndValue(360)
        self.objectRotateTransformAnimation.setDuration(10000)
        self.objectRotateTransformAnimation.setLoopCount(-1)
        self.objectRotateTransformAnimation.start()

        self.objectEntity.addComponent(self.objectMesh)
        self.objectEntity.addComponent(self.objectTransform)
        self.objectEntity.addComponent(self.material)
        self.scene = rootEntity

    def pressed(self, num = 6):
        if(num == 0):
            self.objectMesh = QSphereMesh()
            self.objectMesh.setRadius(0.5)
            self.objectEntity.addComponent(self.objectMesh)
            self.objectEntity.addComponent(self.material)
        elif(num == 1):
            #circular cone
            self.objectMesh = QConeMesh()
            self.objectMesh.setBottomRadius(5)
            self.objectMesh.setLength(10)
            self.objectEntity.addComponent(self.objectMesh)
            self.objectEntity.addComponent(self.material)
        elif(num == 2):
            #cubic
            self.objectMesh = QCuboidMesh()
            self.objectMesh.setXExtent(10)
            self.objectMesh.setYExtent(10)
            self.objectMesh.setZExtent(10)
            self.objectEntity.addComponent(self.objectMesh)
            self.objectEntity.addComponent(self.material)
            pass
        elif(num == 3):
            #cylinder
            self.objectMesh = QCylinderMesh()
            self.objectMesh.setLength(10)
            self.objectMesh.setRadius(5)
            self.objectEntity.addComponent(self.objectMesh)
            self.objectEntity.addComponent(self.material)
            pass
        elif(num == 4):
            #half sphere
            self.objectMesh = QSphereMesh()
            self.objectMesh.setRadius(5)
            self.objectEntity.addComponent(self.objectMesh)
            self.objectEntity.addComponent(self.material)
            pass
        elif(num == 5):
            #hexagonal cone 
            self.objectMesh = QCylinderMesh()
            self.objectMesh.setLength(10)
            self.objectMesh.setRadius(5)
            self.objectMesh.setSlices(6)
            self.objectEntity.addComponent(self.objectMesh)
            pass
        elif(num ==6):
            #pyramid
            self.objectMesh = QConeMesh()
            self.objectMesh.setBottomRadius(7)
            self.objectMesh.setLength(10)
            self.objectMesh.setSlices(4)
            self.objectEntity.addComponent(self.objectMesh)
            pass
        elif(num == 7):
            #triangular cone
            self.objectMesh = QCylinderMesh()
            self.objectMesh.setLength(10)
            self.objectMesh.setRadius(5)
            self.objectMesh.setSlices(3)
            self.objectEntity.addComponent(self.objectMesh)
            pass