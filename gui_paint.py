from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPainter, QPen, QPixmap, QMouseEvent
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor

class DrawableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.last_point = QPoint()
        self.pen_color = Qt.red
        self.pen_width = 4

    def setPixmap(self, pixmap: QPixmap):
        super().setPixmap(pixmap.copy())  # Make a copy so we can draw on it

    def setPenColor(self, color: QColor):
        self.pen_color = color
    
    def setPenWidth(self, width: int):
        self.pen_width = width



    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing:
            painter = QPainter(self.pixmap())
            pen = QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            painter.end()
            self.last_point = event.pos()
            self.update()  # Trigger repaint

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def getModifiedPixmap(self):
        return self.pixmap()
