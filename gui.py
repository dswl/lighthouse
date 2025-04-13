import sys
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from models.church import Model
from gui_paint import DrawableImageLabel
from PyQt5.QtWidgets import QColorDialog
from PyQt5.QtWidgets import QSlider, QLabel



def tensor_to_qpixmap(tensor):
    tensor = tensor.squeeze().permute(1, 2, 0).cpu().clamp(0, 1).numpy()
    img = (tensor * 255).astype(np.uint8)
    h, w, ch = img.shape
    bytes_per_line = ch * w
    # Convert to bytes before passing
    qimg = QImage(img.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)



class ImageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Normalization Viewer")

        # Model
        self.model = Model()
        self.model.setTimestep(180)

        # Layouts
        self.layout = QVBoxLayout()
        self.pen_layout = QHBoxLayout()
        self.image_layout = QHBoxLayout()
        self.button_layout = QHBoxLayout()
        self.button2_layout = QHBoxLayout()

        self.color_button = QPushButton("Choose Color")
        self.color_button.clicked.connect(self.choose_color)

        self.size_label = QLabel("Pen Size:")
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(30)
        self.size_slider.setValue(4)
        self.size_slider.valueChanged.connect(self.change_pen_size)

        self.undo_button = QPushButton("Undo")

        self.pen_layout.addWidget(self.undo_button)
        self.pen_layout.addWidget(self.color_button)
        self.pen_layout.addWidget(self.size_label)
        self.pen_layout.addWidget(self.size_slider)


        # Image slots
        self.image_labels = [QLabel("Slot {}".format(i + 1)) for i in range(3)]
        self.image_labels = [DrawableImageLabel() if i == 0 else QLabel(f"Slot {i + 1}") for i in range(3)]
        for label in self.image_labels:
            label.setFixedSize(256, 256)
            label.setStyleSheet("border: 1px solid black; background: #ddd;")
            label.setAlignment(Qt.AlignCenter)
            self.image_layout.addWidget(label)

        self.undo_button.clicked.connect(self.image_labels[0].undo)
        # Buttons
        self.load_button = QPushButton("Load Image")
        self.normalize_button = QPushButton("Normalize Image")
        self.diffuse_button = QPushButton("Start Diffusion")

        self.load_button.clicked.connect(self.load_image)
        self.normalize_button.clicked.connect(self.normalize_image)
        self.diffuse_button.clicked.connect(self.diffuse)

        for btn in [self.load_button, self.normalize_button, self.diffuse_button]:
            self.button_layout.addWidget(btn)

        # Second Row of Buttons
        self.recurseDiffusion = QPushButton("Manipulate Diffusion")
        self.resetDiffusion = QPushButton("Reset Diffused")
        self.injectNormalize = QPushButton("Normalize and Inject Noise")

        self.recurseDiffusion.clicked.connect(self.recurse_diffusion)
        self.resetDiffusion.clicked.connect(self.reset_diffusion)
        self.injectNormalize.clicked.connect(self.inject_and_normalize)

        for btn in [self.recurseDiffusion, self.resetDiffusion, self.injectNormalize]:
            self.button2_layout.addWidget(btn)

        # Assemble layout
        self.layout.addLayout(self.pen_layout)
        self.layout.addLayout(self.image_layout)
        self.layout.addLayout(self.button_layout)
        self.layout.addLayout(self.button2_layout)
        self.setLayout(self.layout)

    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.image_labels[0].setPenColor(color)

    def change_pen_size(self, value):
        self.image_labels[0].setPenWidth(value)


    def get_drawn_image_tensor(self):
        pixmap = self.image_labels[0].getModifiedPixmap()
        image = pixmap.toImage()
        width, height = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        # arr = np.array(ptr).reshape((height, width, 4))[:, :, :3]  # RGB only
        arr = np.array(ptr).reshape((height, width, 4))[:, :, [2, 1, 0]]  # BGR → RGB
        arr = arr.astype(np.float32) / 255
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.model.device)
        return tensor

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if not file_path:
            return

        self.model.loadImageFromFile(file_path)

        # Display in slot 1
        pixmap = tensor_to_qpixmap(self.model.getImage())
        self.image_labels[0].setPixmap(pixmap)

    def normalize_image(self):
        self.model.normalizeImage()

        # Display in slot 2
        pixmap = tensor_to_qpixmap(self.model.getImage())
        self.image_labels[1].setPixmap(pixmap)

    def diffuse(self):
        for diffusedImage in self.model.diffusionGenerator(verbose=True):
            # print("HELLO")
            pixmap = tensor_to_qpixmap(diffusedImage)
            self.image_labels[2].setPixmap(pixmap)
            QApplication.processEvents()
        
        pixmap = tensor_to_qpixmap(self.model.getDiffused())
        self.image_labels[2].setPixmap(pixmap)

    def recurse_diffusion(self):
        pixmap = tensor_to_qpixmap(self.model.getDiffused())
        self.image_labels[0].setPixmap(pixmap)

    def reset_diffusion(self):
        modified_tensor = self.get_drawn_image_tensor()
        self.model.setImage(modified_tensor)

        pixmap = tensor_to_qpixmap(self.model.getImage())
        self.image_labels[0].setPixmap(pixmap)
        

    def inject_and_normalize(self):
        # self.model.normalizeImage()
        self.model.testSyntheticNoise()
        pixmap = tensor_to_qpixmap(self.model.getImage())
        self.image_labels[1].setPixmap(pixmap)

# Run the app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageApp()
    viewer.show()
    sys.exit(app.exec())
