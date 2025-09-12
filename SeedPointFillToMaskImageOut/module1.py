import sys
import fitz  # PyMuPDF
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QFileDialog, QMainWindow, QAction, QVBoxLayout, QWidget, QSlider, QLabel, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QPoint

class PDFViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pdf_doc = None
        self.current_page = 0
        self.setRenderHint(QPainter.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.image_np = None
        self.display_img = None
        self.tolerance = 10
        self.contrast = 1.0
        self.brightness = 0
        self.last_seed = None
        self.last_mask = None

    def load_pdf(self, path):
        self.pdf_doc = fitz.open(path)
        self.current_page = 0
        self.show_page()
        self.last_seed = None
        self.last_mask = None

    def show_page(self):
        if not self.pdf_doc:
            return
        page = self.pdf_doc.load_page(self.current_page)
        pix = page.get_pixmap()
        if pix.alpha:
            img_format = QImage.Format_RGBA8888
        else:
            img_format = QImage.Format_RGB888
        img = QImage(bytes(pix.samples), pix.width, pix.height, pix.stride, img_format).copy()
        self.image_np = self.qimage_to_numpy(img)
        if self.image_np is None:
            return
        self.display_img = self.image_np.copy()
        self.last_seed = None
        self.last_mask = None
        self.update_display()

    def qimage_to_numpy(self, img):
        img = img.convertToFormat(QImage.Format_RGBA8888)
        width, height = img.width(), img.height()
        byte_count = img.byteCount()
        if width == 0 or height == 0 or byte_count == 0:
            return None
        ptr = img.bits()
        arr = np.frombuffer(ptr.asstring(byte_count), np.uint8).reshape((height, width, 4))
        return arr

    def numpy_to_qpixmap(self, arr):
        h, w, ch = arr.shape
        bytes_per_line = ch * w
        img = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
        return QPixmap.fromImage(img)

    def update_display(self):
        self.scene.clear()
        pixmap = self.numpy_to_qpixmap(self.display_img)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(pixmap_item)
        self.setSceneRect(pixmap_item.boundingRect())
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        if self.image_np is None:
            return
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            if 0 <= x < self.image_np.shape[1] and 0 <= y < self.image_np.shape[0]:
                self.last_seed = (x, y)
                self.apply_flood_fill_and_update()
        super().mousePressEvent(event)

    def apply_flood_fill_and_update(self):
        if self.last_seed is None:
            return
        # Apply contrast and brightness to the whole image
        img = self.image_np.copy().astype(np.float32)
        img[..., :3] = np.clip(self.contrast * img[..., :3] + self.brightness, 0, 255)
        img = img.astype(np.uint8)

        # Flood fill to get the mask
        mask = np.zeros((img.shape[0]+2, img.shape[1]+2), np.uint8)
        lo = self.tolerance
        hi = self.tolerance
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        img_bgr = img[..., :3].copy()
        cv2.floodFill(img_bgr, mask, self.last_seed, (0,0,0), (lo, lo, lo), (hi, hi, hi), flags)
        mask = mask[1:-1,1:-1]
        self.last_mask = mask

        # Overlay blue mask (alpha blended)
        overlay = np.zeros_like(img)
        mask_indices = mask == 255
        overlay[mask_indices] = (255, 0, 0, 128)
        alpha = overlay[..., 3:4] / 255.0
        img[..., :3] = (1 - alpha) * img[..., :3] + alpha * overlay[..., :3]
        self.display_img = img.astype(np.uint8)
        self.update_display()

    def set_tolerance(self, value):
        self.tolerance = value
        if self.last_seed is not None:
            self.apply_flood_fill_and_update()

    def set_contrast(self, value):
        # value: 0-200, where 100 is 1.0
        self.contrast = value / 100.0
        if self.last_seed is not None:
            self.apply_flood_fill_and_update()

    def set_brightness(self, value):
        # value: -100 to 100
        self.brightness = value
        if self.last_seed is not None:
            self.apply_flood_fill_and_update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.viewer = PDFViewer()

        # Tolerance slider
        self.slider_tol = QSlider(Qt.Horizontal)
        self.slider_tol.setMinimum(0)
        self.slider_tol.setMaximum(100)
        self.slider_tol.setValue(10)
        self.slider_tol.setMaximumWidth(200)
        self.slider_tol.valueChanged.connect(self.on_tol_change)
        self.slider_tol_label = QLabel("Tolerance: 10")
        self.slider_tol_label.setMaximumWidth(200)

        # Contrast slider
        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setMinimum(0)
        self.slider_contrast.setMaximum(200)
        self.slider_contrast.setValue(100)
        self.slider_contrast.setMaximumWidth(200)
        self.slider_contrast.valueChanged.connect(self.on_contrast_change)
        self.slider_contrast_label = QLabel("Contrast: 1.0")
        self.slider_contrast_label.setMaximumWidth(200)

        # Brightness slider
        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setMinimum(-100)
        self.slider_brightness.setMaximum(100)
        self.slider_brightness.setValue(0)
        self.slider_brightness.setMaximumWidth(200)
        self.slider_brightness.valueChanged.connect(self.on_brightness_change)
        self.slider_brightness_label = QLabel("Brightness: 0")
        self.slider_brightness_label.setMaximumWidth(200)

        # Layout: sliders above the viewer
        sliders_layout = QVBoxLayout()
        for label, slider in [
            (self.slider_tol_label, self.slider_tol),
            (self.slider_contrast_label, self.slider_contrast),
            (self.slider_brightness_label, self.slider_brightness)
        ]:
            row = QHBoxLayout()
            row.addWidget(label)
            row.addWidget(slider)
            row.setAlignment(Qt.AlignHCenter)
            sliders_layout.addLayout(row)

        main_layout = QVBoxLayout()
        main_layout.addLayout(sliders_layout)
        main_layout.addWidget(self.viewer)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        open_action = QAction("Open PDF", self)
        open_action.triggered.connect(self.open_pdf)
        self.menuBar().addAction(open_action)

    def open_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        if path:
            self.viewer.load_pdf(path)

    def on_tol_change(self, value):
        self.slider_tol_label.setText(f"Tolerance: {value}")
        self.viewer.set_tolerance(value)

    def on_contrast_change(self, value):
        self.slider_contrast_label.setText(f"Contrast: {value/100:.2f}")
        self.viewer.set_contrast(value)

    def on_brightness_change(self, value):
        self.slider_brightness_label.setText(f"Brightness: {value}")
        self.viewer.set_brightness(value)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("PDF Flood Fill Mask")
    window.resize(900, 700)
    window.show()
    sys.exit(app.exec_())