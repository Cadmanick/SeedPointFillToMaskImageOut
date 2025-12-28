import sys
import numpy as np
import cv2
import fitz  # PyMuPDF
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter

class PDFCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.image = None
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_pos = None
        self.mask = None
        self.loDiff = 20
        self.upDiff = 20
        self.gap_size = 1  # Default gap size

    def set_image(self, img):
        self.image = img
        self.mask = None
        self.pan_x = 0
        self.pan_y = 0
        # Fit image to canvas
        canvas_w, canvas_h = self.width(), self.height()
        img_h, img_w = img.shape[:2]
        scale_x = canvas_w / img_w
        scale_y = canvas_h / img_h
        self.zoom_level = min(scale_x, scale_y, 1.0)
        self.update_display()

    def update_display(self):
        if self.image is None:
            self.clear()
            return
        h, w = self.image.shape[:2]
        scale = self.zoom_level
        new_w = int(w * scale)
        new_h = int(h * scale)
        display_img = self.image.copy()
        # Overlay mask if exists
        if self.mask is not None:
            mask_rgb = np.zeros_like(display_img)
            mask_rgb[self.mask > 0] = [0, 0, 255]  # Red overlay
            display_img = cv2.addWeighted(display_img, 0.7, mask_rgb, 0.3, 0)
        resized = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        qimg = QImage(resized.data, resized.shape[1], resized.shape[0], resized.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.rgbSwapped())
        canvas_w, canvas_h = self.width(), self.height()
        display_pixmap = QPixmap(canvas_w, canvas_h)
        display_pixmap.fill(Qt.gray)
        painter = QPainter(display_pixmap)
        x = (canvas_w - new_w) // 2 + self.pan_x
        y = (canvas_h - new_h) // 2 + self.pan_y
        painter.drawPixmap(x, y, pixmap)
        painter.end()
        self.setPixmap(display_pixmap)

    def canvas_to_image_coords(self, x, y):
        canvas_w, canvas_h = self.width(), self.height()
        img_h, img_w = self.image.shape[:2]
        scale = self.zoom_level
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img_x0 = (canvas_w - new_w) // 2 + self.pan_x
        img_y0 = (canvas_h - new_h) // 2 + self.pan_y
        img_x = int((x - img_x0) / scale)
        img_y = int((y - img_y0) / scale)
        img_x = np.clip(img_x, 0, img_w - 1)
        img_y = np.clip(img_y, 0, img_h - 1)
        return img_x, img_y

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.last_mouse_pos = (event.x(), event.y())
        elif event.button() == Qt.LeftButton and self.image is not None:
            img_x, img_y = self.canvas_to_image_coords(event.x(), event.y())
            self.perform_flood_fill(img_x, img_y)
        elif event.button() == Qt.RightButton:
            self.mask = None
            if hasattr(self, 'last_seed'):
                del self.last_seed
            self.update_display()

    def perform_flood_fill(self, img_x, img_y):
        h, w = self.image.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # Preprocess image to close gaps
        proc_img = self.image.copy()
        if self.gap_size > 1:
            kernel = np.ones((self.gap_size, self.gap_size), np.uint8)
            proc_img = cv2.morphologyEx(proc_img, cv2.MORPH_CLOSE, kernel)
        cv2.floodFill(
            proc_img, mask, (img_x, img_y), (255, 255, 255),
            (self.loDiff,)*3, (self.upDiff,)*3,
            flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
        )
        mask = mask[1:-1, 1:-1]
        self.mask = (mask > 0).astype(np.uint8) * 255
        self.last_seed = (img_x, img_y)
        self.update_display()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MiddleButton and self.last_mouse_pos:
            dx = event.x() - self.last_mouse_pos[0]
            dy = event.y() - self.last_mouse_pos[1]
            self.pan_x += dx
            self.pan_y += dy
            self.last_mouse_pos = (event.x(), event.y())
            self.update_display()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.last_mouse_pos = None

    def wheelEvent(self, event):
        if self.image is None:
            return
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        mouse_pos = event.pos()
        mouse_x, mouse_y = mouse_pos.x(), mouse_pos.y()
        h, w = self.image.shape[:2]
        scale = self.zoom_level
        new_w = int(w * scale)
        new_h = int(h * scale)
        rel_x = (mouse_x - self.pan_x - (self.width() - new_w) // 2) / scale
        rel_y = (mouse_y - self.pan_y - (self.height() - new_h) // 2) / scale
        new_zoom = max(0.1, min(self.zoom_level * factor, 8.0))
        self.pan_x = int(mouse_x - rel_x * new_zoom - (self.width() - w * new_zoom) // 2)
        self.pan_y = int(mouse_y - rel_y * new_zoom - (self.height() - h * new_zoom) // 2)
        self.zoom_level = new_zoom
        self.update_display()

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)

class PDFViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Viewer")
        self.resize(1000, 800)
        self.showMaximized()  # Maximize window
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        self.canvas = PDFCanvas()
        main_layout.addWidget(self.canvas)

        # Add Load PDF button
        self.load_button = QPushButton("Load PDF")
        self.load_button.clicked.connect(self.load_pdf)
        main_layout.addWidget(self.load_button)

    def load_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        if not file_path:
            return
        doc = fitz.open(file_path)
        if doc.page_count == 0:
            return
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self.canvas.set_image(img)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PDFViewer()
    win.show()
    sys.exit(app.exec_())