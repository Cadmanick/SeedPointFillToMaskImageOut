import sys
import math
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QLineEdit, QFileDialog, QTextEdit, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QImage, QColor, QCursor
import os

class CanvasWidget(QLabel):
    left_click = pyqtSignal(int, int)
    right_click = pyqtSignal(int, int)
    middle_press = pyqtSignal(int, int)
    middle_move = pyqtSignal(int, int)
    middle_release = pyqtSignal()
    mouse_wheel = pyqtSignal(int)
    mouse_move = pyqtSignal(int, int)
    resize_event = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setStyleSheet("background: #ddd; border: 1px solid #888;")
        self.setMinimumSize(800, 600)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.left_click.emit(event.x(), event.y())
        elif event.button() == Qt.RightButton:
            self.right_click.emit(event.x(), event.y())
        elif event.button() == Qt.MiddleButton:
            self.middle_press.emit(event.x(), event.y())

    def mouseMoveEvent(self, event):
        buttons = event.buttons()
        if buttons & Qt.MiddleButton:
            self.middle_move.emit(event.x(), event.y())
        self.mouse_move.emit(event.x(), event.y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.middle_release.emit()

    def wheelEvent(self, event):
        self.mouse_wheel.emit(event.angleDelta().y())

    def resizeEvent(self, event):
        self.resize_event.emit(event.size().width(), event.size().height())
        super().resizeEvent(event)

class FloodFillApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flood Fill PDF Mask Generator")
        self.resize(1000, 800)

        # --- State variables ---
        self.canvas_width = 800
        self.canvas_height = 600
        self.image = None
        self.tk_image = None
        self.seed_points = []
        self.mask = None
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_pos = None
        self.viewport = None
        self.ocr_candidate_boxes = []
        self.decimated_contour = None
        self.decimated_epsilon = None
        self.line_points = []
        self.SCALE_FACTOR = None
        self.PIXEL_SCALE = None
        self.original_image = None

        # --- Main layout ---
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Sliders frame above canvas ---
        slider_layout = QHBoxLayout()
        main_layout.addLayout(slider_layout)

        self.aggressiveness_slider = self._make_slider("Aggressiveness", 0, 255, 70, self.update_preview)
        slider_layout.addWidget(self.aggressiveness_slider['widget'])
        slider_layout.addSpacing(20)

        self.pixel_slider = self._make_slider("Gap Pixels", 1, 30, 15)
        slider_layout.addWidget(self.pixel_slider['widget'])
        slider_layout.addSpacing(20)

        self.simplify_slider = self._make_slider("Simplify Contour", 1, 100, 5, self.create_simplified_contour)
        slider_layout.addWidget(self.simplify_slider['widget'])
        slider_layout.addSpacing(20)

        self.contrast_slider = self._make_slider("Contrast", 0, 200, 100, self.update_preview)
        slider_layout.addWidget(self.contrast_slider['widget'])
        slider_layout.addSpacing(20)

        self.kernel_slider = self._make_slider("Kernel Size", 5, 100, 5, self.create_simplified_contour)
        slider_layout.addWidget(self.kernel_slider['widget'])
        slider_layout.addSpacing(20)

        self.lambda_slider = self._make_slider("Lambda", 0, 100, 0, self.update_preview)
        slider_layout.addWidget(self.lambda_slider['widget'])

        # --- Canvas below sliders ---
        self.canvas = CanvasWidget()
        main_layout.addWidget(self.canvas)

        # --- Button row below canvas ---
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)

        self.load_button = QPushButton("Load PDF")
        self.load_button.clicked.connect(self.load_pdf)
        button_layout.addWidget(self.load_button)

        self.clear_button = QPushButton("Clear Canvas")
        self.clear_button.clicked.connect(self.clear_canvas)
        button_layout.addWidget(self.clear_button)

        self.fit_button = QPushButton("Fit to View")
        self.fit_button.clicked.connect(self.fit_to_view)
        button_layout.addWidget(self.fit_button)

        self.extract_button = QPushButton("Extract Distances")
        self.extract_button.clicked.connect(self.extract_text_along_decimated_lines)
        button_layout.addWidget(self.extract_button)

        self.measure_button = QPushButton("Measure Distance")
        self.measure_button.clicked.connect(self.enable_measure_mode)
        button_layout.addWidget(self.measure_button)

        self.simplify_contour_button = QPushButton("Create Simplified Contour")
        self.simplify_contour_button.clicked.connect(self.create_simplified_contour)
        button_layout.addWidget(self.simplify_contour_button)

        self.export_geotiff_button = QPushButton("Export GeoTIFF")
        self.export_geotiff_button.clicked.connect(self.export_geotiff)
        button_layout.addWidget(self.export_geotiff_button)

        # --- Add OCR button ---
        self.ocr_button = QPushButton("OCR Along Contour")
        self.ocr_button.clicked.connect(self.ocr_along_contour)
        button_layout.addWidget(self.ocr_button)

        # --- Bottom: Table, scale input, scale factor label ---
        bottom_layout = QHBoxLayout()
        main_layout.addLayout(bottom_layout)

        # Centered distances table
        table_layout = QVBoxLayout()
        bottom_layout.addLayout(table_layout, stretch=2)

        self.table_label = QLabel("Distances")
        self.table_label.setAlignment(Qt.AlignCenter)
        table_layout.addWidget(self.table_label)

        self.table_text = QTextEdit()
        self.table_text.setReadOnly(True)
        self.table_text.setMinimumWidth(500)
        table_layout.addWidget(self.table_text)

        self.coord_label = QLabel("Mouse: (x, y)")
        table_layout.addWidget(self.coord_label, alignment=Qt.AlignRight)

        self.distance_label = QLabel("Measured Distance: N/A")
        table_layout.addWidget(self.distance_label, alignment=Qt.AlignRight)

        # Scale input (left)
        scale_input_layout = QHBoxLayout()
        bottom_layout.addLayout(scale_input_layout, stretch=1)
        scale_input_layout.addWidget(QLabel("Real-world Distance (feet):"))
        self.real_distance_entry = QLineEdit()
        self.real_distance_entry.setFixedWidth(80)
        scale_input_layout.addWidget(self.real_distance_entry)
        self.calc_scale_button = QPushButton("Calc Scale Factor")
        self.calc_scale_button.clicked.connect(self.calculate_scale_factor)
        scale_input_layout.addWidget(self.calc_scale_button)

        # Scale factor label (right)
        self.scale_factor_label = QLabel("Scale Factor: Not set")
        self.scale_factor_label.setStyleSheet("color: blue;")
        bottom_layout.addWidget(self.scale_factor_label, alignment=Qt.AlignRight)

        # --- Canvas event connections ---
        self.canvas.left_click.connect(self.add_seed_point)
        self.canvas.right_click.connect(self.trace_line)
        self.canvas.middle_press.connect(self.start_pan)
        self.canvas.middle_move.connect(self.pan)
        self.canvas.middle_release.connect(self.reset_mouse_pos)
        self.canvas.mouse_wheel.connect(self.zoom)
        self.canvas.mouse_move.connect(self.show_mouse_coords)
        self.canvas.resize_event.connect(self.on_resize)

    def _make_slider(self, label, minv, maxv, val, slot=None):
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0,0,0,0)
        lab = QLabel(label)
        lab.setAlignment(Qt.AlignCenter)
        s = QSlider(Qt.Horizontal)
        s.setMinimum(minv)
        s.setMaximum(maxv)
        s.setValue(val)
        value_label = QLabel(str(val))
        value_label.setAlignment(Qt.AlignCenter)
       
       # Layout: slider on top, label below, then value label below value label
        l.addWidget(lab)
        l.addWidget(s)
        l.addWidget(value_label)

        # Update value label on slider move
        def update_value_label(v):
            value_label.setText(str(v))
            if slot:
                slot()
        s.valueChanged.connect(update_value_label)
        return {'widget': w, 'slider': s, 'label': lab, 'value_label': value_label}

    # --- Event Handlers (to be ported from Tkinter logic) ---

    def add_seed_point(self, x, y):
        if self.image is None:
            return
        img_x, img_y = self.canvas_to_image_coords(x, y)
        self.seed_points = [(img_x, img_y)]
        self.flood_fill_and_show_mask_contours()

    def trace_line(self, x, y):
        # Port your logic from Tkinter's trace_line here
        pass

    def start_pan(self, x, y):
        # Port your logic from Tkinter's start_pan here
        pass

    def pan(self, x, y):
        # Port your logic from Tkinter's pan here
        pass

    def reset_mouse_pos(self):
        # Port your logic from Tkinter's reset_mouse_pos here
        pass

    def apply_flood_fill(self):
        if self.image is None or not self.seed_points:
            return

        flood_img = self.image.copy()
        h, w = flood_img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        aggressiveness = self.aggressiveness_slider['slider'].value()

        for point in self.seed_points:
            cv2.floodFill(
                flood_img, mask, point, (255, 255, 255),
                (aggressiveness,) * 3, (aggressiveness,) * 3,
                flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
            )

        # Remove border and scale mask for display
        mask = mask[1:-1, 1:-1]

        if np.count_nonzero(mask) == 0:
            print("Flood fill did not mark any mask pixels. Try adjusting aggressiveness or seed point.")
        self.mask = mask * 255

    def flood_fill_and_show_mask_contours(self):
        if self.image is None or not self.seed_points:
            print("No image or seed point.")
            return

        # 1. Flood fill to create mask
        flood_img = self.image.copy()
        h, w = flood_img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        aggressiveness = self.aggressiveness_slider['slider'].value()

        for point in self.seed_points:
            cv2.floodFill(
                flood_img, mask, point, (255, 255, 255),
                (aggressiveness,) * 3, (aggressiveness,) * 3,
                flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
            )

        mask = mask[1:-1, 1:-1]
        if np.count_nonzero(mask) == 0:
            print("Flood fill did not mark any mask pixels. Try adjusting aggressiveness or seed point.")
            return

        self.mask = mask * 255

        # 2. Overlay mask on original image
        overlay = self.original_image.copy()
        overlay[self.mask > 0] = [0, 0, 255]  # Red overlay for mask
        blended = cv2.addWeighted(self.original_image, 0.7, overlay, 0.3, 0)

        # 3. Find contours in the mask and draw them on the overlay
        mask_for_contours = (self.mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 255, 0), 2)  # Green contours

        self.image = blended
        self.update_canvas_image()

    def zoom(self, delta):
        if self.image is None:
            return

        # Mouse wheel delta: positive = zoom in, negative = zoom out
        # Typical delta is 120 per notch, so scale accordingly
        factor = 1.1 if delta > 0 else 0.9

        # Get mouse position on canvas (center if not available)
        mouse_pos = self.canvas.mapFromGlobal(QCursor.pos())
        mouse_x, mouse_y = mouse_pos.x(), mouse_pos.y()
        canvas_w, canvas_h = self.canvas.width(), self.canvas.height()

        # Image coordinates before zoom
        img_h, img_w = self.image.shape[:2]
        rel_x = (mouse_x - self.pan_x - (canvas_w - img_w * self.zoom_level) // 2) / self.zoom_level
        rel_y = (mouse_y - self.pan_y - (canvas_h - img_h * self.zoom_level) // 2) / self.zoom_level

        # Update zoom level, clamp to reasonable range
        new_zoom = self.zoom_level * factor
        min_zoom = min(canvas_w / img_w, canvas_h / img_h) * 0.1
        max_zoom = 8.0
        new_zoom = max(min_zoom, min(new_zoom, max_zoom))

        # Adjust pan so the point under the mouse stays under the mouse
        self.pan_x = int(mouse_x - rel_x * new_zoom - (canvas_w - img_w * new_zoom) // 2)
        self.pan_y = int(mouse_y - rel_y * new_zoom - (canvas_h - img_h * new_zoom) // 2)
        self.zoom_level = new_zoom

        self.update_canvas_image()

    def start_pan(self, x, y):
        self.last_mouse_pos = (x, y)

    def pan(self, x, y):
        if self.last_mouse_pos is None:
            return
        dx = x - self.last_mouse_pos[0]
        dy = y - self.last_mouse_pos[1]
        self.pan_x += dx
        self.pan_y += dy
        self.last_mouse_pos = (x, y)
        self.update_canvas_image()

    def reset_mouse_pos(self):
        self.last_mouse_pos = None

    def show_mouse_coords(self, x, y):
        # Convert widget (canvas) coordinates to image coordinates, considering pan and zoom
        pass

    def on_resize(self, width, height):
        self.canvas_width = width
        self.canvas_height = height
        # Port your logic from Tkinter's on_resize here

    def load_pdf(self):
        # Open file dialog to select PDF
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        if not file_path:
            return

        try:
            import fitz  # PyMuPDF
        except ImportError:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Missing Dependency", "PyMuPDF (fitz) is required to load PDFs.\nInstall with: pip install pymupdf")
            return

        # Load first page of PDF as image
        doc = fitz.open(file_path)
        if doc.page_count == 0:
            return
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        self.image = img
        self.original_image = img.copy()
        self.zoom_level = 0.5  # Set zoom to 50% on load
        self.pan_x = 0
        self.pan_y = 0
        self.seed_points = []
        self.mask = None

        self.update_canvas_image()

    def update_canvas_image(self):
        if self.image is None:
            return
        # Resize image to fit canvas, considering zoom and pan
        h, w = self.image.shape[:2]
        scale = self.zoom_level
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create QImage and QPixmap
        qimg = QImage(resized.data, resized.shape[1], resized.shape[0], resized.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.rgbSwapped())

        # Center or pan
        canvas_w, canvas_h = self.canvas.width(), self.canvas.height()
        display_pixmap = QPixmap(canvas_w, canvas_h)
        display_pixmap.fill(QColor("#ddd"))
        painter = None
        try:
            from PyQt5.QtGui import QPainter
            painter = QPainter(display_pixmap)
            x = (canvas_w - new_w) // 2 + self.pan_x
            y = (canvas_h - new_h) // 2 + self.pan_y
            painter.drawPixmap(x, y, pixmap)
        finally:
            if painter:
                painter.end()
        self.canvas.setPixmap(display_pixmap)

    def clear_canvas(self):
        if hasattr(self, 'original_image') and self.original_image is not None:
            self.image = self.original_image.copy()
        self.found_contours = []
        self.update_canvas_image()

    def fit_to_view(self):
        if self.image is None:
            return

        img_h, img_w = self.image.shape[:2]
        canvas_w, canvas_h = self.canvas.width(), self.canvas.height();

        # Calculate scale to fit image inside canvas
        scale_x = canvas_w / img_w
        scale_y = canvas_h / img_h
        fit_scale = min(scale_x, scale_y)

        self.zoom_level = fit_scale

        # Center the image
        new_w = int(img_w * fit_scale)
        new_h = int(img_h * fit_scale)
        self.pan_x = (canvas_w - new_w) // 2
        self.pan_y = (canvas_h - new_h) // 2

        self.update_canvas_image()

    def extract_text_along_decimated_lines(self):
        # Port your logic from Tkinter's extract_text_along_decimated_lines here
        pass

    def export_geotiff(self):
        # Port your logic from Tkinter's export_geotiff here
        pass

    def enable_measure_mode(self):
        # Port your logic from Tkinter's enable_measure_mode here
        pass

    def calculate_scale_factor(self):
        # Port your logic from Tkinter's calculate_scale_factor here
        pass

    def update_preview(self):
        if self.original_image is None:
            return

        # Get contrast value from slider (0-200, default 100)
        contrast_value = self.contrast_slider['slider'].value()
        alpha = contrast_value / 100.0  # 1.0 = original, <1.0 = less, >1.0 = more

        # Convert to grayscale for line detection
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Detect lines using adaptive threshold (binary image: lines are white)
        lines_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Create a 3-channel mask for color image
        lines_mask_color = cv2.cvtColor(lines_mask, cv2.COLOR_GRAY2BGR)

        # Enhance only the lines: increase their contrast
        enhanced_lines = cv2.convertScaleAbs(self.original_image, alpha=alpha, beta=0)

        # Where lines_mask is white, use enhanced_lines; else use original
        result = np.where(lines_mask_color == 255, enhanced_lines, self.original_image)

        # Continue with flood fill preview as before
        aggressiveness = self.aggressiveness_slider['slider'].value()
        if self.seed_points:
            seed_point = self.seed_points[0]
        else:
            h, w = result.shape[:2]
            seed_point = (w // 2, h // 2)

        h, w = result.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        flood_img = result.copy()
        cv2.floodFill(
            flood_img, mask, seed_point, (0, 0, 255),
            (aggressiveness,) * 3, (aggressiveness,) * 3,
            flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
        )
        mask = mask[1:-1, 1:-1]
        overlay = result.copy()
        overlay[mask != 0] = [0, 0, 255]  # Red
        preview = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

        self.image = preview
        self.update_canvas_image()

    def update_aggressiveness_value_label(self, v):
        self.aggressiveness_slider['value_label'].setText(str(int(float(v))))

    def create_simplified_contour(self):
        if self.mask is None or np.count_nonzero(self.mask) == 0:
            return

        # Apply morphological closing/opening with kernel size from slider
        kernel_size = self.kernel_slider['slider'].value()
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_for_contours = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        else:
            mask_for_contours = self.mask.copy()

        if mask_for_contours.max() > 1:
            mask_for_contours = (mask_for_contours > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_for_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return

        largest_contour = max(contours, key=cv2.contourArea)
        slider_value = self.simplify_slider['slider'].value()  # 1-100
        epsilon = (slider_value / 1000.0) * cv2.arcLength(largest_contour, True)  # 0.001–0.1 * arcLength
        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

        print(f"Original points: {len(largest_contour)}, Simplified: {len(simplified)}, Epsilon: {epsilon:.4f}, Kernel: {kernel_size}")

        self.decimated_contour = simplified

        contour_img = self.original_image.copy()
        cv2.drawContours(contour_img, [simplified], -1, (255, 0, 255), 2)
        self.image = contour_img
        self.update_canvas_image()        

    def show_mask_contours(self):
        if self.mask is None or np.count_nonzero(self.mask) == 0:
            print("No mask available.")
            return

        # Ensure mask is binary
        mask_for_contours = (self.mask > 0).astype(np.uint8) * 255

        # Find contours in the mask only
        contours, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contours found in mask.")
            return

        # Draw contours on a copy of the original image
        contour_img = self.original_image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)  # Green contours

        self.image = contour_img
        self.update_canvas_image()        

    def show_outer_contour(self):
        if self.mask is None or np.count_nonzero(self.mask) == 0:
            print("No mask available.")
            return

        # Ensure mask is binary
        mask_for_contours = (self.mask > 0).astype(np.uint8) * 255

        # Find only the outermost contours
        contours, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contours found in mask.")
            return

        # Use the largest contour as the outer contour
        outer_contour = max(contours, key=cv2.contourArea)

        # Draw the outer contour on a copy of the original image
        contour_img = self.original_image.copy()
        cv2.drawContours(contour_img, [outer_contour], -1, (0, 255, 255), 2)  # Yellow for visibility

        self.image = contour_img
        self.update_canvas_image()        

    def show_first_closed_contour_outside_seed(self):
        if self.mask is None or np.count_nonzero(self.mask) == 0 or not self.seed_points:
            print("No mask or seed point available.")
            return

        # Use the first seed point
        seed_x, seed_y = self.seed_points[0]

        # Ensure mask is binary
        mask_for_contours = (self.mask > 0).astype(np.uint8) * 255

        # Find only the outermost contours
        contours, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contours found in mask.")
            return

        # Find the first closed contour outside the seed point (pointPolygonTest < 0)
        min_abs_dist = float('inf')
        outside_contour = None
        for cnt in contours:
            dist = cv2.pointPolygonTest(cnt, (float(seed_x), float(seed_y)), True)
            if dist < 0:  # Seed is outside this contour
                abs_dist = abs(dist)
                if abs_dist < min_abs_dist:
                    min_abs_dist = abs_dist
                    outside_contour = cnt

        if outside_contour is None:
            print("No closed contour found outside the seed point.")
            return

        # Draw only this contour
        contour_img = self.original_image.copy()
        cv2.drawContours(contour_img, [outside_contour], -1, (255, 0, 0), 2)  # Blue for visibility

        self.image = contour_img
        self.update_canvas_image()

    def canvas_to_image_coords(self, x, y):
        """Convert canvas (widget) coordinates to image coordinates, considering pan and zoom."""
        canvas_w, canvas_h = self.canvas.width(), self.canvas.height()
        img_h, img_w = self.original_image.shape[:2]
        scale = self.zoom_level
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        # Calculate top-left of image in canvas
        img_x0 = (canvas_w - new_w) // 2 + self.pan_x
        img_y0 = (canvas_h - new_h) // 2 + self.pan_y
        # Convert canvas to image coordinates
        img_x = int((x - img_x0) / scale)
        img_y = int((y - img_y0) / scale)
        # Clamp to image bounds
        img_x = np.clip(img_x, 0, img_w - 1)
        img_y = np.clip(img_y, 0, img_h - 1)
        return img_x, img_y

    def ocr_along_contour(self):
        try:
            import pytesseract
        except ImportError:
            print("pytesseract is not installed. Install with: pip install pytesseract")
            return

        if self.decimated_contour is None or len(self.decimated_contour) < 2:
            print("No simplified contour available.")
            return

        img = self.original_image.copy()
        contour = self.decimated_contour.reshape(-1, 2)
        patch_length = 150  # Length of the patch along the segment
        patch_width = 100    # Width of the patch perpendicular to the segment

        # Create output directory for ROI segments
        output_dir = "roi_segments"
        os.makedirs(output_dir, exist_ok=True)

        # For highlighting
        highlight_img = img.copy()

        num_samples_per_segment = 3  # Increase this number for more ROIs per segment

        for i in range(len(contour)):
            pt1 = contour[i]
            pt2 = contour[(i + 1) % len(contour)]
            # Compute the direction vector
            dx, dy = pt2 - pt1
            length = np.hypot(dx, dy)
            if length == 0:
                continue
            # Unit direction vector
            ux, uy = dx / length, dy / length
            # Perpendicular vector
            px, py = -uy, ux

            for s in range(num_samples_per_segment):
                t = (s + 0.5) / num_samples_per_segment  # sample at evenly spaced positions
                cx = pt1[0] + t * dx
                cy = pt1[1] + t * dy

                # Four corners of the patch
                corners = np.array([
                    [cx - ux * patch_length / 2 - px * patch_width / 2, cy - uy * patch_length / 2 - py * patch_width / 2],
                    [cx + ux * patch_length / 2 - px * patch_width / 2, cy + uy * patch_length / 2 - py * patch_width / 2],
                    [cx + ux * patch_length / 2 + px * patch_width / 2, cy + uy * patch_length / 2 + py * patch_width / 2],
                    [cx - ux * patch_length / 2 + px * patch_width / 2, cy - uy * patch_length / 2 + py * patch_width / 2],
                ], dtype=np.float32)

                # Destination rectangle
                dst_rect = np.array([
                    [0, 0],
                    [patch_length - 1, 0],
                    [patch_length - 1, patch_width - 1],
                    [0, patch_width - 1]
                ], dtype=np.float32)

                # Perspective transform
                M = cv2.getPerspectiveTransform(corners, dst_rect)
                patch = cv2.warpPerspective(img, M, (patch_length, patch_width))

                # Save the patch image
                patch_filename = os.path.join(output_dir, f"segment_{i:03d}_sample_{s:02d}.png")
                cv2.imwrite(patch_filename, patch)

                # Convert to RGB for pytesseract
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                text = pytesseract.image_to_string(patch_rgb, config='--psm 6').strip()
                print(f"Segment {i} sample {s}: '{text}' (saved as {patch_filename})")

                # Highlight candidate if text is non-empty
                if text:
                    corners_int = np.int32(corners)
                    cv2.polylines(highlight_img, [corners_int], isClosed=True, color=(0, 255, 255), thickness=2)
                    x, y, w, h = cv2.boundingRect(corners_int)
                    cv2.rectangle(highlight_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(
                        highlight_img, text, (corners_int[0][0], corners_int[0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
                    )

        # Show the highlighted image in the GUI
        self.image = highlight_img
        self.update_canvas_image()        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FloodFillApp()
    win.show()
    sys.exit(app.exec_())