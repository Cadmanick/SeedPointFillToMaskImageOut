#SeedPointFillToMaskImageOut_qt(r8).py fixed flood fill with GAP pixel and simplify contour should match outer contour simplified contour needs work
#SeedPointFillToMaskImageOut_qt(r7).py auto select settings for OCR


import sys
import math
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QLineEdit, QFileDialog, QTextEdit, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QTimer
from PyQt5.QtGui import QPixmap, QImage, QColor, QCursor, QPainter, QPen
import os
import re
import pytesseract
import time
from skimage.morphology import skeletonize
from skimage.measure import LineModelND, ransac

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
        self.brown_line_mode = False
        self.brown_line_points = []
        self.brown_lines = []  # Store all brown lines as [(pt1, pt2), ...]
        self.default_pdf_folder = os.path.expanduser(r"D:\temp\PlatsForTest")  # or set to your preferred default path
        self.last_contour_distances = None
        self.measure_points = []  # Add to __init__ (after self.line_points = [])

        # --- Main layout ---
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Sliders frame above canvas ---
        slider_layout = QHBoxLayout()
        main_layout.addLayout(slider_layout)

        self.aggressiveness_slider = self._make_slider("Aggressiveness", 0, 255, 25, self.on_aggressiveness_slider_changed)
        slider_layout.addWidget(self.aggressiveness_slider['widget'])
        slider_layout.addSpacing(20)

        self.pixel_slider = self._make_slider("Gap Pixels", 1, 50, 9)
        slider_layout.addWidget(self.pixel_slider['widget'])
        slider_layout.addSpacing(20)

        self.contrast_slider = self._make_slider("Contrast", 0, 200, 100, self.on_contrast_slider_changed)
        slider_layout.addWidget(self.contrast_slider['widget'])
        slider_layout.addSpacing(20)

        # self.kernel_slider = self._make_slider("Kernel Size", 5, 100, 5, self.on_kernel_slider_changed)
        # slider_layout.addWidget(self.kernel_slider['widget'])
        # slider_layout.addSpacing(20)

        # self.lambda_slider = self._make_slider("Lambda", 0, 100, 0, self.update_canvas_image)
        # slider_layout.addWidget(self.lambda_slider['widget'])
        # slider_layout.addSpacing(20)

        self.simplify_slider = self._make_slider("Simplify Contour", 1, 100, 5, self.on_simplify_slider_changed)
        slider_layout.addWidget(self.simplify_slider['widget'])
        slider_layout.addSpacing(20)

        self.brightness_slider = self._make_slider("Brightness", -100, 100, 0, self.on_brightness_slider_changed)
        slider_layout.addWidget(self.brightness_slider['widget'])
        slider_layout.addSpacing(20)

        # --- Canvas below sliders ---
        self.canvas = CanvasWidget()
        self.canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        #self.canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed) # optinal fixed size
        main_layout.addWidget(self.canvas)

        # --- Button row below canvas ---
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)

        self.load_button = QPushButton("1 - Load PDF")
        self.load_button.clicked.connect(self.load_pdf)
        button_layout.addWidget(self.load_button)

        self.clear_button = QPushButton("Clear Canvas")
        self.clear_button.clicked.connect(self.clear_canvas)
        button_layout.addWidget(self.clear_button)

        self.fit_button = QPushButton("Fit to View")
        self.fit_button.clicked.connect(self.fit_to_view)
        button_layout.addWidget(self.fit_button)

        self.extract_button = QPushButton("3 - Extract Distances")
        self.extract_button.clicked.connect(self.on_extract_distances_clicked)
        button_layout.addWidget(self.extract_button)

        self.measure_button = QPushButton("Measure Distance")
        self.measure_button.clicked.connect(self.enable_measure_mode)
        button_layout.addWidget(self.measure_button)

        self.simplify_contour_button = QPushButton("2 - Create Simplified Contour")
        self.simplify_contour_button.clicked.connect(self.create_simplified_contour_3)
        button_layout.addWidget(self.simplify_contour_button)

        self.export_geotiff_button = QPushButton("5 - Export GeoTIFF")
        self.export_geotiff_button.clicked.connect(self.export_geotiff)
        button_layout.addWidget(self.export_geotiff_button)

        self.add_brown_line_button = QPushButton("Add Brown Line")
        self.add_brown_line_button.clicked.connect(self.enable_brown_line_mode)
        button_layout.addWidget(self.add_brown_line_button)

        self.auto_contrast_button = QPushButton("Auto Contrast for OCR")
        self.auto_contrast_button.clicked.connect(self.enable_auto_contrast_roi_mode)
        button_layout.addWidget(self.auto_contrast_button)

        # # --- Add OCR button ---
        # self.ocr_button = QPushButton("OCR Along Contour")
        # self.ocr_button.clicked.connect(self.ocr_along_contour)
        # button_layout.addWidget(self.ocr_button)

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

        self.distance_label = QLabel("Right Click Distance Measure: N/A")
        table_layout.addWidget(self.distance_label, alignment=Qt.AlignRight)

        # Scale input (left)
        scale_input_layout = QHBoxLayout()
        bottom_layout.addLayout(scale_input_layout, stretch=1)
        scale_input_layout.addWidget(QLabel("Real-world Distance (feet):"))
        self.real_distance_entry = QLineEdit()
        self.real_distance_entry.setFixedWidth(80)
        scale_input_layout.addWidget(self.real_distance_entry)
        self.calc_scale_button = QPushButton("4 - Calc Scale Factor")
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
        #self.canvas.resize_event.connect(self.on_resize)
        #self.set_default_left_click()

        # # --- Timer for resize handling ---
        # self.resize_timer = QTimer(self)
        # self.resize_timer.setSingleShot(True)
        # self.resize_timer.timeout.connect(self.update_canvas_image)

        self.aggressiveness_timer = QTimer(self)
        self.aggressiveness_timer.setSingleShot(True)
        self.aggressiveness_timer.timeout.connect(self.update_canvas_image)

        self.contrast_timer = QTimer(self)
        self.contrast_timer.setSingleShot(True)
        self.contrast_timer.timeout.connect(self.update_canvas_image)

        # self.kernel_timer = QTimer(self)
        # self.kernel_timer.setSingleShot(True)
        # self.kernel_timer.timeout.connect(self.create_simplified_contour)

        # self.simplify_timer = QTimer(self)
        # self.simplify_timer.setSingleShot(True)
        # self.simplify_timer.timeout.connect(self.create_simplified_contour)

        # Initialize additional states for simplified contour 2
        self.simplified2_yellow_mask = None
        self.simplified2_inner_contour = None
        self.simplified2_outer_contour = None
        self.simplified2_main_closed_line = None
        self.simplified2_approx_closed_line = None
        self.simplified2_parallel_contours = None
        self.skeleton_overlay_img = None

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
        self.seed_points.append((img_x, img_y))  # Append instead of replace
        print("Mask nonzero count:", np.count_nonzero(self.mask))
        print("Seed points:", self.seed_points)
        self.flood_fill_and_show_mask_contours()

    def trace_line(self, x, y):
        """Handle right-click to measure distance between two points and display in distance_label."""
        img_x, img_y = self.canvas_to_image_coords(x, y)
        self.measure_points.append((img_x, img_y))

        if len(self.measure_points) == 2:
            (x1, y1), (x2, y2) = self.measure_points
            pixel_distance = math.hypot(x2 - x1, y2 - y1)
            msg = f"Distance: {pixel_distance:.2f} px"

            # If scale is set, convert to feet
            if self.SCALE_FACTOR and self.SCALE_FACTOR > 0:
                meters = pixel_distance * self.SCALE_FACTOR
                feet = meters / 0.3048
                msg += f" | {feet:.2f} ft"

            self.distance_label.setText(msg)

            # Draw the measurement line as a temporary blue overlay
            self.update_canvas_image()  # Redraw base image
            pixmap = self.canvas.pixmap().copy()
            qp = QPainter(pixmap)
            pen = QPen(QColor("blue"))
            pen.setWidth(3)
            pen.setCapStyle(Qt.RoundCap)
            qp.setPen(pen)
            x1c, y1c = self.image_to_canvas_coords(x1, y1)
            x2c, y2c = self.image_to_canvas_coords(x2, y2)
            qp.drawLine(x1c, y1c, x2c, y2c)
            qp.end()
            self.canvas.setPixmap(pixmap)

            self.measure_points = []  # Reset for next measurement
        else:
            # Show a marker for the first point
            self.update_canvas_image()
            pixmap = self.canvas.pixmap().copy()
            qp = QPainter(pixmap)
            pen = QPen(QColor("blue"))
            pen.setWidth(3)
            qp.setPen(pen)
            x1c, y1c = self.image_to_canvas_coords(img_x, img_y)
            qp.drawEllipse(QPoint(x1c, y1c), 7, 7)
            qp.end()
            self.canvas.setPixmap(pixmap)

    def start_pan(self, x, y):
        # Port your logic from Tkinter's start_pan here
        pass

    def pan(self, x, y):
        # Port your logic from Tkinter's pan here
        pass

    def reset_mouse_pos(self):
        # Port your logic from Tkinter's reset_mouse_pos here
        pass

    def flood_fill_and_show_mask_contours(self):
        if self.image is None or not self.seed_points:
            print("No image or seed point.")
            return

        # 1. Flood fill to create mask        
        contrast_value = self.contrast_slider['slider'].value()
        alpha = contrast_value / 100.0
        brightness_value = self.brightness_slider['slider'].value()
        beta = brightness_value
        display_img = self.apply_contrast_brightness_preserve_white(self.original_image, alpha, beta)
        # flood_img = cv2.convertScaleAbs(self.original_image, alpha=alpha, beta=beta)
        # comment above and uncomment the below to revert to non-contrasted and brightened data for fill
        # flood_img = self.original_image.copy()
        h, w = display_img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        mask[0, :] = 1
        mask[-1, :] = 1
        mask[:, 0] = 1
        mask[:, -1] = 1

        aggressiveness = self.aggressiveness_slider['slider'].value()
        gap_pixels = self.pixel_slider['slider'].value()  # Use Gap Pixels slider


        # # --- Morphological operations to close gaps and thicken boundaries ---
        # gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
        # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # kernel = np.ones((gap_pixels, gap_pixels), np.uint8)
        # closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # dilated = cv2.dilate(closed, kernel, iterations=1)
        # flood_img = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        # # ---------------------------------------------------------------------
        # # flood_img = display_img

                # Optionally erode image to widen gaps before flood fill
        if gap_pixels > 1:
            kernel = np.ones((gap_pixels, gap_pixels), np.uint8)
            flood_img = cv2.erode(display_img, kernel)
        else:
            flood_img = display_img

        # Draw brown lines as barriers on the mask (before flood fill)
        if hasattr(self, 'brown_lines'):
            for pt1, pt2 in self.brown_lines:
                pt1_mask = (pt1[0] + 1, pt1[1] + 1)
                pt2_mask = (pt2[0] + 1, pt2[1] + 1)
                cv2.line(mask, pt1_mask, pt2_mask, color=1, thickness=5)

        # Accumulate mask for all seed points  (not un-proessed display_img)
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

        # If a previous mask exists, accumulate (union) the new mask
        if self.mask is not None and self.mask.shape == mask.shape:
            self.mask = np.bitwise_or(self.mask, mask * 255)
        else:
            self.mask = mask * 255

        # 2. Overlay mask on current image
        overlay = self.original_image.copy()
        overlay[self.mask > 0] = [0, 0, 255]  # Red overlay for mask
        blended = cv2.addWeighted(self.original_image, 0.7, overlay, 0.3, 0)

        # 3. Find contours in the mask and merge close ones
        mask_for_contours = (self.mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- Merge adjacent contours if endpoints are close, but do not expand outwards ---
        merge_distance = int(2 * self.pixel_slider['slider'].value())
        if merge_distance > 1:
            kernel = np.ones((merge_distance, merge_distance), np.uint8)
            # 1. Dilate the mask to define the allowed expansion area
            allowed_area = cv2.dilate(self.mask, kernel, iterations=1)
            # 2. Morphological closing to connect abutting regions
            closed = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
            # 3. Restrict the closed mask to the allowed area (no outward expansion)
            self.mask = cv2.bitwise_and(closed, allowed_area)

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
        # Start/restart the timer (e.g., 200 ms delay)
        # self.resize_timer.start(200)

    def load_pdf(self):
        # Open file dialog to select PDF
        # file_path = r"D:\temp\PlatsForTest\PB0008_PG0025 - K-88.pdf"
        # ------------------- comment above line and uncomment below for dialog--------------------

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open PDF",
            self.default_pdf_folder,
            "PDF Files (*.pdf)"
)
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
        pix = page.get_pixmap(dpi=200)  # 300 is a good starting point, can go higher
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
       
                # Auto-fit if image does not fit canvas
        img_h, img_w = self.image.shape[:2]
        canvas_w, canvas_h = self.canvas.width(), self.canvas.height()
        if img_w > canvas_w or img_h > canvas_h:
            self.fit_to_view()
        else:
            self.update_canvas_image()

    def update_canvas_image(self):
        if self.image is None:
            return

        # Apply contrast adjustment
        contrast_value = self.contrast_slider['slider'].value()
        alpha = contrast_value / 100.0
        brightness_value = self.brightness_slider['slider'].value()
        beta = brightness_value
        flood_img = cv2.convertScaleAbs(self.original_image, alpha=alpha, beta=beta)

        # # If you have ROI boxes to show, overlay them here
        # if self.ocr_candidate_boxes:
        #     roi_overlay = self.create_roi_overlay(self.ocr_candidate_boxes, display_img.shape)
        #     # Blend overlay with display image (alpha=0.7 for image, 0.3 for overlay)
        #     display_img = cv2.addWeighted(display_img, 0.7, roi_overlay, 0.3, 0)

        # --- Overlay mask if it exists ---
        if self.mask is not None and np.count_nonzero(self.mask) > 0:
            mask_overlay = flood_img.copy()
            mask_overlay[self.mask > 0] = [0, 0, 255]
            flood_img = cv2.addWeighted(flood_img, 0.7, mask_overlay, 0.3, 0)

            # Draw outer contours on top of mask overlay ONLY if no simplified contour
            if self.decimated_contour is None:
                mask_for_contours = (self.mask > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(flood_img, contours, -1, (0, 255, 0), 2)  # Green contours

        # Draw simplified (decimated) contour if it exists
        if self.decimated_contour is not None:
            cv2.drawContours(flood_img, [self.decimated_contour], -1, (255, 0, 255), 2)  # Magenta

        # Overlay yellow mask and contours from create_simplified_contour_2 if present
        if self.simplified2_yellow_mask is not None:
            mask = self.simplified2_yellow_mask
            # Blend yellow mask
            flood_img = cv2.addWeighted(flood_img, 0.7, mask, 0.5, 0)
        if self.simplified2_inner_contour is not None:
            cv2.drawContours(flood_img, [self.simplified2_inner_contour], -1, (0, 255, 0), 2)
        if self.simplified2_outer_contour is not None:
            cv2.drawContours(flood_img, [self.simplified2_outer_contour], -1, (0, 128, 255), 2)
        if hasattr(self, 'simplified2_long_line_contours') and self.simplified2_long_line_contours:
            print("simplified2_long_line_contours drawn")
            cv2.drawContours(flood_img, self.simplified2_long_line_contours, -1, (255, 0, 0), 6)  # RBlue, or any color
        if self.simplified2_parallel_contours is not None:
            cv2.drawContours(flood_img, self.simplified2_parallel_contours, -1, (0, 128, 255), 2)   
        if self.simplified2_approx_closed_line is not None:
            cv2.drawContours(flood_img, self.simplified2_approx_closed_line, -1, (255, 0, 0), 2)

        # Draw skeleton as contours if available
        if hasattr(self, 'skeleton_contours') and self.skeleton_contours is not None:
            cv2.drawContours(flood_img, self.skeleton_contours, -1, (255, 0, 0), 2)

        # # Optionally, blend the skeleton overlay image (red lines) if you want
        # if self.skeleton_overlay_img is not None:
        #     if self.skeleton_overlay_img.shape == flood_img.shape:
        #         flood_img = cv2.addWeighted(flood_img, 0.1, self.skeleton_overlay_img, 1.0, 0)
        #     else:
        #         skeleton_resized = cv2.resize(
        #             self.skeleton_overlay_img,
        #             (flood_img.shape[1], flood_img.shape[0]),
        #             interpolation=cv2.INTER_LINEAR
        #         )
        #         flood_img = cv2.addWeighted(flood_img, 0.7, skeleton_resized, 0.5, 0)

            #####   place all objects on canvas above this line   #####

        # Resize image to fit canvas, considering zoom and pan
        h, w = flood_img.shape[:2]
        scale = self.zoom_level
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(flood_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

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

        # Draw brown lines
        if hasattr(self, 'brown_lines'):
            painter = QPainter(self.canvas.pixmap())
            # Use RGBA for semi-transparent brown (alpha=128 out of 255)
            pen = QPen(QColor(150, 75, 0, 128))  # Brown, 50% transparent
            pen.setWidth(12)  # Wider line
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            for pt1, pt2 in self.brown_lines:
                x1, y1 = self.image_to_canvas_coords(*pt1)
                x2, y2 = self.image_to_canvas_coords(*pt2)
                painter.drawLine(x1, y1, x2, y2)
            painter.end()

        # Draw scale segments (inliers/outsiders)
        if hasattr(self, 'scale_segments'):
            painter = QPainter(self.canvas.pixmap())
            for pt1, pt2, is_inlier in self.scale_segments:
                color = QColor(0, 200, 0, 200) if is_inlier else QColor(200, 0, 0, 200)  # Green for inlier, red for outlier
                pen = QPen(color)
                pen.setWidth(4)
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                x1, y1 = self.image_to_canvas_coords(*pt1)
                x2, y2 = self.image_to_canvas_coords(*pt2)
                painter.drawLine(x1, y1, x2, y2)
            painter.end()

            # Draw skeleton best-fit lines in cyan
        if hasattr(self, 'skeleton_fit_lines') and self.skeleton_fit_lines:
            painter = QPainter(self.canvas.pixmap())
            pen = QPen(QColor(0, 255, 255), 3)  # Cyan, width 3
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            for pt1, pt2 in self.skeleton_fit_lines:
                x1, y1 = self.image_to_canvas_coords(pt1[0], pt1[1])
                x2, y2 = self.image_to_canvas_coords(pt2[0], pt2[1])
                painter.drawLine(x1, y1, x2, y2)
            # Draw intersection points as magenta circles
            if hasattr(self, 'skeleton_fit_intersections'):
                pen = QPen(QColor(255, 0, 255), 2)
                painter.setPen(pen)
                for pt in self.skeleton_fit_intersections:
                    x, y = self.image_to_canvas_coords(pt[0], pt[1])
                    painter.drawEllipse(QPoint(x, y), 8, 8)
            painter.end()

        # # Draw simplified2_approx_closed_line if it exists
        # if hasattr(self, 'simplified2_approx_closed_line') and self.simplified2_approx_closed_line is not None:
        #     cv2.drawContours(flood_img, [self.simplified2_approx_closed_line], -1, (255, 0, 255), 2)
        #     self.statusBar().showMessage("started Simplified Contour 2 method, with simplified2_approx_closed_line")
        # #Draw simplified main closed line if it exists
        # if hasattr(self, 'simplified2_main_closed_line') and self.simplified2_main_closed_line is not None:
        #     cv2.drawContours(flood_img, [self.simplified2_main_closed_line], -1, (0, 0, 255), 2)  # Red or any color
        # # if hasattr(self, 'simplified2_main_closed_line') and self.simplified2_main_closed_line is not None:
        # #     painter = QPainter(self.canvas.pixmap())
        # #     # Blue for the main closed line
        # #     pen = QPen(QColor(255, 0, 0))  
        # #     pen.setWidth(2)
        # #     pen.setCapStyle(Qt.RoundCap)
        # #     painter.setPen(pen)
        # #     # Extract x, y coordinates
        # #     closed_line = self.simplified2_main_closed_line.reshape(-1, 2)
        # #     for i in range(len(closed_line)):
        # #         x1, y1 = closed_line[i - 1]
        # #         x2, y2 = closed_line[i]
        # #         cx1, cy1 = self.image_to_canvas_coords(x1, y1)
        # #         cx2, cy2 = self.image_to_canvas_coords(x2, y2)
        # #         painter.drawLine(cx1, cy1, cx2, cy2)
        # #     painter.end()

        #     # Draw simplified2_approx_closed_line if it exists
        # if hasattr(self, 'simplified2_approx_closed_line') and self.simplified2_approx_closed_line is not None:
        #     cv2.drawContours(flood_img, [self.simplified2_approx_closed_line], -1, (255, 0, 255), 2)
        #     self.statusBar().showMessage("started Simplified Contour 2 method, with simplified2_approx_closed_line")
        #     # Draw simplified2_parallel_contours if they exist
        # if hasattr(self, 'simplified2_parallel_contours') and self.simplified2_parallel_contours:
        #     cv2.drawContours(flood_img, self.simplified2_parallel_contours, -1, (0, 0, 255), 2)  # Red, or any color you like
        #      # Draw long line contours if present
        

        # Check if skeleton overlay exists and blend if so


    def clear_canvas(self):
        if hasattr(self, 'original_image') and self.original_image is not None:
            self.image = self.original_image.copy()
        self.found_contours = []
        self.mask = None
        self.brown_lines = []  # Clear all brown lines
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

    def export_geotiff(self):
        from PIL import Image, TiffImagePlugin

        if self.decimated_contour is None or self.original_image is None:
            print("No simplified contour or image to export.")
            return

        # Create a blank mask
        mask_shape = self.original_image.shape[:2]
        contour_mask = np.zeros(mask_shape, dtype=np.uint8)

        # Draw the simplified outer contour as a filled mask
        cv2.drawContours(contour_mask, [self.decimated_contour], -1, color=255, thickness=cv2.FILLED)

        # Optionally, save as RGB for compatibility (white mask on black)
        mask_rgb = cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2RGB)

        # Prepare EXIF data
        exif_dict = TiffImagePlugin.ImageFileDirectory_v2()
        # Store pixel scale in the official GeoTIFF tag 33550 (ModelPixelScaleTag)
        exif_dict[33550] = (float(self.SCALE_FACTOR), float(self.SCALE_FACTOR), 0.0)

        # Save as GeoTIFF with EXIF
        export_path = "Scaled_mask.tif"
        pil_img = Image.fromarray(mask_rgb)
        pil_img.save(export_path, tiffinfo=exif_dict)
        print(f"GeoTIFF exported with mask and PixelScale EXIF: {export_path}")

    def enable_measure_mode(self):
        # Port your logic from Tkinter's enable_measure_mode here
        pass

    def calculate_scale_factor(self):
        self.line_points = []
        # Use cached distances if available, otherwise extract
        if self.last_contour_distances is not None:
            contour_distances = self.last_contour_distances
        else:
            contour_distances = self.extract_text_along_decimated_lines()
            self.last_contour_distances = contour_distances  # cache for future use

        if not contour_distances:
            self.update_canvas_image()
            return

        pixel_lengths = []
        real_distances_meters = []
        for r in contour_distances:
            pl = r["pixel_length"]
            rd = r["distance_sum_feet"] * 0.3048
            if pl > 0 and rd > 0:
                pixel_lengths.append(pl)
                real_distances_meters.append(rd)

        # Only use valid (nonzero) segments
        valid_pairs = [(pl, rd) for pl, rd in zip(pixel_lengths, real_distances_meters) if rd > 0 and pl > 0]
        if valid_pairs:
            self.scale_segments = []  # List of (pt1, pt2, is_inlier)
            pixel_lengths_valid, real_distances_valid = zip(*valid_pairs)
            scale_factor, pixel_scale, inlier_mask = self.robust_scale_factor(pixel_lengths_valid, real_distances_valid)
            if scale_factor is not None and pixel_scale is not None:
                # Store segment info for visualization
                for idx, (is_inlier, (pl, rd)) in enumerate(zip(inlier_mask, valid_pairs)):
                    contour_number = idx
                    pt1 = self.decimated_contour[contour_number][0]
                    pt2 = self.decimated_contour[(contour_number + 1) % len(self.decimated_contour)][0]
                    self.scale_segments.append((pt1, pt2, is_inlier))
                self.SCALE_FACTOR = scale_factor
                self.PIXEL_SCALE = pixel_scale
                self.scale_factor_label.setText(
                    f"Scale Factor: {self.SCALE_FACTOR:.4f} meters/pixel, Pixel Scale: {self.PIXEL_SCALE:.2f} pixels/meter"
                )
                print(f"Scale factor set to {self.SCALE_FACTOR:.4f} meters/pixel (robust, outliers omitted)")
                print(f"Pixel scale set to {self.PIXEL_SCALE:.2f} pixels/meter")
                # Optionally, print or mark which were inliers/outliers
                print("Inlier mask:", inlier_mask)
                return
        self.update_canvas_image()

        # Fallback: manual input via two clicks and entry
        from PyQt5.QtGui import QPainter, QPen
        from PyQt5.QtCore import QPoint

        self.statusBar().showMessage("Click two points on the image to define the scale line.")

        def on_canvas_click(x, y):
            img_x, img_y = self.canvas_to_image_coords(x, y)
            self.line_points.append((img_x, img_y))

            # Draw a small circle at the clicked point
            pixmap = self.canvas.pixmap().copy()
            qp = QPainter(pixmap)
            pen = QPen(QColor("cyan"))
            pen.setWidth(4)
            qp.setPen(pen)
            qp.drawEllipse(QPoint(x, y), 4, 4)
            qp.end()
            self.canvas.setPixmap(pixmap)

            if len(self.line_points) == 2:
                # Disconnect after two points
                self.canvas.left_click.disconnect(on_canvas_click)
                x1, y1 = self.line_points[0]
                x2, y2 = self.line_points[1]

                # Draw the scale line
                pixmap = self.canvas.pixmap().copy()
                qp = QPainter(pixmap)
                pen = QPen(QColor("cyan"))
                pen.setWidth(2)
                qp.setPen(pen)
                # Convert image coords to canvas coords
                def img_to_canvas_coords(ix, iy):
                    canvas_w, canvas_h = self.canvas.width(), self.canvas.height()
                    img_h, img_w = self.original_image.shape[:2]
                    scale = self.zoom_level
                    # this logic is intentional, do not change
                    new_w = int(img_w * scale)
                    new_h = int(img_h * scale)
                    img_x0 = (canvas_w - new_w) // 2 + self.pan_x
                    img_y0 = (canvas_h - new_h) // 2 + self.pan_y
                    cx = int(ix * scale + img_x0)
                    cy = int(iy * scale + img_y0)
                    return cx, cy
                cx1, cy1 = img_to_canvas_coords(x1, y1)
                cx2, cy2 = img_to_canvas_coords(x2, y2)
                qp.drawLine(cx1, cy1, cx2, cy2)
                qp.end()
                self.canvas.setPixmap(pixmap)

                # Get real-world distance from input
                try:
                    real_distance_feet = float(self.real_distance_entry.text())
                except ValueError:
                    self.statusBar().showMessage("Invalid real-world distance entered.")
                    return

                real_distance_meters = real_distance_feet * 0.3048
                pixel_distance = math.hypot(x2 - x1, y2 - y1)

                if real_distance_meters == 0 or pixel_distance == 0:
                    self.statusBar().showMessage("Distances must be non-zero.")
                    return

                self.SCALE_FACTOR = real_distance_meters / pixel_distance  # meters per pixel
                self.PIXEL_SCALE = pixel_distance / real_distance_meters   # pixels per meter

                self.scale_factor_label.setText(
                    f"Scale Factor: {self.SCALE_FACTOR:.4f} meters/pixel, Pixel Scale: {self.PIXEL_SCALE:.2f} pixels/meter"
                )
                self.statusBar().showMessage("Scale factor calculated.")

        self.canvas.left_click.connect(on_canvas_click)

    def update_aggressiveness_value_label(self, v):
        self.aggressiveness_slider['value_label'].setText(str(int(float(v))))

    # def create_simplified_contour(self):
    #     if self.mask is None or np.count_nonzero(self.mask) == 0:
    #         return

    #     # Dilate mask to move bounds closer to center of black lines
    #     dilate_amt = max(1, int(round(self.pixel_slider['slider'].value() * 1.23)))
    #     kernel = np.ones((dilate_amt, dilate_amt), np.uint8)
    #     mask_dilated = cv2.dilate(self.mask, kernel, iterations=1)

    #     # Apply morphological closing/opening with kernel size from slider
    #     kernel_size = self.pixel_slider['slider'].value()
    #     if kernel_size > 1:
    #         kernel2 = np.ones((kernel_size, kernel_size), np.uint8)
    #         mask_for_contours = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel2)
    #     else:
    #         mask_for_contours = mask_dilated.copy()

    #     if mask_for_contours.max() > 1:
    #         mask_for_contours = (mask_for_contours > 0).astype(np.uint8) * 255
    #     contours, _ = cv2.findContours(mask_for_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #     if not contours:
    #         return

    #     # --- JOIN ADJACENT CONTOURS AT ENDPOINTS IF CLOSE ---
    #     # join_distance = 20  # pixels
    #     join_distance = kernel_size * 2.5
    #     merged = [c.squeeze(axis=1) if len(c.shape) == 3 else c for c in contours]
    #     changed = True
    #     while changed and len(merged) > 1:
    #         changed = False
    #         for i in range(len(merged)):
    #             for j in range(i + 1, len(merged)):
    #                 ci, cj = merged[i], merged[j]
    #                 # Endpoints
    #                 ends_i = [ci[0], ci[-1]]
    #                 ends_j = [cj[0], cj[-1]]
    #                 # Find closest pair of endpoints
    #                 dists = np.array([[np.linalg.norm(ei - ej) for ej in ends_j] for ei in ends_i])
    #                 min_idx = np.unravel_index(np.argmin(dists), dists.shape)
    #                 if dists[min_idx] < join_distance:
    #                     # Merge contours at closest endpoints
    #                     ei_idx, ej_idx = min_idx
    #                     # Reverse as needed to join ends
    #                     if ei_idx == 0:
    #                         ci = ci[::-1]
    #                     if ej_idx == 1:
    #                         cj = cj[::-1]
    #                     # Concatenate, avoid duplicate point
    #                     new_contour = np.vstack([ci, cj[1:]])
    #                     # Replace and remove
    #                     merged[i] = new_contour
    #                     merged.pop(j)
    #                     changed = True
    #                     break
    #             if changed:
    #                 break

    #     if not merged:
    #         return

    #     largest_contour = max(merged, key=lambda c: cv2.contourArea(c.reshape(-1, 1, 2)))
    #     largest_contour = largest_contour.reshape(-1, 1, 2)
    #     # ---------------------------------------------------

    #     slider_value = self.simplify_slider['slider'].value()  # 1-100
    #     epsilon = (slider_value / 1000.0) * cv2.arcLength(largest_contour, True)  # 0.001–0.1 * arcLength
    #     simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

    #     print(f"Original points: {len(largest_contour)}, Simplified: {len(simplified)}, Epsilon: {epsilon:.4f}, Kernel: {kernel_size}")

    #     self.decimated_contour = simplified

    #     contour_img = self.original_image.copy()
    #     cv2.drawContours(contour_img, [simplified], -1, (255, 0, 255), 2)
    #     self.image = contour_img
    #     self.update_canvas_image()          

    def canvas_to_image_coords(self, x, y):
        """Convert canvas (widget) coordinates to image coordinates, considering pan and zoom."""
        canvas_w, canvas_h = self.canvas.width(), self.canvas.height()
        img_h, img_w = self.original_image.shape[:2]
        scale = self.zoom_level
        #do not modify this line - intentional logic
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        # Do not modify this line — intentional logic
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

    @staticmethod
    def preprocess_roi_for_ocr(roi):
        # Convert to grayscale
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi

        # Denoise
        roi_denoised = cv2.fastNlMeansDenoising(roi_gray, None, h=30, templateWindowSize=7, searchWindowSize=21)

        # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        roi_sharp = cv2.filter2D(roi_denoised, -1, kernel)

        # Preprocess ROI to connect characters
        kernel = np.ones((2, 2), np.uint8)
        roi_closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

        # Contrast enhancement
        roi_eq = cv2.equalizeHist(roi_sharp)

        # Threshold (try both adaptive and binary)
        roi_thresh = roi_eq 
        # Uncomment below to try adaptive thresholding and comment above line
        # roi_thresh = cv2.adaptiveThreshold(
        #     roi_eq, 255,
        #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY,
        #     11, 2
        # )
        # Optionally try: _, roi_thresh = cv2.threshold(roi_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Upscale more aggressively
        roi_up = cv2.resize(roi_thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Pad horizontally
        pad = 20
        roi_up = cv2.copyMakeBorder(roi_up, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=255)

        # --- Output the preprocessed ROI image ---
        output_dir = "roi_debug_output"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"roi_{int(time.time() * 1000)}.png"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, roi_up)
        print(f"Saved preprocessed ROI to {output_path}")

        return roi_up

    def extract_text_along_decimated_lines(self):
        if self.image is None or self.decimated_contour is None:
            return None

        img = self.image.copy()
        contour = self.decimated_contour
        line_width = 100
        results = []

        # Use the largest contour for distance calculation
        if hasattr(self, 'outer_contour') and self.outer_contour is not None:
            main_contour = max(self.outer_contour, key=cv2.contourArea)
        else:
            main_contour = contour

        max_distance_to_contour = 100  # pixels

        for i in range(len(contour)):
            pt1 = contour[i][0]
            pt2 = contour[(i + 1) % len(contour)][0]

            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            length = int(math.hypot(dx, dy))
            if length < 10:
                continue

            # Midpoint of the segment
            mx = int((pt1[0] + pt2[0]) / 2)
            my = int((pt1[1] + pt2[1]) / 2)

            # Distance from midpoint to contour
            dist = cv2.pointPolygonTest(main_contour, (mx, my), True)
            if abs(dist) > max_distance_to_contour:
                continue  # Skip if farther than 100 pixels

            angle = math.degrees(math.atan2(dy, dx))
            cx = (pt1[0] + pt2[0]) / 2.0
            cy = (pt1[1] + pt2[1]) / 2.0

            try:
                M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

                pts = np.array([[pt1], [pt2]], dtype=np.float32)
                pts_rot = cv2.transform(pts, M)
                x1r, y1r = pts_rot[0][0]
                x2r, y2r = pts_rot[1][0]

                x_min = int(min(x1r, x2r))
                x_max = int(max(x1r, x2r))
                y_center = int((y1r + y2r) / 2)
                y_min = max(0, y_center - line_width // 2)
                y_max = min(rotated.shape[0], y_center + line_width // 2)

                if x_min >= x_max or y_min >= y_max:
                    continue
                roi = rotated[y_min:y_max, x_min:x_max]
                if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
                    continue

                # --- Preprocess ROI before OCR ---
                roi_proc = self.preprocess_roi_for_ocr(roi)
                #comment out above line and use below line to skip preprocessing
                #roi_proc = roi
                ocr_text = pytesseract.image_to_string(roi_proc, config="--psm 6").strip()

                roi_inverted = cv2.rotate(roi, cv2.ROTATE_180)
                #roi_inverted_proc = self.preprocess_roi_for_ocr(roi_inverted)
                #comment out above line and use below line to skip preprocessing for inverted
                roi_inverted_proc = roi_inverted
                ocr_text_inverted = pytesseract.image_to_string(roi_inverted_proc, config="--psm 6").strip()
                # ----------------------------------

                all_texts = [ocr_text, ocr_text_inverted]
                number_matches = []
                for text in all_texts:
                    number_matches += re.findall(r"\d{1,4}(?:\.\d{1,3})?\s*['\"]?", text)
                distances = []
                for match in number_matches:
                    num_str = re.sub(r"[^\d.]", "", match)
                    if '.' in num_str:
                        try:
                            distances.append(float(num_str))
                        except ValueError:
                            continue
                distance_sum = sum(distances) if distances else 0.0
                results.append({
                    "index": i,
                    "distance_sum_feet": distance_sum,
                    "pt1": pt1,
                    "pt2": pt2,
                    "pixel_length": math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                })

            except Exception as e:
                print(f"Exception on line {i}: {e}")
                continue

        # Display results in the Qt table_text widget
        self.table_text.clear()
        if results:
            for result in results:
                self.table_text.append(f"Contour {result['index']}: Distance sum = {result['distance_sum_feet']:.2f}")
        else:
            self.table_text.append("No distances found along decimated contour lines.")

        # Append scale factor to the table
        if self.SCALE_FACTOR is not None:
            self.table_text.append(f"\nScale Factor: {self.SCALE_FACTOR:.4f} meters/pixel")
        else:
            self.table_text.append("\nScale Factor: Not set")

        return results if results else None

    def Distance_find(self, ocr_results):
        """
        Sum all distances and update GUI.
        Returns: total_distance, array of all distances
        """
        all_distances = []
        for r in ocr_results:
            all_distances.extend(r["distances"])
        total_distance = sum(all_distances)
        self.distance_label.setText(f"Measured Distance: {total_distance:.2f}")
        self.distance_candidates = all_distances
        return total_distance, all_distances
    
    def create_roi_overlay(self, roi_boxes, image_shape):
        """
        Draws ROI boxes as overlays for display.
        roi_boxes: list of np.int32 corner arrays
        image_shape: shape of the image to overlay on
        Returns: overlay image
        """
        overlay = np.zeros(image_shape, dtype=np.uint8)
        for box in roi_boxes:
            cv2.polylines(overlay, [box], isClosed=True, color=(0, 255, 255), thickness=2)
        return overlay

    def sort_clockwise(self, points):
        if not points:
            return []

        # Compute center of all points
        cx = np.mean([pt[0] for _, pt in points])
        cy = np.mean([pt[1] for _, pt in points])

        def angle(p):
            return np.arctan2(p[1] - cy, p[0] - cx)

        return sorted(points, key=lambda x: angle(x[1]))
  
    def enable_brown_line_mode(self):
        self.brown_line_mode = True
        self.brown_line_points = []
        self.statusBar().showMessage("Brown line mode: Click two points to add a brown line (snaps to nearest contour).")
        self.canvas.left_click.disconnect()
        self.canvas.left_click.connect(self.brown_line_click)
        self.canvas.mouse_move.connect(self.brown_line_mouse_move)

    def brown_line_click(self, x, y):
        img_x, img_y = self.canvas_to_image_coords(x, y)
        snap_x, snap_y = self.snap_to_nearest_contour((img_x, img_y))
        self.brown_line_points.append((snap_x, snap_y))
        if len(self.brown_line_points) == 1:
            # Draw blue snap indicator
            self.draw_snap_indicator((snap_x, snap_y))
        elif len(self.brown_line_points) == 2:
            # Add the brown line
            self.brown_lines.append(tuple(self.brown_line_points))
            self.brown_line_mode = False
            self.statusBar().showMessage("Brown line added.")
            self.brown_line_points = []
            self.update_canvas_image()
            # Restore normal left click
            self.set_default_left_click()
            self.canvas.mouse_move.disconnect(self.brown_line_mouse_move)

    def brown_line_mouse_move(self, x, y):
        if not self.brown_line_mode or len(self.brown_line_points) >= 2:
            return
        img_x, img_y = self.canvas_to_image_coords(x, y)
        snap_x, snap_y = self.snap_to_nearest_contour((img_x, img_y))
        self.draw_snap_indicator((snap_x, snap_y))

    def snap_to_nearest_contour(self, pt):
        # Snap to nearest point on any contour (decimated_contour)
        if self.decimated_contour is None:
            return pt
        contour = self.decimated_contour.reshape(-1, 2)
        dists = np.linalg.norm(contour - np.array(pt), axis=1)
        idx = np.argmin(dists)
        return tuple(contour[idx])

    def draw_snap_indicator(self, pt):
        # Draw a blue circle at pt on the canvas, with larger radius and thinner outline
        self.update_canvas_image()  # Redraw base image and lines
        pixmap = self.canvas.pixmap().copy()
        from PyQt5.QtGui import QPainter, QPen
        from PyQt5.QtCore import QPoint
        qp = QPainter(pixmap)
        pen = QPen(QColor("blue"))
        pen.setWidth(4)  # Thinner outline
        qp.setPen(pen)
        canvas_x, canvas_y = self.image_to_canvas_coords(pt[0], pt[1])
        qp.drawEllipse(QPoint(canvas_x, canvas_y), 12, 12)  # Larger radius
        qp.end()
        self.canvas.setPixmap(pixmap)
            # Optionally highlight the nearest segment
        if self.decimated_contour is not None:
            contour = self.decimated_contour.reshape(-1, 2)
            dists = np.linalg.norm(contour - np.array(pt), axis=1)
            idx = np.argmin(dists)
            pt1 = contour[idx]
            pt2 = contour[(idx + 1) % len(contour)]
            pen_line = QPen(QColor("blue"))
            pen_line.setWidth(2)
            qp.setPen(pen_line)
            x1, y1 = self.image_to_canvas_coords(pt1[0], pt1[1])
            x2, y2 = self.image_to_canvas_coords(pt2[0], pt2[1])
            qp.drawLine(x1, y1, x2, y2)

    def image_to_canvas_coords(self, img_x, img_y):
        # Inverse of canvas_to_image_coords
        canvas_w, canvas_h = self.canvas.width(), self.canvas.height()
        img_h, img_w = self.original_image.shape[:2]
        scale = self.zoom_level
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img_x0 = (canvas_w - new_w) // 2 + self.pan_x
        img_y0 = (canvas_h - new_h) // 2 + self.pan_y
        canvas_x = int(img_x * scale + img_x0)
        canvas_y = int(img_y * scale + img_y0)
        return canvas_x, canvas_y

    def robust_scale_factor(self, pixel_lengths, real_distances_meters):
        """
        Robustly calculate scale factor (meters per pixel) using least squares and outlier rejection.
        Returns: scale_factor, pixel_scale, inlier_mask
        """
        import numpy as np
        pixel_lengths = np.array(pixel_lengths)
        real_distances_meters = np.array(real_distances_meters)
        if len(pixel_lengths) < 2 or len(real_distances_meters) < 2:
            return None, None, None

        # Linear fit: real_distance = scale_factor * pixel_length
        A = np.vstack([pixel_lengths, np.ones(len(pixel_lengths))]).T
        result = np.linalg.lstsq(A, real_distances_meters, rcond=None)
        scale_factor, intercept = result[0]

        # Calculate residuals
        predicted = scale_factor * pixel_lengths + intercept
        residuals = real_distances_meters - predicted
        std_res = np.std(residuals)

        # Identify inliers (within 2 standard deviations)
        inlier_mask = np.abs(residuals) < 2 * std_res

        # Refit using only inliers
        if np.sum(inlier_mask) >= 2:
            A_in = np.vstack([pixel_lengths[inlier_mask], np.ones(np.sum(inlier_mask))]).T
            result_in = np.linalg.lstsq(A_in, real_distances_meters[inlier_mask], rcond=None)
            scale_factor, intercept = result_in[0]
            pixel_scale = 1.0 / scale_factor if scale_factor != 0 else None
        else:
            pixel_scale = None

        return scale_factor, pixel_scale, inlier_mask

    def on_aggressiveness_slider_changed(self):
        # Restart the timer every time the slider value changes
        self.aggressiveness_timer.start(200)

    def on_contrast_slider_changed(self):
        self.contrast_timer.start(200)

    def on_pixel_slider_changed(self):
        self.kernel_timer.start(200)

    def on_simplify_slider_changed(self):
        self.simplify_timer.start(200)

    def find_best_contrast_for_ocr(self, roi=None, contrast_range=None):
        """
        Try different contrast settings and pick the one that gives the best OCR result.
        roi: region of interest (numpy array). If None, use the whole original image.
        contrast_range: list or range of contrast values to try (default: 60 to 180).
        Returns: best_contrast, best_text, best_score
        """
        if roi is None:
            roi = self.original_image
        if contrast_range is None:
            contrast_range = range(60, 181, 10)  # Try values from 60 to 180

        best_score = -1
        best_contrast = None
        best_text = ""
        for contrast in contrast_range:
            alpha = contrast / 100.0
            enhanced = cv2.convertScaleAbs(roi, alpha=alpha, beta=0)
            # Preprocess for OCR if needed
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) == 3 else enhanced
            text = pytesseract.image_to_string(gray, config="--psm 6")
            # Score: count number of digits found (or use another metric)
            score = len(re.findall(r"\d", text))
            if score > best_score:
                best_score = score
                best_contrast = contrast
                best_text = text
        return best_contrast, best_text, best_score

    def auto_contrast_for_ocr(self):
        best_contrast, best_text, best_score = self.find_best_contrast_for_ocr()
        if best_contrast is not None:
            self.contrast_slider['slider'].setValue(best_contrast)
            self.statusBar().showMessage(f"Best contrast for OCR: {best_contrast} (score: {best_score})")
            print("Best OCR text sample:", best_text)
        else:
            self.statusBar().showMessage("Auto contrast failed to find a better setting.")

    def enable_auto_contrast_roi_mode(self):
        self.statusBar().showMessage("Click near a contour segment to select ROI for auto contrast.")
        try:
            self.canvas.left_click.disconnect(self.add_seed_point)
        except Exception:
            pass
        self.canvas.left_click.connect(self.auto_contrast_roi_pick)

    def paintEvent(self, event):
        super().paintEvent(event)
        if getattr(self, 'roi_selecting', False) and hasattr(self, 'roi_start') and hasattr(self, 'roi_end'):
            painter = QPainter(self)
            pen = QPen(QColor(255, 255, 0, 128))  # Semi-transparent yellow
            pen.setWidth(2)
            painter.setPen(pen)
            x1, y1 = self.roi_start
            x2, y2 = self.roi_end
            painter.drawRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            painter.end()

    def auto_contrast_roi_pick(self, x, y):
        img_x, img_y = self.canvas_to_image_coords(x, y)
        if self.decimated_contour is None:
            self.statusBar().showMessage("No contour available.")
            return

        contour = self.decimated_contour.reshape(-1, 2)
        # Find nearest segment
        min_dist = float('inf')
        nearest_idx = 0
        for i in range(len(contour)):
            pt1 = contour[i]
            pt2 = contour[(i + 1) % len(contour)]
            v = np.array(pt2) - np.array(pt1)
            w = np.array([img_x, img_y]) - np.array(pt1)
            if np.dot(v, v) == 0:
                proj = pt1
            else:
                t = np.clip(np.dot(w, v) / np.dot(v, v), 0, 1)
                proj = pt1 + t * v
            dist = np.linalg.norm(np.array([img_x, img_y]) - proj)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        pt1 = contour[nearest_idx]
        pt2 = contour[(nearest_idx + 1) % len(contour)]
        seg_vec = np.array(pt2) - np.array(pt1)
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1:
            self.statusBar().showMessage("Contour segment too short.")
            return
        seg_dir = seg_vec / seg_len
        perp_dir = np.array([-seg_dir[1], seg_dir[0]])
        center = (np.array(pt1) + np.array(pt2)) / 2
        half_width = 50
        p1 = center - seg_vec / 2 + perp_dir * half_width
        p2 = center + seg_vec / 2 + perp_dir * half_width
        p3 = center + seg_vec / 2 - perp_dir * half_width
        p4 = center - seg_vec / 2 - perp_dir * half_width
        roi_corners = np.array([p1, p2, p3, p4], dtype=np.float32)
        dst_rect = np.array([
            [0, 0],
            [int(seg_len), 0],
            [int(seg_len), 100],
            [0, 100]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(roi_corners, dst_rect)
        roi = cv2.warpPerspective(self.original_image, M, (int(seg_len), 100))

        best_contrast, best_text, best_score = self.find_best_contrast_for_ocr(roi)
        if best_contrast is not None:
            self.contrast_slider['slider'].setValue(best_contrast)
            self.statusBar().showMessage(f"Best contrast for OCR (ROI): {best_contrast} (score: {best_score})")
            print("Best OCR text sample (ROI):", best_text)
        else:
            self.statusBar().showMessage("Auto contrast failed to find a better setting for ROI.")

        # Draw ROI for feedback
        overlay = self.image.copy() if self.image is not None else self.original_image.copy()
        roi_poly = roi_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [roi_poly], isClosed=True, color=(0, 255, 255), thickness=2)
        self.image = overlay
        self.update_canvas_image()

        # Restore normal left click
        self.set_default_left_click()
    
    def on_extract_distances_clicked(self):
        self.last_contour_distances = self.extract_text_along_decimated_lines()
    
    def set_default_left_click(self):
        try:
            self.canvas.left_click.disconnect(self.auto_contrast_roi_pick)
        except Exception:
            pass
        try:
            self.canvas.left_click.disconnect(self.brown_line_click)
        except Exception:
            pass
        try:
            self.canvas.left_click.disconnect(self.add_seed_point)
        except Exception:
            pass
        self.canvas.left_click.connect(self.add_seed_point)

    def on_brightness_slider_changed(self):
        self.update_canvas_image()
    
    def reset_all_sliders(self):
        # Reset all sliders to their default values
        self.aggressiveness_slider['slider'].setValue(25)
        self.pixel_slider['slider'].setValue(2)
        self.contrast_slider['slider'].setValue(100)
        # self.kernel_slider['slider'].setValue(5)
        # self.lambda_slider['slider'].setValue(0)
        self.simplify_slider['slider'].setValue(5)
        self.brightness_slider['slider'].setValue(0)

        # Also update the displayed values
        self.aggressiveness_slider['value_label'].setText("25")
        self.pixel_slider['value_label'].setText("2")
        self.contrast_slider['value_label'].setText("100")
        # self.kernel_slider['value_label'].setText("5")
        # self.lambda_slider['value_label'].setText("0")
        self.simplify_slider['value_label'].setText("5")
        self.brightness_slider['value_label'].setText("0")

                # Reset other relevant states if necessary
        self.mask = None
        self.seed_points = []
        self.image = self.original_image.copy() if self.original_image is not None else None

        self.update_canvas_image()

    @staticmethod
    def apply_contrast_brightness_preserve_white(img, alpha, beta):
        img = img.astype(np.float32)
        # Save mask of white pixels
        white_mask = np.all(img == 255, axis=2) if img.ndim == 3 else (img == 255)
        # Apply contrast and brightness
        img = img * alpha + beta
        # Restore white pixels
        if img.ndim == 3:
            img[white_mask] = 255
        else:
            img[white_mask] = 255
        # Clip to [0, 255] and convert back to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def create_simplified_contour_2(self):
        if self.mask is None or np.count_nonzero(self.mask) == 0:
            return

        mask_orig = (self.mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        outer_contour = max(contours, key=cv2.contourArea)

        expand_amt = max(1, int(self.pixel_slider['slider'].value() * 4))
        kernel = np.ones((expand_amt, expand_amt), np.uint8)
        mask_expanded = cv2.dilate(mask_orig, kernel, iterations=1)
        contours_exp, _ = cv2.findContours(mask_expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_exp:
            return
        outer_contour_exp = max(contours_exp, key=cv2.contourArea)

        mask_between = np.zeros_like(mask_orig)
        cv2.drawContours(mask_between, [outer_contour_exp], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(mask_between, [outer_contour], -1, 0, thickness=cv2.FILLED)

        yellow_mask = np.zeros_like(self.original_image)
        yellow_mask[mask_between > 0] = (0, 255, 255)  # Yellow

        # 1. Get the underlaid image (use contrast/brightness adjusted or self.original_image)
        underlaid = self.apply_contrast_brightness_preserve_white(self.original_image, 1.0, 0)  # or use your adjusted image

        # 2. Convert to grayscale
        gray = cv2.cvtColor(underlaid, cv2.COLOR_BGR2GRAY)

        # 3. Apply threshold or edge detection
        # Option 1: Simple threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Option 2: Canny edge detection (for dashed/complex lines, sometimes better)
        # binary = cv2.Canny(gray, 50, 150)

        # 4. Find contours on the binary image
        masked_binary = cv2.bitwise_and(binary, binary, mask=mask_between)
        all_contours, _ = cv2.findContours(masked_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        long_line_contours = []
        min_length = 20  # Adjust as needed for "long" lines

        for cnt in all_contours:
            arc_len = cv2.arcLength(cnt, False)
            if arc_len > min_length:
                epsilon = 0.01 * arc_len
                approx = cv2.approxPolyDP(cnt, epsilon, False)
                long_line_contours.append(approx)
        print(f"Found {len(long_line_contours)} long line contours")
        self.simplified2_long_line_contours = long_line_contours if long_line_contours else None
        
        # Store other overlays as before
        self.simplified2_yellow_mask = yellow_mask
        self.simplified2_inner_contour = outer_contour
        self.simplified2_outer_contour = outer_contour_exp

        self.update_canvas_image()
        # # Find all contours in the yellow mask
        # all_contours, _ = cv2.findContours(mask_between, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # epsilon_outer = 0.01 * cv2.arcLength(outer_contour, True)
        # approx_outer = cv2.approxPolyDP(outer_contour, epsilon_outer, True)

        # # parallel_contours = []
        # # angle_threshold = 10  # degrees

        # # for cnt in all_contours:
        # #     if cv2.contourArea(cnt) < 100:  # skip small contours
        # #         continue
        # #     # Approximate contour
        # #     epsilon = 0.01 * cv2.arcLength(cnt, True)
        # #     approx = cv2.approxPolyDP(cnt, epsilon, True)
        # #     # Compare angles with outer contour
        # #     if len(approx) < 2 or len(approx_outer) < 2:
        # #         continue
        # #     angles_inner = []
        # #     angles_outer = []
        # #     for i in range(len(approx)):
        # #         p1 = approx[i][0]
        # #         p2 = approx[(i+1)%len(approx)][0]
        # #         v = p2 - p1
        # #         angle = np.arctan2(v[1], v[0])
        # #         angles_inner.append(angle)
        # #     for i in range(len(approx_outer)):
        # #         p1 = approx_outer[i][0]
        # #         p2 = approx_outer[(i+1)%len(approx_outer)][0]
        # #         v = p2 - p1
        # #         angle = np.arctan2(v[1], v[0])
        # #         angles_outer.append(angle)
        # #     # Compare mean angle difference
        # #     mean_angle_inner = np.mean(angles_inner)
        # #     mean_angle_outer = np.mean(angles_outer)
        # #     angle_diff = np.abs(np.rad2deg(mean_angle_inner - mean_angle_outer))
        # #     if angle_diff > 180:
        # #         angle_diff = 360 - angle_diff
        # #     if angle_diff < angle_threshold:
        # #         parallel_contours.append(approx)

        # # # Fit lines to each parallel contour and extend to intersection
        # # # For simplicity, use the largest parallel contour
        # # if parallel_contours:
        # #     largest = max(parallel_contours, key=cv2.contourArea)
        # #     # Fit lines to each segment
        # #     lines = []
        # #     for i in range(len(largest)):
        # #         p1 = largest[i][0]
        # #         p2 = largest[(i+1)%len(largest)][0]
        # #         lines.append((p1, p2))
        # #     # Find intersections
        # #     def line_intersection(line1, line2):
        # #         xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        # #         ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
        # #         def det(a, b):
        # #             return a[0] * b[1] - a[1] * b[0]
        # #         div = det(xdiff, ydiff)
        # #         if div == 0:
        # #             return None
        # #         d = (det(*line1), det(*line2))
        # #         x = det(d, xdiff) / div
        # #         y = det(d, ydiff) / div
        # #         return int(x), int(y)
        # #     intersections = []
        # #     for i in range(len(lines)):
        # #         l1 = lines[i]
        # #         l2 = lines[(i+1)%len(lines)]
        # #         pt = line_intersection(l1, l2)
        # #         if pt is not None:
        # #             intersections.append([[pt]])
        # #     approx_poly = np.array(intersections, dtype=np.int32)
        # #     self.simplified2_approx_closed_line = approx_poly
        # #     # Print the list of vertices
        # #     print("simplified2_approx_closed_line vertices:")
        # #     for pt in approx_poly.reshape(-1, 2):
        # #         print(tuple(pt))
        # #     self.simplified2_parallel_contours = parallel_contours
        # # else:
        # #     self.simplified2_approx_closed_line = None
        # #     self.simplified2_parallel_contours = None
        # #     print("simplified2_approx_closed_line vertices: none")

        # self.simplified2_yellow_mask = yellow_mask
        # self.simplified2_inner_contour = outer_contour
        # self.simplified2_outer_contour = outer_contour_exp
        # # self.image = blended
        # self.update_canvas_image()
        
        # --- Find, simplify, and store long line contours ---


    def create_simplified_contour_3(self):
        if self.mask is None or np.count_nonzero(self.mask) == 0:
            return

        mask_orig = (self.mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        outer_contour = max(contours, key=cv2.contourArea)
        epsilon_outer = 0.01 * cv2.arcLength(outer_contour, True)
        approx_outer = cv2.approxPolyDP(outer_contour, epsilon_outer, True)

        expand_amt = max(1, int(self.pixel_slider['slider'].value() * 4))
        kernel = np.ones((expand_amt, expand_amt), np.uint8)
        mask_expanded = cv2.dilate(mask_orig, kernel, iterations=1)
        contours_exp, _ = cv2.findContours(mask_expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_exp:
            return
        outer_contour_exp = max(contours_exp, key=cv2.contourArea)

        mask_between = np.zeros_like(mask_orig)
        cv2.drawContours(mask_between, [outer_contour_exp], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(mask_between, [outer_contour], -1, 0, thickness=cv2.FILLED)

        yellow_mask = np.zeros_like(self.original_image)
        yellow_mask[mask_between > 0] = (0, 255, 255)  # Yellow

        # 1. Get the underlaid image (use contrast/brightness adjusted or self.original_image)
        underlaid = self.apply_contrast_brightness_preserve_white(self.original_image, 1.0, 0)  # or use your adjusted image

        # 2. Convert to grayscale
        gray = cv2.cvtColor(underlaid, cv2.COLOR_BGR2GRAY)

        # 3. Apply threshold or edge detection
        # Option 1: Simple threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Option 2: Canny edge detection (for dashed/complex lines, sometimes better)
        # binary = cv2.Canny(gray, 50, 150)

        # 4. Find contours on the binary image
        masked_binary = cv2.bitwise_and(binary, binary, mask=mask_between)
        masked_binary[mask_between == 0] = 255  # Make mask area white
        all_contours, _ = cv2.findContours(masked_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # --- DEBUGGING: Show all contours found in the masked binary image
        debug_img = underlaid.copy()
        cv2.drawContours(debug_img, all_contours, -1, (255, 0, 0), 1)  # Draw in red
        cv2.imshow("All Contours", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # ----------------------------------------

        # --- DEBUGGING: Show the masked binary image
        cv2.imshow("Masked Binary", masked_binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # ----------------------------------------

        # long_line_contours = []
        # min_length = 2  # Adjust as needed for "long" lines

        # --- Skeletonize the masked area to get the centerline ---
        from skimage.morphology import skeletonize
        
        # Invert masked_binary so the line is white (foreground) and background is black
        masked_binary_inverted = cv2.bitwise_not(masked_binary)

        # Convert to boolean for skeletonize (expects 0/1)
        masked_bool = (masked_binary_inverted > 0)
        skeleton = skeletonize(masked_bool)
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)

        # Find contours on the skeleton (centerline)
        skeleton_contours, _ = cv2.findContours(skeleton_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        raw_centerlines = [cnt for cnt in skeleton_contours if len(cnt) >= 10]
        # def contour_mostly_inside_mask(contour, mask, threshold=0.75):
        #     pts = contour.reshape(-1, 2)
        #     inside = mask[pts[:,1], pts[:,0]] > 0
        #     return np.mean(inside) >= threshold

        # raw_centerlines = [
        #     cnt for cnt in skeleton_contours
        #     if len(cnt) >= 10 and contour_mostly_inside_mask(cnt, mask_between)
        # raw_centerlines = [cnt for cnt in skeleton_contours if len(cnt) >= 10]

        # Save for debugging if needed
        cv2.imwrite(r"C:\Users\nicho\source\repos\miT\SeedPointFillToMaskImageOut-miT\SeedPointFillToMaskImageOut\maskedBinary1.png", masked_binary)
        cv2.imwrite(r"C:\Users\nicho\source\repos\miT\SeedPointFillToMaskImageOut-miT\SeedPointFillToMaskImageOut\skeleton.png", skeleton_uint8)

        # # Convert underlaid to BGR if needed
        # if len(underlaid.shape) == 2:
        #     underlaid_bgr = cv2.cvtColor(underlaid, cv2.COLOR_GRAY2BGR)
        # else:
        #     underlaid_bgr = underlaid.copy()

        # # Resize skeleton to match underlaid image size
        # skeleton_resized = cv2.resize(
        #     skeleton_uint8,
        #     (underlaid_bgr.shape[1], underlaid_bgr.shape[0]),
        #     interpolation=cv2.INTER_NEAREST
        # )

        # Create a red overlay image for the skeleton for output display... does not work...
        if len(underlaid.shape) == 2:
            underlaid_bgr = cv2.cvtColor(underlaid, cv2.COLOR_GRAY2BGR)
        else:
            underlaid_bgr = underlaid.copy()

        skeleton_resized = cv2.resize(
            skeleton_uint8,
            (underlaid_bgr.shape[1], underlaid_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        red_overlay = np.zeros_like(underlaid_bgr)
        red_overlay[skeleton_resized > 0] = (0, 0, 255)  # BGR: red
        alpha = 1.0  # Full opacity for the skeleton
        overlay_img = cv2.addWeighted(underlaid_bgr, 1.0, red_overlay, alpha, 0)

        self.skeleton_contours = raw_centerlines if raw_centerlines else None
        self.skeleton_overlay_img = overlay_img  # This is the red overlay image
        #self.simplified2_approx_closed_line = raw_centerlines if raw_centerlines else None  # This is a list of contours
        # Store the raw centerlines for drawing as contours
        # self.simplified2_approx_closed_line = raw_centerlines if raw_centerlines else None
        # --- Find and fit long lines to skeletonized contours ---
        min_length = self.pixel_slider['slider'].value() * 4
        long_lines = [cnt for cnt in raw_centerlines if len(cnt) >= min_length]

        min_length = self.pixel_slider['slider'].value() * 4
        self.compute_skeleton_fit_lines_and_intersections(
            raw_centerlines,
            mask_between.shape,
            min_length
        )

        self.simplified2_yellow_mask = yellow_mask
        self.simplified2_inner_contour = outer_contour
        self.simplified2_outer_contour = outer_contour_exp
        self.update_canvas_image()
        
    def compute_skeleton_fit_lines_and_intersections(self, raw_centerlines, mask_shape, min_length):
        """
        Given skeleton contours, compute best-fit lines (extended to mask/image bounds)
        and their intersection points. Then trim each line to run only between its intersection points.
        Stores results in self.skeleton_fit_lines and self.skeleton_fit_intersections.
        """
        fit_lines = []
        h, w = mask_shape

        # 1. Fit lines to contours
        for cnt in raw_centerlines:
            if len(cnt) < min_length:
                continue
            cnt_pts = cnt.reshape(-1, 2).astype(np.float32)
            if cnt_pts.shape[0] < 2:
                continue
            [vx, vy, x0, y0] = cv2.fitLine(cnt_pts, cv2.DIST_L2, 0, 0.01, 0.01)
            # Calculate two points far along the line direction (for intersection math)
            left_y = int((-x0 * vy / vx) + y0) if vx != 0 else 0
            right_y = int(((w - x0) * vy / vx) + y0) if vx != 0 else h-1
            top_x = int((-y0 * vx / vy) + x0) if vy != 0 else 0
            bottom_x = int(((h - y0) * vx / vy) + x0) if vy != 0 else w-1

            points = []
            if 0 <= left_y < h:
                points.append((0, left_y))
            if 0 <= right_y < h:
                points.append((w-1, right_y))
            if 0 <= top_x < w:
                points.append((top_x, 0))
            if 0 <= bottom_x < w:
                points.append((bottom_x, h-1))
            if len(points) >= 2:
                fit_lines.append((points[0], points[1]))

        # 2. Find all intersections
        def line_intersection(line1, line2):
            (x1, y1), (x2, y2) = line1
            (x3, y3), (x4, y4) = line2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return None  # Parallel lines
            px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
            return (px, py)

        intersections = []
        line_to_intersections = [[] for _ in fit_lines]
        for i in range(len(fit_lines)):
            for j in range(i+1, len(fit_lines)):
                pt = line_intersection(fit_lines[i], fit_lines[j])
                if pt is not None:
                    intersections.append((int(round(pt[0])), int(round(pt[1]))))
                    line_to_intersections[i].append(pt)
                    line_to_intersections[j].append(pt)

        # 3. For each line, trim to its intersection points
        trimmed_lines = []
        for idx, (pt1, pt2) in enumerate(fit_lines):
            pts = line_to_intersections[idx]
            if len(pts) < 2:
                # Not enough intersections, keep as is
                trimmed_lines.append((pt1, pt2))
                continue
            # Project intersection points onto the line, sort by distance along the line
            [vx, vy] = np.array(pt2) - np.array(pt1)
            norm = np.hypot(vx, vy)
            if norm == 0:
                trimmed_lines.append((pt1, pt2))
                continue
            vx, vy = vx / norm, vy / norm
            def proj_t(pt):
                return (pt[0] - pt1[0]) * vx + (pt[1] - pt1[1]) * vy
            pts_sorted = sorted(pts, key=proj_t)
            trimmed_lines.append((tuple(map(int, map(round, pts_sorted[0]))),
                                  tuple(map(int, map(round, pts_sorted[-1])))))

        self.skeleton_fit_lines = trimmed_lines
        self.skeleton_fit_intersections = [tuple(map(int, map(round, pt))) for pt in intersections]  










if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FloodFillApp()
    win.show()
    sys.exit(app.exec_())