#OCR.Testing_LATEST.py
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QGraphicsView, QGraphicsScene, QFileDialog, QSizePolicy,
    QSlider, QLabel, QHBoxLayout, QRadioButton, QButtonGroup, QProgressBar
)
from PyQt5.QtGui import QPixmap, QPalette, QColor, QPen, QImage
from PyQt5.QtCore import Qt, QRectF, QPointF
import pytesseract
import cv2
import numpy as np
import re
import itertools

class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self._pixmap_item = None
        self._zoom = 0
        self._user_zoomed = False
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self._panning = False
        self._pan_start = None
        self.ocr_boxes = []
        self.selected_box_idx = None
        self._rect_items = []
        self.bearings = []
        self.distances = []

    def set_image(self, pixmap, transform=None):
        self.scene().clear()
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self.setSceneRect(QRectF(self._pixmap_item.pixmap().rect()))
        if transform is not None:
            self.setTransform(transform)
        else:
            self.reset_zoom()
        self.ocr_boxes = []
        self.selected_box_idx = None
        self._rect_items = []

    def reset_zoom(self):
        self._zoom = 0
        self._user_zoomed = False
        self.setTransform(self.transform().fromScale(1, 1).inverted()[0])
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if self._pixmap_item:
            zoom_in_factor = 1.25
            zoom_out_factor = 0.8
            if event.angleDelta().y() > 0:
                factor = zoom_in_factor
                self._zoom += 1
            else:
                factor = zoom_out_factor
                self._zoom -= 1
            if self._zoom < -10:
                self._zoom = -10
            elif self._zoom > 20:
                self._zoom = 20
            else:
                self._user_zoomed = True
                self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap_item and not self._user_zoomed:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        if self.ocr_boxes and self._pixmap_item:
            pos = self.mapToScene(event.pos())
            for idx, (x, y, w, h, text) in enumerate(self.ocr_boxes):
                if QRectF(x, y, w, h).contains(pos):
                    self.selected_box_idx = idx
                    self.highlight_text(self.ocr_boxes)
                    selected = text.replace(' ', '')
                    # Replace double direction letters at the start
                    selected = re.sub(r'^(EE)', 'E', selected)
                    selected = re.sub(r'^(WW)', 'W', selected)
                    selected = re.sub(r'^(NN)', 'N', selected)
                    selected = re.sub(r'^(SS)', 'S', selected)
                    # If starts with N/W/S/E and contains a '.', remove all '.'
                    if selected and selected[0] in 'NWSE' and '.' in selected:
                        selected = selected.replace('.', '')
                    if selected and not (selected[0] in 'NWSE' or selected[0].isdigit()):
                        m = re.search(r'[0-9]', selected)
                        if m:
                            selected = selected[m.start():]

                    # Only process if starts with N/W/S/E or is a decimal number
                    is_bearing = selected and selected[0] in 'NWSE'
                    is_decimal = bool(re.match(r'^\d*\.\d+$', selected))
                    if not is_bearing and not is_decimal:
                        return

                    if is_bearing:
                        if len(selected) > 8:
                            return
                        self.bearings.append(selected)
                    if is_decimal:
                        self.distances.append(selected)

                    if len(self.bearings) < 1:
                        pass
                    else:
                        print("Bearings:", self.bearings)
                    if len(self.distances) < 1:
                        pass
                    else:
                        print("Distances:", self.distances)
                    return
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)

    def highlight_text(self, boxes):
        for item in getattr(self, "_rect_items", []):
            if item.scene() is self.scene():
                self.scene().removeItem(item)
        self._rect_items = []
        self.ocr_boxes = boxes
        for idx, (x, y, w, h, text) in enumerate(boxes):
            pen = QPen(QColor(0, 255, 0), 2)
            if idx == self.selected_box_idx:
                pen = QPen(QColor(255, 0, 0), 3)
            rect_item = self.scene().addRect(x, y, w, h, pen)
            self._rect_items.append(rect_item)

    def clear_bearings_and_distances(self):
        self.bearings.clear()
        self.distances.clear()
        print("Bearings cleared:", self.bearings)
        print("Distances cleared:", self.distances)

    
def merge_boxes(boxes, x_threshold=20, y_threshold=10):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged = []
    used = [False] * len(boxes)
    for i, (x, y, w, h, text) in enumerate(boxes):
        if used[i]:
            continue
        cluster = [(x, y, w, h, text)]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            xj, yj, wj, hj, textj = boxes[j]
            if abs(y - yj) < y_threshold or (y <= yj <= y + h) or (yj <= y <= yj + hj):
                if 0 < xj - (x + w) < x_threshold or 0 < x - (xj + wj) < x_threshold:
                    cluster.append((xj, yj, wj, hj, textj))
                    used[j] = True
        cluster = sorted(cluster, key=lambda b: b[0])
        xs = [b[0] for b in cluster]
        ys = [b[1] for b in cluster]
        ws = [b[0] + b[2] for b in cluster]
        hs = [b[1] + b[3] for b in cluster]
        texts = [b[4] for b in cluster]
        min_x = min(xs)
        min_y = min(ys)
        max_x = max(ws)
        max_y = max(hs)
        merged.append((min_x, min_y, max_x - min_x, max_y - min_y, ' '.join(texts)))
    return merged

class MainWindow(QMainWindow):
    MERGE_X_THRESH = 30  # Increase for more aggressive merging
    MERGE_Y_THRESH = 20

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR Testing | Automatic Text Detection w/Adjustments")

        self.viewer = ImageViewer()

        load_btn = QPushButton("Load PNG")
        load_btn.clicked.connect(self.load_png)
        load_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        ocr_btn = QPushButton("Highlight Text")
        ocr_btn.clicked.connect(self.highlight_text)
        ocr_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        clear_btn = QPushButton("Clear Bearings/Distances")
        clear_btn.clicked.connect(self.viewer.clear_bearings_and_distances)
        clear_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        auto_btn = QPushButton("Auto Find Bearings/Distances")
        auto_btn.clicked.connect(self.auto_find_clean_bearings_and_distances)
        auto_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # --- Mode radio buttons ---
        self.adjust_radio = QRadioButton("Image Adjustment Only")
        self.text_radio = QRadioButton("Image Adjustment w/ Automatic Text Detection")
        self.invert_radio = QRadioButton("Invert 180°")
        self.text_radio.setChecked(True)  # <-- Start in text mode
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.adjust_radio)
        self.mode_group.addButton(self.text_radio)
        # Invert is not mutually exclusive, so not added to mode_group

        self.invert_radio.toggled.connect(self.on_mode_changed)

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(QLabel("Mode:"))
        radio_layout.addWidget(self.adjust_radio)
        radio_layout.addWidget(self.text_radio)
        radio_layout.addWidget(self.invert_radio)

        # --- Sliders for real-time adjustment ---
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(50)
        self.brightness_label = QLabel("Brightness")
        self.brightness_value = QLabel(str(self.brightness_slider.value()))

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(-100)
        self.contrast_slider.setMaximum(100)
        self.contrast_slider.setValue(50)
        self.contrast_label = QLabel("Contrast")
        self.contrast_value = QLabel(str(self.contrast_slider.value()))

        self.erosion_slider = QSlider(Qt.Horizontal)
        self.erosion_slider.setMinimum(0)
        self.erosion_slider.setMaximum(5)
        self.erosion_slider.setValue(0)
        self.erosion_label = QLabel("Erosion")
        self.erosion_value = QLabel(str(self.erosion_slider.value()))

        self.dilation_slider = QSlider(Qt.Horizontal)
        self.dilation_slider.setMinimum(0)
        self.dilation_slider.setMaximum(5)
        self.dilation_slider.setValue(0)
        self.dilation_label = QLabel("Dilation")
        self.dilation_value = QLabel(str(self.dilation_slider.value()))

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(0)
        self.threshold_label = QLabel("Threshold")
        self.threshold_value = QLabel(str(self.threshold_slider.value()))

        self.gaussian_slider = QSlider(Qt.Horizontal)
        self.gaussian_slider.setMinimum(0)
        self.gaussian_slider.setMaximum(15)
        self.gaussian_slider.setSingleStep(2)
        self.gaussian_slider.setPageStep(2)
        self.gaussian_slider.setValue(0)
        self.gaussian_label = QLabel("Gaussian Blur")
        self.gaussian_value = QLabel(str(self.gaussian_slider.value()))

        self.merge_x_slider = QSlider(Qt.Horizontal)
        self.merge_x_slider.setMinimum(0)
        self.merge_x_slider.setMaximum(100)
        self.merge_x_slider.setValue(30)
        self.merge_x_label = QLabel("Merge X")
        self.merge_x_value = QLabel(str(self.merge_x_slider.value()))

        self.merge_y_slider = QSlider(Qt.Horizontal)
        self.merge_y_slider.setMinimum(0)
        self.merge_y_slider.setMaximum(100)
        self.merge_y_slider.setValue(20)
        self.merge_y_label = QLabel("Merge Y")
        self.merge_y_value = QLabel(str(self.merge_y_slider.value()))

        # Now build the list
        sliders_and_labels = [
            (self.brightness_label, self.brightness_slider, self.brightness_value),
            (self.contrast_label, self.contrast_slider, self.contrast_value),
            (self.erosion_label, self.erosion_slider, self.erosion_value),
            (self.dilation_label, self.dilation_slider, self.dilation_value),
            (self.threshold_label, self.threshold_slider, self.threshold_value),
            (self.gaussian_label, self.gaussian_slider, self.gaussian_value),
            (self.merge_x_label, self.merge_x_slider, self.merge_x_value),
            (self.merge_y_label, self.merge_y_slider, self.merge_y_value)
        ]

        # Split into two rows
        mid = len(sliders_and_labels) // 2
        row1 = sliders_and_labels[:mid]
        row2 = sliders_and_labels[mid:]

        slider_row1 = QHBoxLayout()
        slider_row2 = QHBoxLayout()

        for group, layout in [(row1, slider_row1), (row2, slider_row2)]:
            for label, slider, value in group:
                vlayout = QVBoxLayout()
                vlayout.addWidget(label, alignment=Qt.AlignHCenter)
                vlayout.addWidget(slider)
                vlayout.addSpacing(4)
                value.setAlignment(Qt.AlignHCenter)
                vlayout.addWidget(value, alignment=Qt.AlignHCenter)
                layout.addLayout(vlayout)

        slider_layout = QVBoxLayout()
        slider_layout.addLayout(slider_row1)
        slider_layout.addLayout(slider_row2)

        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(18)
        slider_layout.addWidget(self.progress_bar)

        # --- Controls layout (buttons + mode) ---
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(load_btn)
        controls_layout.addWidget(ocr_btn)
        controls_layout.addWidget(clear_btn)
        controls_layout.addWidget(auto_btn)
        controls_layout.addLayout(radio_layout)

        layout = QVBoxLayout()
        layout.addLayout(controls_layout)
        layout.addWidget(self.viewer)
        layout.addLayout(slider_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.resize(800, 400)

        self._last_pixmap = None
        self.connect_sliders(self.highlight_text)
        self.highlight_text()

    def connect_sliders(self, slot):
        try:
            self.brightness_slider.valueChanged.disconnect()
            self.contrast_slider.valueChanged.disconnect()
            self.erosion_slider.valueChanged.disconnect()
            self.dilation_slider.valueChanged.disconnect()
            self.threshold_slider.valueChanged.disconnect()
            self.gaussian_slider.valueChanged.disconnect()
            self.merge_x_slider.valueChanged.disconnect()
            self.merge_y_slider.valueChanged.disconnect()
        except Exception:
            pass
        self.brightness_slider.valueChanged.connect(slot)
        self.contrast_slider.valueChanged.connect(slot)
        self.erosion_slider.valueChanged.connect(slot)
        self.dilation_slider.valueChanged.connect(slot)
        self.threshold_slider.valueChanged.connect(slot)
        self.gaussian_slider.valueChanged.connect(slot)
        self.merge_x_slider.valueChanged.connect(slot)
        self.merge_y_slider.valueChanged.connect(slot)
        self.brightness_slider.valueChanged.connect(lambda v: self.brightness_value.setText(str(v)))
        self.contrast_slider.valueChanged.connect(lambda v: self.contrast_value.setText(str(v)))
        self.erosion_slider.valueChanged.connect(lambda v: self.erosion_value.setText(str(v)))
        self.dilation_slider.valueChanged.connect(lambda v: self.dilation_value.setText(str(v)))
        self.threshold_slider.valueChanged.connect(lambda v: self.threshold_value.setText(str(v)))
        self.gaussian_slider.valueChanged.connect(lambda v: self.gaussian_value.setText(str(v)))
        self.merge_x_slider.valueChanged.connect(lambda v: self.merge_x_value.setText(str(v)))
        self.merge_y_slider.valueChanged.connect(lambda v: self.merge_y_value.setText(str(v)))

    def on_mode_changed(self):
        if self.adjust_radio.isChecked():
            self.connect_sliders(self.update_preview)
            self.update_preview()
        else:
            self.connect_sliders(self.highlight_text)
            self.highlight_text()

    def load_png(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open PNG", "", "PNG Files (*.png)")
        if fname:
            pixmap = QPixmap(fname)
            if not pixmap.isNull():
                self._last_pixmap = pixmap
                if self.adjust_radio.isChecked():
                    self.update_preview()
                else:
                    self.highlight_text()

    def preprocess_for_ocr(self, arr):
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        eq = clahe.apply(gray)
        alpha = self.contrast_slider.value() / 100.0
        beta = self.brightness_slider.value()
        adj = cv2.convertScaleAbs(eq, alpha=alpha, beta=beta)
        gaussian_ksize = self.gaussian_slider.value()
        if gaussian_ksize % 2 == 0:
            gaussian_ksize += 1
        if gaussian_ksize > 1:
            adj = cv2.GaussianBlur(adj, (gaussian_ksize, gaussian_ksize), 0)
        thresh_val = self.threshold_slider.value()
        if thresh_val > 0:
            _, adj = cv2.threshold(adj, thresh_val, 255, cv2.THRESH_BINARY)
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        erosion_iter = self.erosion_slider.value()
        dilation_iter = self.dilation_slider.value()
        if erosion_iter > 0:
            adj = cv2.erode(adj, kernel, iterations=erosion_iter)
        if dilation_iter > 0:
            adj = cv2.dilate(adj, kernel, iterations=dilation_iter)
        return adj

    def preprocess_for_ocr_params(self, arr, brightness, contrast, threshold, gaussian):
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        eq = clahe.apply(gray)
        alpha = contrast / 100.0
        beta = brightness
        adj = cv2.convertScaleAbs(eq, alpha=alpha, beta=beta)
        gaussian_ksize = gaussian
        if gaussian_ksize % 2 == 0:
            gaussian_ksize += 1
        if gaussian_ksize > 1:
            adj = cv2.GaussianBlur(adj, (gaussian_ksize, gaussian_ksize), 0)
        if threshold > 0:
            _, adj = cv2.threshold(adj, threshold, 255, cv2.THRESH_BINARY)
        return adj

    def get_current_image_array(self):
        """Get the current image as a numpy array, applying invert if needed."""
        if not self._last_pixmap:
            return None
        qimg = self._last_pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        width, height = qimg.width(), qimg.height()
        bytes_per_line = qimg.bytesPerLine()
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        arr = np.array(ptr, dtype=np.uint8).reshape((height, bytes_per_line))
        arr = arr[:, :width * 3].reshape((height, width, 3))
        if self.invert_radio.isChecked():
            arr = cv2.rotate(arr, cv2.ROTATE_180)
        return arr

    def highlight_text(self):
        if not self._last_pixmap:
            return
        arr = self.get_current_image_array()
        if arr is None:
            return
        pre = self.preprocess_for_ocr(arr)
        if len(pre.shape) == 2:
            pre_rgb = cv2.cvtColor(pre, cv2.COLOR_GRAY2RGB)
        else:
            pre_rgb = pre
        qimg_pre = QImage(pre_rgb.data, pre_rgb.shape[1], pre_rgb.shape[0], pre_rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg_pre)
        current_transform = self.viewer.transform()
        self.viewer.set_image(pixmap, transform=current_transform)
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=NWSE.0123456789'
        data = pytesseract.image_to_data(pre, output_type=pytesseract.Output.DICT, config=custom_config)
        boxes = []
        expand = 4
        width, height = pre.shape[1], pre.shape[0]
        for i in range(len(data['text'])):
            text = data['text'][i]
            if text.strip() and any(c.isalnum() for c in text):
                text = text.replace('°', '').replace("'", '').replace('"', '')
                text = text.replace('’', "").replace('”', '')
                text = text.replace('"', "").replace("'", "")
                text = text.replace('_', '')
                text = text.replace('%', '')
                text = text.replace(',', '.')
                text = text.replace('$', 'S')
                text = text.replace('B', '8')
                text = re.sub(r'[a-z]', '', text)
                text = text.replace('-', '')
                if text.endswith('.'):
                    text = text[:-1]
                if text.strip():
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    x_exp = max(0, x - expand)
                    y_exp = max(0, y - expand)
                    w_exp = min(width - x_exp, w + 2 * expand)
                    h_exp = min(height - y_exp, h + 2 * expand)
                    min_box_w, min_box_h = 10, 10
                    if w_exp >= min_box_w and h_exp >= min_box_h:
                        boxes.append((x_exp, y_exp, w_exp, h_exp, text))
        merged_boxes = merge_boxes(
            boxes,
            x_threshold=self.merge_x_slider.value(),
            y_threshold=self.merge_y_slider.value()
        )
        self.viewer.highlight_text(merged_boxes)

    def update_preview(self):
        if not self._last_pixmap:
            return
        arr = self.get_current_image_array()
        if arr is None:
            return
        pre = self.preprocess_for_ocr(arr)
        if len(pre.shape) == 2:
            pre_rgb = cv2.cvtColor(pre, cv2.COLOR_GRAY2RGB)
        else:
            pre_rgb = pre
        qimg_pre = QImage(pre_rgb.data, pre_rgb.shape[1], pre_rgb.shape[0], pre_rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg_pre)
        self.viewer.set_image(pixmap)

    def clear_boxes(self):
        """Clear the highlighted OCR boxes."""
        self.viewer.ocr_boxes = []
        self.viewer.selected_box_idx = None
        self.viewer._rect_items = []
        self.viewer.scene().clear()
        if self._last_pixmap:
            self.viewer.set_image(self._last_pixmap)

    def auto_find_clean_bearings_and_distances(self):
        original_bearings = list(self.viewer.bearings)
        original_distances = list(self.viewer.distances)

        brightness_range = range(-50, 51, 25)
        contrast_range = range(50, 151, 25)
        threshold_range = range(0, 201, 50)
        gaussian_range = range(0, 7, 2)

        param_combinations = list(itertools.product(
            brightness_range, contrast_range, threshold_range, gaussian_range
        ))
        total_combinations = len(param_combinations)
        attempt = 0
        found_new_distance = False
        found_new_bearing = False
        new_distance = None
        new_bearing = None

        orig_arr = self.get_current_image_array()
        if orig_arr is None:
            print("No image loaded.")
            return

        sliders = [
            self.brightness_slider, self.contrast_slider, self.erosion_slider,
            self.dilation_slider, self.threshold_slider, self.gaussian_slider,
            self.merge_x_slider, self.merge_y_slider
        ]
        for slider in sliders:
            slider.blockSignals(True)
            slider.setVisible(False)

        def update_progress():
            percent = int((attempt / total_combinations) * 100)
            self.progress_bar.setValue(percent)
            self.progress_bar.setFormat(f"Attempts: {attempt}/{total_combinations}")
            QApplication.processEvents()

        for brightness, contrast, threshold, gaussian in param_combinations:
            attempt += 1
            pre = self.preprocess_for_ocr_params(orig_arr, brightness, contrast, threshold, gaussian)
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=NWSE.0123456789'
            data = pytesseract.image_to_data(pre, output_type=pytesseract.Output.DICT, config=custom_config)
            for i in range(len(data['text'])):
                text = data['text'][i]
                if not (text and any(c.isalnum() for c in text)):
                    continue
                selected = text.replace(' ', '')
                selected = re.sub(r'^(EE)', 'E', selected)
                selected = re.sub(r'^(WW)', 'W', selected)
                selected = re.sub(r'^(NN)', 'N', selected)
                selected = re.sub(r'^(SS)', 'S', selected)
                if selected and selected[0] in 'NWSE' and '.' in selected:
                    selected = selected.replace('.', '')
                if selected and not (selected[0] in 'NWSE' or selected[0].isdigit()):
                    m = re.search(r'[0-9]', selected)
                    if m:
                        selected = selected[m.start():]
                is_decimal = bool(re.match(r'^\d*\.\d+$', selected))
                is_bearing = selected and selected[0] in 'NWSE' and len(selected) >= 6
                if is_decimal and not found_new_distance and selected not in self.viewer.distances:
                    new_distance = selected
                    self.viewer.distances.append(selected)
                    found_new_distance = True
                    print(f"New distance found: {new_distance}")
                    print("All distances:", self.viewer.distances)
                if is_bearing and not found_new_bearing and selected not in self.viewer.bearings:
                    if len(selected) > 8:
                        continue
                    if any(x in selected for x in ("EE", "WW", "NN", "SS")):
                        continue
                    if not selected[-1] in ("E", "W"):
                        continue
                    if selected[-1].isdigit():
                        continue
                    new_bearing = selected
                    self.viewer.bearings.append(selected)
                    found_new_bearing = True
                    print(f"New bearing found: {new_bearing}")
                    print("All bearings:", self.viewer.bearings)
            if attempt % 10 == 0 or found_new_distance or found_new_bearing:
                update_progress()
            if found_new_distance and found_new_bearing:
                break

        for slider in sliders:
            slider.blockSignals(False)
            slider.setVisible(True)
        self.connect_sliders(self.highlight_text)

        update_progress()
        if not found_new_distance and not found_new_bearing:
            self.viewer.bearings = original_bearings
            self.viewer.distances = original_distances
            print("No new valid bearing or distance found in tested adjustment ranges.")
        elif found_new_distance and found_new_bearing:
            print("Auto-find complete: new distance and bearing found.")
        elif found_new_distance:
            print("Auto-find complete: new distance found, no new bearing.")
        elif found_new_bearing:
            print("Auto-find complete: new bearing found, no new distance.")
def set_dark_mode(app):
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    app.setStyleSheet("""
        QPushButton { background-color: #444; color: #fff; border-radius: 4px; padding: 6px 18px; }
        QPushButton:hover { background-color: #666; }
        QGraphicsView { background-color: #222; }
    """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_dark_mode(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())