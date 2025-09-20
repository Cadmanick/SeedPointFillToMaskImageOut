#test
import sys
import cv2
import numpy as np
import pytesseract
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QCheckBox, QDialog, QTextEdit, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRect, QPoint

class OCRPreviewDialog(QDialog):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OCR Result")
        self.resize(500, 300)
        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setText(text)
        layout.addWidget(self.text_edit)

class ImageLabel(QLabel):
    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setMouseTracking(True)
        self.image = None
        self.display_image = None
        self.start_point = None
        self.end_point = None
        self.selection_rect = None
        self.selecting = False
        self.ocr_callback = None

        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.last_pan_point = None
        self.fit_on_load = True
        self.selected_candidate_idx = None
        self.text_candidates = []  # List of (x1, y1, x2, y2, text)

        self.setEnabled(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_image(self, image):
        self.image = image
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.selection_rect = None
        self.fit_on_load = True
        self.text_candidates = []
        self.update_display()

    def set_text_candidates(self, candidates):
        self.text_candidates = candidates
        self.update_display()

    def fit_to_widget(self):
        if self.image is None:
            return
        iw, ih = self.image.shape[1], self.image.shape[0]
        ww, wh = self.width(), self.height()
        if iw == 0 or ih == 0 or ww == 0 or wh == 0:
            return
        scale = min(ww / iw, wh / ih)
        self.zoom_factor = scale
        self.pan_offset = QPoint((ww - iw * scale) // 2, (wh - ih * scale) // 2)

    def resizeEvent(self, event):
        if self.fit_on_load:
            self.fit_to_widget()
        self.update_display()
        super().resizeEvent(event)

    def update_display(self):
        if self.image is None:
            self.clear()
            return
        # Fit to widget on load
        if self.fit_on_load:
            self.fit_to_widget()
            self.fit_on_load = False

        iw, ih = self.image.shape[1], self.image.shape[0]
        scale = self.zoom_factor
        new_w, new_h = int(iw * scale), int(ih * scale)
        resized = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        qimg = QImage(resized.data, resized.shape[1], resized.shape[0],
                      resized.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        # Create a blank pixmap for panning
        canvas = QPixmap(self.width(), self.height())
        canvas.fill(QColor(30, 30, 30))
        painter = QPainter(canvas)
        painter.drawPixmap(self.pan_offset, pixmap)
        # Draw selection rectangle if present
        if self.selection_rect:
            pen = QPen(QColor(255, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.selection_rect)
        # Draw text candidates and OCR results
        if hasattr(self, "text_candidates") and self.text_candidates:
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            # Only draw text if you want, but do not draw any rectangles here.
            for idx, (x1, y1, x2, y2, text) in enumerate(self.text_candidates):
                p1 = self.image_to_widget(x1, y1)
                p2 = self.image_to_widget(x2, y2)
                rect = QRect(p1, p2)
        painter.end()
        self.setPixmap(canvas)

    def widget_to_image(self, point):
        # Convert widget (label) coordinates to image coordinates
        x = (point.x() - self.pan_offset.x()) / self.zoom_factor
        y = (point.y() - self.pan_offset.y()) / self.zoom_factor
        if self.image is not None:
            x = np.clip(x, 0, self.image.shape[1] - 1)
            y = np.clip(y, 0, self.image.shape[0] - 1)
        return int(x), int(y)

    def image_to_widget(self, x, y):
        # Convert image coordinates to widget coordinates
        wx = int(x * self.zoom_factor + self.pan_offset.x())
        wy = int(y * self.zoom_factor + self.pan_offset.y())
        return QPoint(wx, wy)

    def mousePressEvent(self, event):
        if self.image is None:
            return
        if event.button() == Qt.LeftButton:
            clicked_point = event.pos()
            found = False
            for idx, (x1, y1, x2, y2, _) in enumerate(self.text_candidates):
                p1 = self.image_to_widget(x1, y1)
                p2 = self.image_to_widget(x2, y2)
                rect = QRect(p1, p2).normalized()
                if rect.contains(clicked_point):
                    self.selected_candidate_idx = idx
                    self.update_display()
                    found = True
                    break
            if not found:
                self.selected_candidate_idx = None
                self.selecting = True
                self.start_point = event.pos()
                self.end_point = event.pos()
                self.selection_rect = QRect(self.start_point, self.end_point)
                self.update_display()
        elif event.button() == Qt.RightButton:
            self.last_pan_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.selecting and self.image is not None:
            self.end_point = event.pos()
            self.selection_rect = QRect(self.start_point, self.end_point).normalized()
            self.update_display()
        elif self.last_pan_point is not None:
            delta = event.pos() - self.last_pan_point
            self.pan_offset += delta
            self.last_pan_point = event.pos()
            self.update_display()

    def mouseReleaseEvent(self, event):
        if self.selecting and self.image is not None:
            self.selecting = False
            self.end_point = event.pos()
            self.selection_rect = QRect(self.start_point, self.end_point).normalized()
            self.update_display()
            if self.ocr_callback:
                roi = self.get_selected_roi()
                self.ocr_callback(roi)
        if event.button() == Qt.RightButton:
            self.last_pan_point = None

    def wheelEvent(self, event):
        if self.image is None:
            return
        # Zoom in/out
        angle = event.angleDelta().y()
        factor = 1.25 if angle > 0 else 0.8
        old_zoom = self.zoom_factor
        self.zoom_factor = np.clip(self.zoom_factor * factor, 0.05, 20)
        # Adjust pan to keep mouse position stable
        mouse_pos = event.pos()
        x_img, y_img = self.widget_to_image(mouse_pos)
        new_mouse_pos = self.image_to_widget(x_img, y_img)
        self.pan_offset += mouse_pos - new_mouse_pos
        self.update_display()

    def get_selected_roi(self):
        if self.image is None or self.selection_rect is None:
            return None
        # Convert selection rect from widget to image coordinates
        x1, y1 = self.widget_to_image(self.selection_rect.topLeft())
        x2, y2 = self.widget_to_image(self.selection_rect.bottomRight())
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        roi = self.image[y1:y2, x1:x2]
        return roi

    def mouseDoubleClickEvent(self, event):
        if self.image is None:
            return
        if event.button() == Qt.LeftButton:
            print("Double click detected")  # Debug
            clicked_point = event.pos()
            for idx, (x1, y1, x2, y2, _) in enumerate(self.text_candidates):
                p1 = self.image_to_widget(x1, y1)
                p2 = self.image_to_widget(x2, y2)
                rect = QRect(p1, p2).normalized()
                if rect.contains(clicked_point):
                    print(f"Double clicked box {idx}")  # Debug
                    self.selected_candidate_idx = idx
                    self.update_display()
                    if self.main_window:
                        self.main_window.show_candidate_ocr_result(idx)
                   
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PNG OCR GUI")
        self.resize(900, 700)

        self.image = None
        self.preprocess_enabled = True
        self.rotate_enabled = False

        # Widgets
        self.image_label = ImageLabel(main_window=self, parent=self)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.ocr_callback = self.process_ocr

        self.load_button = QPushButton("Load PNG")
        self.load_button.clicked.connect(self.load_png)

        self.preproc_checkbox = QCheckBox("Enable Preprocessing")
        self.preproc_checkbox.setChecked(True)
        self.preproc_checkbox.stateChanged.connect(self.toggle_preprocessing)

        self.rotate_checkbox = QCheckBox("Rotate 180")
        self.rotate_checkbox.setChecked(False)
        self.rotate_checkbox.stateChanged.connect(self.toggle_rotation)

        self.highlight_button = QPushButton("Highlight Text Candidates")
        self.highlight_button.clicked.connect(self.highlight_text_candidates)

        self.ocr_candidates_button = QPushButton("OCR Text Candidates")  # <-- Add this button
        self.ocr_candidates_button.clicked.connect(self.ocr_text_candidates)  # <-- Connect

        # Layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.load_button)
        top_layout.addWidget(self.preproc_checkbox)
        top_layout.addWidget(self.rotate_checkbox)
        top_layout.addWidget(self.highlight_button)
        top_layout.addWidget(self.ocr_candidates_button)  # <-- Add to layout
        top_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_png(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open PNG Image", "", "PNG Files (*.png);;All Files (*)"
        )
        if not file_path:
            return
        img = cv2.imread(file_path)
        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image = img
        self.update_image_display()

    def update_image_display(self):
        img = self.image
        if img is None:
            self.image_label.set_image(None)
            return
        if self.rotate_enabled:
            img = cv2.rotate(img, cv2.ROTATE_180)
        self.image_label.set_image(img)

    def toggle_preprocessing(self, state):
        self.preprocess_enabled = (state == Qt.Checked)

    def toggle_rotation(self, state):
        self.rotate_enabled = (state == Qt.Checked)
        self.update_image_display()

    def process_ocr(self, roi):
        if roi is None:
            return
        img = roi.copy()
        if self.preprocess_enabled:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        config = "--psm 6"
        text = pytesseract.image_to_string(img, config=config)
        dlg = OCRPreviewDialog(text, self)
        dlg.exec_()

    def highlight_text_candidates(self):
        img = self.image_label.image
        if img is None:
            return

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h_img, w_img = img.shape[:2]

        # 1. Adaptive threshold to get binary image
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
        )

        # 2. Morphological closing with a wide kernel to connect letters into words
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
        morph = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 3. Find contours (potential words)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        word_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 15 < w < w_img * 0.95 and 10 < h < h_img * 0.5:
                word_boxes.append((x, y, x + w, y + h))

        # 4. Optionally, merge overlapping/close boxes (for robustness)
        def boxes_close(b1, b2, gap=15):
            x11, y11, x12, y12 = b1
            x21, y21, x22, y22 = b2
            return not (x12 + gap < x21 or x22 + gap < x11 or y12 + gap < y21 or y22 + gap < y11)

        def group_boxes(boxes, gap=15):
            n = len(boxes)
            parent = list(range(n))
            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i
            def union(i, j):
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj
            for i in range(n):
                for j in range(i + 1, n):
                    if boxes_close(boxes[i], boxes[j], gap=gap):
                        union(i, j)
            groups = {}
            for i in range(n):
                root = find(i)
                groups.setdefault(root, []).append(boxes[i])
            merged = []
            for group in groups.values():
                xs = [b[0] for b in group] + [b[2] for b in group]
                ys = [b[1] for b in group] + [b[3] for b in group]
                merged.append((min(xs), min(ys), max(xs), max(ys)))
            return merged

        word_boxes = group_boxes(word_boxes, gap=15)

        # 5. Draw blue rectangles for each word
        img_with_boxes = img.copy()
        for (x1, y1, x2, y2) in word_boxes:
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Blue

        self.image_label.set_image(img_with_boxes)

        # 6. Store rectangles as text candidates (for later OCR)
        buffer = 10
        text_candidates = []
        for (x1, y1, x2, y2) in word_boxes:
            bx1 = max(0, x1 - buffer)
            by1 = max(0, y1 - buffer)
            bx2 = min(w_img, x2 + buffer)
            by2 = min(h_img, y2 + buffer)
            text_candidates.append((bx1, by1, bx2, by2, ""))
        self.image_label.set_text_candidates(text_candidates)

    def ocr_text_candidates(self):
        img = self.image_label.image
        candidates = self.image_label.text_candidates
        def perform_ocr_on_roi(roi):
            img_roi = roi.copy()
            if self.preprocess_enabled:
                if len(img_roi.shape) == 3:
                    img_roi = cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY)
                img_roi = cv2.adaptiveThreshold(
                    img_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
            config = "--psm 6"
            text = pytesseract.image_to_string(img_roi, config=config).strip()
            return text

        if img is None or not candidates:
            QMessageBox.information(self, "OCR Results", "No text candidates to process.")
            return

        results = []
        # Use the same logic as process_ocr for each candidate
        for idx, (x1, y1, x2, y2, _) in enumerate(candidates, 1):
            roi = img[y1:y2, x1:x2]
            text = perform_ocr_on_roi(roi)
            results.append(f"Box {idx}:\n{text if text else '[No text]'}\n")

        msg = "\n".join(results) if results else "No text found in candidates."
        QMessageBox.information(self, "OCR Results", msg)

    def show_candidate_ocr_result(self, idx):
        img = self.image_label.image
        candidates = self.image_label.text_candidates
        if img is None or not candidates or idx < 0 or idx >= len(candidates):
            return
        x1, y1, x2, y2, _ = candidates[idx]
        roi = img[y1:y2, x1:x2]
        img_roi = roi.copy()
        if self.preprocess_enabled:
            if len(img_roi.shape) == 3:
                img_roi = cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY)
            img_roi = cv2.adaptiveThreshold(
                img_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        config = "--psm 6"
        text = pytesseract.image_to_string(img_roi, config=config)
        QMessageBox.information(
            self,
            f"OCR Result for Box {idx+1}",
            text if text.strip() else "[No text]"
        )
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())