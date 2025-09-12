import sys
import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog

def pdf_ocr_text(pdf_path):
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--psm 6')
        print(f"\n--- Page {page_num + 1} ---\n{text}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    pdf_path, _ = QFileDialog.getOpenFileName(
        None, "Select PDF file", "", "PDF files (*.pdf)"
    )
    if pdf_path:
        pdf_ocr_text(pdf_path)
    else:
        print("No file selected.")
