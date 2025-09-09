import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtPdf import QPdfDocument
from PyQt5.QtPdfWidgets import QPdfView

class PDFViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Viewer")
        self.resize(800, 600)

        self.pdf_view = QPdfView()
        self.pdf_doc = QPdfDocument(self)
        self.pdf_view.setDocument(self.pdf_doc)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.pdf_view)
        self.setCentralWidget(central_widget)

        self.open_pdf()

    def open_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        if file_path:
            self.pdf_doc.load(file_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = PDFViewer()
    viewer.show()
    sys.exit(app.exec_())