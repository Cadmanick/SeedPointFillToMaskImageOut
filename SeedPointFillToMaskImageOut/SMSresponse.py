import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
import numpy as np

def parse_sms_mms_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    messages = []
    prev_body = None

    # Parse SMS messages
    for sms in root.findall('sms'):
        body = sms.get('body', '')
        sms_type = sms.get('type', '')
        date_ms = sms.get('date', '')
        if sms_type == '1':
            direction = 'FER'
        elif sms_type == '2':
            direction = 'NWP'
        else:
            direction = 'unknown'
        try:
            date_time = datetime.fromtimestamp(int(date_ms) / 1000)
            date_time_str = date_time.strftime('%Y-%m-%d %H:%M:%S')
            date_int = int(date_ms)
        except Exception:
            date_time_str = 'unknown'
            date_int = 0
        if body == prev_body:
            continue
        messages.append({
            'text': body,
            'type': direction,
            'date_time': date_time_str,
            'date_int': date_int,
            'message_kind': 'SMS'
        })
        prev_body = body

    # Parse MMS messages
    for mms in root.findall('mms'):
        msg_box = mms.get('msg_box', '')
        if msg_box == '1':
            direction = 'FER'
        elif msg_box == '2':
            direction = 'NWP'
        else:
            direction = 'unknown'
        date_ms = mms.get('date', '')
        try:
            date_time = datetime.fromtimestamp(int(date_ms) / 1000)
            date_time_str = date_time.strftime('%Y-%m-%d %H:%M:%S')
            date_int = int(date_ms)
        except Exception:
            date_time_str = 'unknown'
            date_int = 0
        text = ''
        parts = mms.find('parts')
        if parts is not None:
            for part in parts.findall('part'):
                if part.get('ct') == 'text/plain':
                    text += part.get('text', '')
        if text == prev_body:
            continue
        messages.append({
            'text': text,
            'type': direction,
            'date_time': date_time_str,
            'date_int': date_int,
            'message_kind': 'MMS'
        })
        prev_body = text

    return messages

def compute_response_times(messages):
    """Add a 'response_time' field to each message, showing the time since the previous message from NWP to FER."""
    prev_index = None
    for i, msg in enumerate(messages):
        if i == 0:
            msg['response_time'] = ""
            prev_index = i
            continue
        prev_msg = messages[prev_index]
        # Only compute response time if previous is NWP and current is FER
        if prev_msg['type'] == 'NWP' and msg['type'] == 'FER':
            delta = msg['date_int'] - prev_msg['date_int']
            # Format as H:MM:SS
            hours, remainder = divmod(delta // 1000, 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                msg['response_time'] = f"{hours}:{minutes:02}:{seconds:02}"
            else:
                msg['response_time'] = f"{minutes}:{seconds:02}"
        else:
            msg['response_time'] = ""
        prev_index = i
    return messages

class SMSMMSViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMS/MMS XML Viewer")
        self.setGeometry(100, 100, 1000, 600)
        self.messages = []

        layout = QVBoxLayout()
        self.open_button = QPushButton("Open XML File")
        self.open_button.clicked.connect(self.open_and_display)
        layout.addWidget(self.open_button)

        self.graph_button = QPushButton("Show Response Time Graph")
        self.graph_button.clicked.connect(self.show_response_time_graph)
        layout.addWidget(self.graph_button)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Date/Time", "Sender", "Kind", "Text", "Response Time"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table)

        self.setLayout(layout)

    def open_and_display(self):
        xml_path, _ = QFileDialog.getOpenFileName(
            self, "Select SMS/MMS XML file", "", "XML files (*.xml);;All files (*)"
        )
        if not xml_path:
            return
        try:
            self.messages = parse_sms_mms_xml(xml_path)
            self.messages = sorted(self.messages, key=lambda x: x['date_int'])
            self.messages = compute_response_times(self.messages)
            self.display_messages(self.messages)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse XML:\n{e}")

    def display_messages(self, messages):
        self.table.setRowCount(len(messages))
        for row, msg in enumerate(messages):
            items = [
                QTableWidgetItem(msg['date_time']),
                QTableWidgetItem(msg['type']),
                QTableWidgetItem(msg['message_kind']),
                QTableWidgetItem(msg['text']),
                QTableWidgetItem(msg.get('response_time', ""))
            ]
            if msg['type'] == "NWP":
                for item in items:
                    item.setBackground(QColor(255, 0, 0, 127))
            for col, item in enumerate(items):
                self.table.setItem(row, col, item)

        header = self.table.horizontalHeader()
        # Set all columns to interactive
        header.setSectionResizeMode(QHeaderView.Interactive)
        # Set the "Text" column (index 3) to 200px wide initially
        self.table.setColumnWidth(3, 200)

    def show_response_time_graph(self):
        # Collect response times and their months, only for NWP->FER
        months = []
        response_times = []
        for msg in self.messages:
            rt = msg.get('response_time', "")
            if rt and msg['date_time'] != 'unknown' and msg['type'] == 'FER':
                # Convert date_time to month only (YYYY-MM)
                month_str = msg['date_time'][:7]
                # Convert response_time to seconds
                parts = [int(p) for p in rt.split(":")]
                if len(parts) == 3:
                    seconds = parts[0]*3600 + parts[1]*60 + parts[2]
                elif len(parts) == 2:
                    seconds = parts[0]*60 + parts[1]
                else:
                    continue
                months.append(month_str)
                response_times.append(seconds)
        if not months:
            QMessageBox.information(self, "No Data", "No response times to plot.")
            return

        # Group by month and compute average (in hours)
        from collections import defaultdict
        month_to_times = defaultdict(list)
        for m, t in zip(months, response_times):
            month_to_times[m].append(t)
        sorted_months = sorted(month_to_times.keys())
        avg_times = [np.mean(month_to_times[m]) / 3600.0 for m in sorted_months]  # Convert to hours

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(sorted_months, avg_times, marker='o')
        plt.xticks(rotation=45)
        plt.ylabel("Average Response Time (hours)")
        plt.xlabel("Month")
        plt.title("Average NWP→FER Response Time By Month")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = SMSMMSViewer()
    viewer.show()
    sys.exit(app.exec_())