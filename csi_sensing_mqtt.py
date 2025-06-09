# 정리된 CSI 실시간 시각화 및 추론 시스템 코드

# 표준 라이브러리
import sys, os, time, argparse, datetime, queue
from io import StringIO
from multiprocessing import Process, Queue, Value, Array, Lock, Manager
from threading import Thread
from collections import defaultdict

# 외부 라이브러리
import numpy as np
import torch
import csv, json, re
import pyqtgraph as pg
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
from scipy.signal import butter, filtfilt
import paho.mqtt.client as mqtt

# 사용자 정의 모듈
from models import *
from mqtt_config import BROKER_ADDRESS, PORT, TOPIC

# 전역 설정
SEQUENCE_LENGTH = 180
CSI_SAVE_PATH = f"/csi/datasets/{datetime.datetime.now().strftime('%m%d')}"
os.makedirs(CSI_SAVE_PATH, exist_ok=True)

# 공유 객체
manager = Manager()
LABELS = manager.dict({"time": "", "occ": "", "loc": "", "act": ""})
isPushedBtn = Value('b', False)
exit_flag = Value('b', False)

# 공유 배열
csi_raw_array_shared = [Array('d', SEQUENCE_LENGTH * 192) for _ in range(2)]
csi_bt_array_shared = [Array('d', SEQUENCE_LENGTH * 114) for _ in range(2)]
csi_fg_array_shared = [Array('d', SEQUENCE_LENGTH * 114) for _ in range(2)]

# 큐 및 프로세스 리스트
data_queue = Queue(maxsize=180)
inference_queue = Queue(maxsize=180)
storage_queue = Queue(maxsize=180)
PROCESSES = []

# CSI 전처리

def get_amplitude(csi):
    csi = np.concatenate((csi[254:368], csi[132:246]), axis=0)
    even, odd = csi[::2], csi[1::2]
    return np.sqrt(np.square(even) + np.square(odd)) / 15.0

def butterworth_filter(data, cutoff, fs, order):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data, axis=0)

def apply_filter(data_chunk):
    filtered = butterworth_filter(data_chunk, cutoff=0.1, fs=2, order=1)
    avg = np.mean(filtered, axis=0)
    static = np.tile(avg, (data_chunk.shape[0], 1))
    return filtered, filtered, static

# MQTT 수신

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8")
    parts = payload.split(',', 4)
    mac = parts[0]
    entries = re.findall(r'\{CSI_DATA,[^}]+\}', parts[4])
    for entry in entries:
        try:
            row = next(csv.reader(StringIO(entry.strip('{}'))))
            timestamp, raw_json = row[2], row[-1]
            csi = json.loads(raw_json)
            data_queue.put((mac, timestamp, csi))
        except:
            continue

# 모델 로딩

def load_model(path, n_classes):
    model = WiFiCSICNNAttention(num_classes=n_classes, num_esp=2)
    model.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
    model.eval()
    return model

# 추론 프로세스

def inference_process(queue, label_dict, exit_flag):
    model_loc = load_model("/csi/weight/esp01_weight/loc.pt", 4)
    model_act = load_model("/csi/weight/esp01_weight/act.pt", 4)
    while not exit_flag.value:
        try:
            port, data = queue.get(timeout=1)
            tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).permute(0, 1, 2, 3)
            with torch.no_grad():
                loc_pred = model_loc(tensor).argmax().item()
                act_pred = model_act(tensor).argmax().item()
            label_dict["time"] = datetime.datetime.now().strftime("%H:%M:%S")
            label_dict["loc"] = ["Z0","Z1","Z2","Z3"][loc_pred]
            label_dict["act"] = ["Exr","Sit","Stand","Walk"][act_pred]
        except:
            continue

# 시각화 GUI

class CSIViewer(QMainWindow):
    def __init__(self, label_dict):
        super().__init__()
        self.label_dict = label_dict
        self.initUI()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_view)
        self.timer.start(50)

    def initUI(self):
        self.setWindowTitle("CSI Real-Time Viewer")
        self.setGeometry(100, 100, 1200, 600)
        layout = QVBoxLayout()
        self.graphWidget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphWidget)

        self.plots, self.heatmaps = [], []
        for i in range(2):
            plot = self.graphWidget.addPlot(title=f"PORT {i}")
            img = pg.ImageItem()
            plot.addItem(img)
            self.plots.append(plot)
            self.heatmaps.append(img)
            if i == 0:
                self.graphWidget.nextRow()

        self.status = QLabel("Ready")
        layout.addWidget(self.status)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def update_view(self):
        for i in range(2):
            with csi_bt_array_shared[i].get_lock():
                arr = np.frombuffer(csi_bt_array_shared[i].get_obj()).reshape(SEQUENCE_LENGTH, 114)
            self.heatmaps[i].setImage(arr, levels=(0, 1), autoLevels=False)
        self.status.setText(
            f"Time: {self.label_dict.get('time', '')}  | Loc: {self.label_dict.get('loc', '')} | Act: {self.label_dict.get('act', '')}"
        )

    def closeEvent(self, event):
        exit_flag.value = True
        for p in PROCESSES:
            p.terminate()
            p.join()
        event.accept()

# 메인 실행

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_ADDRESS, PORT)
    client.loop_start()

    inf_proc = Process(target=inference_process, args=(inference_queue, LABELS, exit_flag))
    inf_proc.start()
    PROCESSES.append(inf_proc)

    app = QApplication(sys.argv)
    viewer = CSIViewer(LABELS)
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
