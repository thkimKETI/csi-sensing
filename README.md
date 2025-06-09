# 📡 Real-Time CSI Visualization & Human Activity Recognition System

This repository provides a **real-time CSI sensing system** that processes Wi-Fi Channel State Information (CSI) for multi-device visualization and deep learning-based human activity & location recognition. It supports **4 ESP devices** and applies **Butterworth filtering** to extract meaningful features for inference.

---

## 🔧 Features

- 🧠 Real-time inference of both human activity and location (zone-based)
- 📊 Interactive PyQt5 GUI with real-time CSI heatmaps and multi-label outputs
- 🧼 Denoising pipeline using **Butterworth filtering**
- 🧮 Plug-and-play with **CNN or Transformer-based models**
- 📡 MQTT-based real-time CSI ingestion from ESP devices
- 💾 Automatic timestamped data logging

---

## 🗂️ Directory Structure

```
csi-inference-system/
├── models/                     # Model definitions (CNN, Transformer)
├── weight/
│   └── esp01_weight/           # Pretrained weights for activity/location
├── datasets/                   # Saved CSI data
├── mqtt_config.py              # MQTT configuration
├── main.py                     # Main script (launches pipeline and GUI)
└── README.md
```

---

## 📦 Requirements

- Python 3.8+
- MQTT Broker (e.g., Mosquitto)
- ESP32/ESP8266 sending CSI packets
- `torch`, `pyqt5`, `pyqtgraph`, `numpy`, `pandas`, `scipy`, `paho-mqtt`, `tqdm`

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**
```text
torch
pyqt5
pyqtgraph
numpy
pandas
scipy
tqdm
paho-mqtt
```

---

## 🔗 Download Pretrained Weights

Download the pretrained models for **location** and **activity recognition** from Google Drive:

| Model Type     | Path                                 | Download Link |
|----------------|--------------------------------------|----------------|
| Location Model | `csi/weight/esp01_weight/loc.pt`     | [📥 Download](https://drive.google.com/file/d/1t1Di4KkHQOpncNmZmSdYPAN-0ZtC8Yqc/view?usp=sharing) |
| Activity Model | `csi/weight/esp01_weight/act.pt`     | [📥 Download](https://drive.google.com/file/d/1reTq928hYPGpaUEugrAVeZoKxW_10U28/view?usp=sharing) |

> After downloading, place them in the following directory structure:
>
> ```bash
> csi/
> └── weight/
>     └── esp01_weight/
>         ├── loc.pt
>         └── act.pt
> ```

If the folders do not exist, create them manually.

---

## 🚀 How to Run

### 1. Set up MQTT

Edit `mqtt_config.py` to match your broker:
```python
BROKER_ADDRESS = "localhost"  # or your broker's IP
PORT = 1883
TOPIC = "csi/data"
```

### 2. Launch the system

```bash
python main.py --inf_sec 3 --model CNN --acquire
```

Arguments:
- `--inf_sec`: Inference time window in seconds (default: 3)
- `--model`: Model type (`CNN` or `Transformer`)
- `--acquire`: If provided, logs CSI data to file

---

## 🖥️ GUI Overview

- **PORT 0–3:** CSI heatmaps (raw, filtered, foreground)
- **Radio Button:** Toggle between raw and filtered view
- **Button:** Start/stop CSI acquisition
- **Labels:** Display current time, inferred location, and activity

---

## 🧪 Output

- Location Classes: Z0, Z1, Z2, Z3
- Activity Classes: Exr, Sit, Stand, Walk
- Multi-label Output: Example: Time: 15:42:12 | Location: Z1 | Activity: Walk

---

## 📌 Notes

- Supports 4 ESP devices transmitting CSI packets.
- Real-time inference and visualization tested on standard desktops.
- Foreground is computed as difference from a Butterworth-filtered average.

---

## 🧑‍💻 Maintainer

**Taehyeon Kim, Ph.D.**  
Senior Researcher, Korea Electronics Technology Institute (KETI)  
📧 [taehyeon.kim@keti.re.kr](mailto:taehyeon.kim@keti.re.kr)  🌐 [Homepage](https://rcard.re.kr/detail/OISRzd7ua0tW0A1zMEwbKQ/information)

**Dongwoo Kang**  
Researcher, Korea Electronics Technology Institute (KETI)  
📧 [dongwookang@keti.re.kr](mailto:dongwookang@keti.re.kr) 

---

## 📜 License

This project is released under a custom license inspired by the MIT License. See [`LICENSE`](./LICENSE.txt) file for details.

⚠️ **Important Notice**  
Use of this code—commercial or non-commercial, including academic research, model training, product integration, and distribution—**requires prior written permission** from the author. Unauthorized usage will be treated as a license violation.
