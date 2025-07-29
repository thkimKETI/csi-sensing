# 📡 Real-Time CSI Visualization & Human Activity Recognition System

This repository provides a **real-time CSI sensing system** that processes Wi-Fi Channel State Information (CSI) for multi-device visualization and deep learning-based **human activity** and **location recognition**.  
It supports **1 or 2 ESP32-S3 devices** and includes both **SVM** and **deep neural models** (2D CNN, Transformer). CSI data is denoised using a **Butterworth filter**, and visualized via a **PyQt5 GUI** interface.

<div align="center">
  <img src="./csi_sensing_demo.png" width="600">
</div>

---

## 🔧 Features

- 🧠 Real-time inference of both **human activity** and **zone-based location**
- 📊 PyQt5 GUI with CSI heatmaps and prediction display
- 🧼 Preprocessing pipeline using **Butterworth filtering**
- 🧮 Supports **SVM**, **2D CNN**, and **Transformer** models
- 📡 Real-time CSI streaming via **MQTT** from **ESP32-S3 devices**
- 💾 Timestamped CSI data logging for training and analysis

---

## 🛠 Requirements

All dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

> 💡 Please ensure the correct versions are installed to avoid compatibility issues.

---

## 📦 Pretrained Weights

Download pretrained models and place them in the root directory (`./`):

| ESP Devices | Location Model | Activity Model |
|-------------|----------------|----------------|
| 1 Device    | Coming Soon    | Coming Soon    |
| 2 Devices   | [✅ Download](https://drive.google.com/file/d/1t1Di4KkHQOpncNmZmSdYPAN-0ZtC8Yqc/view?usp=sharing) | [✅ Download](https://drive.google.com/file/d/1reTq928hYPGpaUEugrAVeZoKxW_10U28/view?usp=sharing) |

> 🔬 Current models are trained using CSI from **2 ESP32-S3 devices** (input shape: `(1, 2, 180, 114)`).  
> ⏳ Models for single-ESP configurations will be released soon.

---

## 🚀 How to Run

### 1. Set up MQTT

Edit the MQTT config in `mqtt_config.py`:
```python
BROKER_ADDRESS = "localhost"
PORT = 1883
TOPIC = "csi/data"
MAC_ADDRESS = ["MAC_1", "MAC_2"]
```

### 2. Run by Model Type

Each script assumes **real-time inference (batch = 1)** and supports **both action and location recognition**, depending on the task selected internally.

#### ✅ Single ESP Device (`1D` Configuration)

```bash
# Neural model (2D CNN or Transformer)
python 1D_csi_sensing_nn.py

# Classical SVM model
python 1D_csi_sensing_svm.py
```
> 📌 CSI input shape: `(1, 1, 180, 114)` — single ESP32-S3 device  
> 🧠 Internally uses **2D CNN** for location and **Transformer** for action recognition

#### ✅ Two ESP Devices (`2D` Configuration)

```bash
# Neural model (2D CNN or Transformer)
python 2D_csi_sensing_nn.py

# Classical SVM model
python 2D_csi_sensing_svm.py
```
> 📌 CSI input shape: `(1, 2, 180, 114)` — stacked from 2 ESP32-S3 devices  
> 🧠 Internally uses the same model types (2D CNN or Transformer) depending on task

---

## 📐 Input Shape Summary

| Script | ESP Devices | Input Shape       | Description                       |
|--------|-------------|-------------------|-----------------------------------|
| `1D_csi_sensing_*.py` | 1         | `(1, 1, 180, 114)` | Single ESP (1 channel)            |
| `2D_csi_sensing_*.py` | 2         | `(1, 2, 180, 114)` | Multi-ESP (stacked by device)     |

---

## 🧠 Model Architecture Summary

| Script Name             | ESP Count | Tasks Supported        | Model Types Used          |
|------------------------|-----------|-------------------------|---------------------------|
| `1D_csi_sensing_nn.py` | 1         | Action / Location       | Transformer / 2D CNN      |
| `1D_csi_sensing_svm.py`| 1         | Action / Location       | SVM                       |
| `2D_csi_sensing_nn.py` | 2         | Action / Location       | Transformer / 2D CNN      |
| `2D_csi_sensing_svm.py`| 2         | Action / Location       | SVM                       |

> 🧩 The difference between 1D and 2D sensing is **not in network structure**, but in the number of ESPs and the depth of input tensor.

---

## 📄 Publications

1. **Taehyeon Kim**, et al.  
   _Wifi Channel State Information Sensing based on Introspective Metric Learning_  
   *IEEE ICSPIS*, 2024  
   [🔗 Link](https://ieeexplore.ieee.org/abstract/document/10812595)

2. **Taehyeon Kim**, et al.  
   _WiFi's Unspoken Tales: Deep Neural Network Decodes Human Behavior from Channel State Information_  
   *IEEE/ACM BDCAT*, 2023  
   [🔗 Link](https://dl.acm.org/doi/abs/10.1145/3632366.3632374)

3. **Taehyeon Kim**, et al.  
   _Neural Representation Learning for WiFi Channel State Information: A Unified Model for Action and Location Recognition_  
   *To be submitted*, IEEE Access, 2025

---

## 🧑‍💻 Maintainers

**Taehyeon Kim, Ph.D.**  
Senior Researcher, Korea Electronics Technology Institute (KETI)  
📧 [taehyeon.kim@keti.re.kr](mailto:taehyeon.kim@keti.re.kr) | 🌐 [Homepage](https://rcard.re.kr/detail/OISRzd7ua0tW0A1zMEwbKQ/information)

**Dongwoo Kang**  
Researcher, Korea Electronics Technology Institute (KETI)  
📧 [dongwookang@keti.re.kr](mailto:dongwookang@keti.re.kr)

---

## 📜 License

This project is released under a **custom license inspired by the MIT License**.  
See [`LICENSE`](./LICENSE.txt) for full terms.

⚠️ **Important Notice**:  
Use of this code — including academic research, model training, product integration, or distribution — **requires prior written permission** from the authors.

---

## 🙏 Acknowledgments

This research was supported by the Technology Innovation Program  
**[20026230] Development of AIoT device utilizing Channel State Information (CSI) for AI-based lifestyle recognition**,  
funded by the **Ministry of Trade, Industry & Energy (MOTIE)** and the **Korea Evaluation Institute of Industrial Technology (KEIT)**.
