# 📡 Real-Time CSI Visualization & Human Activity Recognition System

This repository provides a **real-time CSI sensing system** that processes Wi-Fi Channel State Information (CSI) for multi-device visualization and deep learning-based **human activity** and **location recognition**.  
It supports **1 or 2 ESP32-S3 devices** and includes both **SVM** and **deep neural models** (2D CNN, Transformer). CSI data is denoised using a **Butterworth filter**, and visualized via a **PyQt5 GUI** interface.

<div align="center">
  <img src="./csi_sensing_demo.gif" width="550">
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

Download pretrained models below. Place them in the root directory (`./`) after downloading:

### ✅ Neural Network Models

| ESP Devices | Location Model (2D CNN) | Activity Model (Transformer) |
|-------------|-------------------------|-------------------------------|
| 1 Device    | [📥 Download](https://drive.google.com/file/d/1-DPTa-9CMdbhfu3GDwL2p36T3q4jzIPi/view?usp=sharing) | [📥 Download](https://drive.google.com/file/d/1OkoV1TwQydeG0R5AZbvKDd3y8CrKsPua/view?usp=sharing) |
| 2 Devices   | [📥 Download](https://drive.google.com/file/d/1YQeQKdtMYZAnctU5RaQji7DVGoKeZxx_/view?usp=sharing) | [📥 Download](https://drive.google.com/file/d/15rCJSn-3p6Xfni8g_2sw8Hq94O1SqexV/view?usp=sharing) |

### ✅ SVM Models

| ESP Devices | Location Model (SVM) | Activity Model (SVM) |
|-------------|----------------------|------------------------|
| 1 Device    | [📥 Download](https://drive.google.com/file/d/17jUJa_uPXEo6bkPl4vTXvqy6wvpr1_dN/view?usp=sharing) | [📥 Download](https://drive.google.com/file/d/1SDWjVzpFo63i7JZcJ5ktC4P215xFW6fq/view?usp=sharing)|
| 2 Devices   | [📥 Download](https://drive.google.com/file/d/1j8yWBLsJ0pFZb19lf0mDa2y-BDXM11ai/view?usp=sharing) | [📥 Download](https://drive.google.com/file/d/1bPLGhz_xzEKq9VXHXnpI1cf5d53NlQTt/view?usp=sharing) |

---

## 🚀 How to Run

### 1. Set up MQTT

#### ✅ For 1D sensing (single ESP)

```bash
# Edit mqtt_config_1D.py 
BROKER_ADDRESS = "localhost"
PORT = 1883
TOPIC = "csi/data"
MAC_ADDRESS = ["MAC_1"]
```

#### ✅ Two ESP Devices (`2D` Configuration)

```bash
# Edit mqtt_config_2D.py 
BROKER_ADDRESS = "localhost"
PORT = 1883
TOPIC = "csi/data"
MAC_ADDRESS = ["MAC_1", "MAC_2"]
```

### 2. Run by Model Type

Each script assumes **real-time inference (batch = 1)** and supports **both action and location recognition**, depending on the task selected internally.

#### ✅ Single ESP Device (`1D` Configuration)

```bash
# Neural model (2D CNN for localization, Transformer for action recognition)
python 1D_csi_sensing_nn.py

# Classical SVM model
python 1D_csi_sensing_svm.py
```

#### ✅ Two ESP Devices (`2D` Configuration)

```bash
# Neural model (2D CNN for localization, Transformer for action recognition)
python 2D_csi_sensing_nn.py

# Classical SVM model
python 2D_csi_sensing_svm.py
```
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
