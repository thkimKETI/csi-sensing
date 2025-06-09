# 📡 Real-Time CSI Visualization & Human Activity Recognition System

This repository provides a **real-time CSI sensing system** that processes Wi-Fi Channel State Information (CSI) for multi-device visualization and deep learning-based human activity & location recognition. It supports **4 ESP devices** and applies **Butterworth filtering** to extract meaningful features for inference.

<div align="center">
  <img src="./csi_sensing_demo.png" width="600">
</div>

---

## 🔧 Features

- 🧠 Real-time inference of both human activity and location (zone-based)
- 📊 Interactive PyQt5 GUI with real-time CSI heatmaps and multi-label outputs
- 🧼 Denoising pipeline using **Butterworth filtering**
- 🧮 Plug-and-play with **CNN or Transformer-based models**
- 📡 MQTT-based real-time CSI ingestion from **ESP32-S3 devices**
- 💾 Automatic timestamped data logging

---

## 🔧 Requirements

All required Python packages are listed in the `requirements.txt` file.  
Please install them by running:

```bash
pip install -r requirements.txt
```

> 💡 Make sure to check the `requirements.txt` file for the exact library versions used in this system.

---

## 🔗 Pretrained Weights

📦 Download pretrained models and place them under `csi/weight/esp01_weight/`
> 📁 If the folders do not exist, please create them manually.

| Number of Devices | Location Model | Activity Model |
|-------------------|----------------|----------------|
| 1 Device          | TBD            | TBD            |
| 2 Devices         | [✅ Download](https://drive.google.com/file/d/1t1Di4KkHQOpncNmZmSdYPAN-0ZtC8Yqc/view?usp=sharing) | [✅ Download](https://drive.google.com/file/d/1reTq928hYPGpaUEugrAVeZoKxW_10U28/view?usp=sharing) |
| 4 Devices         | TBD            | TBD            |
> 🧪 The currently released models are trained using CSI data from **2 ESP32-S3 devices**.  
> ⏳ Models for 1 and 4 devices will be released later.


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

---

## 📄 Publications

This project is supported by a series of research efforts. Related publications include:

1. **Taehyun Kim**, et al.  
   _Wifi Channel State Information Sensing based on Introspective Metric Learning_  
   *International Conference on Signal Processing and Information Security*, IEEE, 2024
   [🔗 Link](https://ieeexplore.ieee.org/abstract/document/10812595)

2. **Taehyun Kim**, et al.  
   _WiFi's Unspoken Tales: Deep Neural Network Decodes Human Behavior from Channel State Information_  
   *International Conference on Big Data Computing, Applications and Technologies*, IEEE/ACM, 2023
   [🔗 Link]([https://ieeexplore.ieee.org/abstract/document/1081259](https://dl.acm.org/doi/abs/10.1145/3632366.3632374)

3. **Taehyun Kim**, et al.  
   _Neural Represenation Learning for WiFi Channel State Information: A Unified Model for Action and Location Recognition_  
   *To be submitted*, IEEE Access, 2025.

---

## 🙏 Acknowledgments

This work was supported by the Technology Innovation Program [20026230, Development of AIoT device utilizing Channel State information(CSI) for a AI-based lifestyle recognition] funded by the Ministry of Trade, Industry & Energy(MOTIE) and Korea Evaluation Institute of Industrial Technology(KEIT).
