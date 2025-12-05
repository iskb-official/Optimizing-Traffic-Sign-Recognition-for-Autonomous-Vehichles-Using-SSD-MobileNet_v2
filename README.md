# 🚗 Optimizing Traffic Sign Recognition for Autonomous Vehicles Using SSD-MobileNet_v2

This repository contains the implementation, models, results, and documentation for the paper titled:

> **“Optimizing Traffic Sign Recognition for Autonomous Vehicles Using SSD-MobileNet_v2”**  

---

## 📄 Overview

This project presents a lightweight and efficient **Traffic Sign Recognition System (TSRS)** designed for real-time operation on embedded autonomous vehicle platforms. It utilizes the **SSD-MobileNet_v2** model for traffic sign detection, and integrates:

- Real-time image acquisition and preprocessing  
- Sensor fusion using GPS and IMU  
- PWM-based motor control via TM4C123GH6PZ MCU  
- Wireless communication using UART/Bluetooth

---

## 📁 Repository Structure

```bash
├── /model/                  # Trained SSD-MobileNet_v2 model files
├── /datasets/               # Custom traffic sign datasets (sample)
├── /code/                   # Main Python and microcontroller source code
│   ├── detection.py
│   ├── pwm_control.ino
│   └── kalman_filter.py
├── /results/                # Test images, performance charts, and logs
├── paper/                   # IEEE paper PDF and LaTeX source (optional)
│   └── IEEE_Paper_Submission.pdf
├── README.md                # This file

---

## Citation

J. M. Mosaddeka, H. M. Shakib and A. Awais, "Optimizing Traffic Sign Recognition for Autonomous Vehicles Using SSD-MobileNet_v2," 2025 10th International Conference on Information Science, Computer Technology and Transportation (ISCTT), Nanchong, China, 2025, pp. 154-158, doi: 10.1109/ISCTT66403.2025.11137882. keywords: {Information science;Embedded systems;Merging;Transportation;Process control;Lighting;Sensor fusion;Real-time systems;Artificial intelligence;Autonomous vehicles;Autonomous vehicles;Artificial Intelligence;SSDMobileNet_v2;Embedded Systems;Sensor Fusion},

---
