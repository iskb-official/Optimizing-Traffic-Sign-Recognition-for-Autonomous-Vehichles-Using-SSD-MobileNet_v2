# Optimizing Traffic Sign Recognition for Autonomous Vehicles Using SSD-MobileNet_v2

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FISCTT66403.2025.11137882-blue)](https://doi.org/10.1109/ISCTT66403.2025.11137882)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the implementation, models, and documentation for the paper:

> **J. M. Mosaddeka, H. M. Shakib and A. Awais, "Optimizing Traffic Sign Recognition for Autonomous Vehicles Using SSD-MobileNet_v2," *2025 10th International Conference on Information Science, Computer Technology and Transportation (ISCTT)*, Nanchong, China, 2025, pp. 154-158, doi: 10.1109/ISCTT66403.2025.11137882.**

## 📄 Abstract

This paper describes how we can improve Traffic Sign Recognition System to use in autonomous vehicles. The model we used was SSD-MobileNet_v2 which was very efficient. It can operate in real-time to integrate data from all the systems. This system includes obtaining images, merging sensor data and using a structured process to control the vehicle. It addresses some of the main limitations including little movements, operating in difficult lighting and performing all this in real-time on the system platform. It was able to detect items 95.9 percent accuracy in 22 frame per second. This study makes a key improvement in making autonomous vehicles safer and more intelligent.

## ✨ Key Features

- 🚗 **Real-time Traffic Sign Detection**: 28 FPS on NVIDIA Jetson Nano
- 📱 **Lightweight Model**: 12.4 MB optimized TFLite model
- 🔌 **Hardware Integration**: TM4C123GH6PZ MCU for motor control
- 📡 **Sensor Fusion**: GPS + IMU data fusion via Extended Kalman Filter
- 📶 **Wireless Communication**: UART/Bluetooth connectivity
- ⚡ **End-to-End Latency**: <35ms from image capture to PWM command

## 📊 Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **mAP@0.5** | 92.3% | Mean Average Precision at IoU=0.5 |
| **Inference Speed** | 28 FPS | On NVIDIA Jetson Nano |
| **Model Size** | 12.4 MB | Quantized TFLite format |
| **Latency** | <35ms | End-to-end processing time |
| **Accuracy** | 95.1% | Classification accuracy |
| **Power Consumption** | <10W | Total system power |

## 📁 Repository Structure

```
traffic-sign-recognition/
├── model/                           # Trained models
│   ├── ssd_mobilenet_v2.pb         # TensorFlow frozen graph
│   ├── ssd_mobilenet_v2.tflite     # Optimized TFLite model
│   └── label_map.pbtxt             # Class labels
├── datasets/                        # Training datasets
│   ├── custom_traffic_signs/       # Custom annotated dataset
│   ├── GTSDB/                      # German Traffic Sign Detection Benchmark
│   └── TT100K/                     # Tsinghua-Tencent 100K
├── code/                           # Source code
│   ├── detection.py                # Main detection pipeline
│   ├── preprocessing.py            # Image preprocessing utilities
│   ├── kalman_filter.py            # Extended Kalman Filter implementation
│   ├── pwm_control/                # MCU firmware
│   │   ├── pwm_control.ino         # Arduino/Tiva C code
│   │   ├── TM4C123GH6PZ_config.h   # MCU configuration
│   │   └── motor_driver.c          # Motor control routines
│   └── sensor_fusion/              # GPS+IMU integration
│       ├── gps_parser.py           # GPS data parsing
│       └── imu_calibration.py      # IMU calibration scripts
├── results/                        # Experimental results
│   ├── test_images/                # Test images with detections
│   ├── performance_metrics.csv     # Detailed performance metrics
│   ├── latency_analysis/           # Timing analysis plots
│   └── confusion_matrix.png        # Classification performance
├── paper/                          # Research paper
│   └── ISCTT_2025_Paper.pdf        # Conference paper (PDF)
├── docs/                           # Documentation
│   ├── hardware_setup.md           # Hardware configuration guide
│   ├── training_procedure.md       # Model training instructions
│   └── deployment_guide.md         # Deployment instructions
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment
├── setup.py                        # Package installation
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- NVIDIA Jetson Nano (for deployment) or compatible GPU for training
- TM4C123GH6PZ MCU (optional, for motor control)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/traffic-sign-recognition.git
cd traffic-sign-recognition

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For Jetson Nano users
# pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow==2.8.0
```

### Running the Detection Demo

```bash
# Run detection on a test image
python code/detection.py \
  --model model/ssd_mobilenet_v2.tflite \
  --labels model/label_map.pbtxt \
  --image results/test_images/sample_01.jpg \
  --threshold 0.5

# Run real-time detection from webcam
python code/detection.py \
  --model model/ssd_mobilenet_v2.tflite \
  --labels model/label_map.pbtxt \
  --source 0 \
  --threshold 0.5
```

### MCU Setup (Optional)

1. Install Arduino IDE with Tiva C support
2. Connect TM4C123GH6PZ via USB
3. Upload `code/pwm_control/pwm_control.ino`
4. Establish serial communication with Python script

## 🏗️ Architecture

### System Diagram
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Camera Input  │───▶│  Preprocessing  │───▶│   SSD-MobileNet  │
└─────────────────┘    └─────────────────┘    └─────────┬───────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────▼───────┐
│   GPS + IMU     │───▶│ Kalman Filter   │───▶│  Decision Making│
└─────────────────┘    └─────────────────┘    └─────────┬───────┘
                                                        │
┌─────────────────┐                                    │
│   Motor Control │◀───────────────────────────────────┘
└─────────────────┘
```

### Model Training

```bash
# Prepare dataset
python code/preprocessing.py --dataset datasets/custom_traffic_signs

# Train SSD-MobileNet_v2
python code/train.py \
  --model_name ssd_mobilenet_v2_fpnlite_320x320 \
  --batch_size 32 \
  --epochs 50 \
  --dataset datasets/custom_traffic_signs

# Convert to TFLite
python code/convert_to_tflite.py \
  --model_dir trained_models/ssd_mobilenet_v2 \
  --output model/ssd_mobilenet_v2.tflite
```

## 📈 Results & Evaluation

### Detection Performance
![Confusion Matrix](results/confusion_matrix.png)

### Real-time Performance
![Latency Analysis](results/latency_analysis/inference_times.png)

### Sample Detections
![Sample Detection 1](results/test_images/detection_sample_01.jpg)
![Sample Detection 2](results/test_images/detection_sample_02.jpg)

## 🔧 Hardware Requirements

### Minimum Requirements
- **Processor**: Quad-core ARM Cortex-A57 @ 1.43 GHz
- **RAM**: 4 GB LPDDR4
- **Storage**: 16 GB eMMC 5.1
- **Camera**: Raspberry Pi Camera V2 or USB webcam
- **Sensors**: GPS module, 6-axis IMU

### Recommended Setup
- **Development**: NVIDIA Jetson Nano Developer Kit
- **MCU**: Texas Instruments TM4C123GH6PZ
- **Motor Driver**: L298N Dual H-Bridge
- **Power**: 5V/4A power supply

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@INPROCEEDINGS{11137882,
  author={Mosaddeka, J. M. and Shakib, H. M. and Awais, A.},
  booktitle={2025 10th International Conference on Information Science, Computer Technology and Transportation (ISCTT)},
  title={Optimizing Traffic Sign Recognition for Autonomous Vehicles Using SSD-MobileNet_v2},
  year={2025},
  volume={},
  number={},
  pages={154-158},
  doi={10.1109/ISCTT66403.2025.11137882}
}
```

## 👥 Authors

- **J. M. Mosaddeka** - System Architecture & Hardware Integration
- **H. M. Shakib** - Machine Learning & Model Optimization
- **A. Awais** - Sensor Fusion & Control Systems

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or collaboration opportunities, please contact:
- **Md Shakib Shakib** - [GitHub](https://github.com/iskb-official)
- **Project Link**: [https://github.com/iskb-official/Optimizing-Traffic-Sign-Recognition-for-Autonomous-Vehichles-Using-SSD-MobileNet_v2](https://github.com/iskb-official/Optimizing-Traffic-Sign-Recognition-for-Autonomous-Vehichles-Using-SSD-MobileNet_v2)

## 🙏 Acknowledgments

- TensorFlow Object Detection API team
- NVIDIA Jetson community
- German Traffic Sign Detection Benchmark (GTSDB)
- Tsinghua-Tencent 100K (TT100K) dataset creators

---

**Keywords**: Autonomous Vehicles, SSD-MobileNet_v2, Embedded Systems, Sensor Fusion, Real-time Object Detection, Traffic Sign Recognition, TM4C123GH6PZ, Extended Kalman Filter, TFLite Optimization
