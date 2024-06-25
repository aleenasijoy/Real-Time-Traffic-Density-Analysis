# Real-Time-Traffic-Density-Analysis
A Python project for real-time traffic density analysis using YOLOv8 and LSTM for traffic prediction.
# Traffic Density Analysis

This repository contains two Python scripts for real-time traffic density analysis using YOLOv8 for object detection and LSTM for traffic density prediction.

## Overview

The project consists of two main scripts:

1. **TRaffic.py**: Captures live video feed, detects vehicles in two lanes, classifies traffic density, and saves the data to CSV files.
2. **trafficlstmfinal.py**: Loads the traffic density data, preprocesses it, trains LSTM models, and makes predictions for future traffic densities.

## Requirements

- Python 3.x
- OpenCV
- YOLOv8
- TensorFlow
- Pandas
- Numpy
- scikit-learn

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/TrafficDensityAnalysis.git
    cd TrafficDensityAnalysis
    ```

2. **Install the required packages**:
    ```bash
    pip install opencv-python-headless ultralytics tensorflow pandas numpy scikit-learn
    ```

## Usage

### TrafficDensity.py

This script captures live video feed from a camera, processes the frames using YOLOv8 to detect vehicles, and classifies the traffic density in two lanes.

To run the script:
```bash
python TrafficDensity.py
