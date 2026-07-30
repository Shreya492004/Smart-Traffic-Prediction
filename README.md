# 🚦 Smart Traffic Prediction

An intelligent web-based traffic prediction and congestion control system that uses **Machine Learning** and **Computer Vision** to predict traffic density and optimize traffic signal timings. The application supports both **image-based vehicle detection** using YOLOv3 and **manual traffic data prediction** using an XGBoost model.

---

## 📖 Overview

Traffic congestion is a major challenge in urban transportation, leading to increased travel time, fuel consumption, and environmental pollution. This project provides a smart solution by predicting traffic density and dynamically allocating traffic signal timings based on real-time traffic conditions.

Users can either upload a traffic image for automatic vehicle detection or manually enter vehicle counts to predict congestion levels.

---

## ✨ Features

- 🚗 Vehicle detection using YOLOv3 and OpenCV
- 🤖 Traffic prediction using XGBoost Machine Learning model
- 📤 Upload traffic images for automatic analysis
- 📝 Manual traffic data prediction
- 🚦 Dynamic traffic signal timing generation
- 🔐 Secure login authentication
- 📊 Predicts Low, Normal, High, and Heavy traffic levels
- 💻 Responsive and user-friendly web interface

---

## 🛠️ Technologies Used

### Frontend
- HTML5
- CSS3
- Bootstrap 5

### Backend
- Python
- Flask

### Machine Learning
- XGBoost
- Scikit-learn
- Joblib

### Computer Vision
- OpenCV
- YOLOv3

### Dataset
- CSV Traffic Dataset

---

## 📂 Project Structure

```
Smart-Traffic-Prediction/
│
├── app.py
├── train_model.py
├── traffic_data.csv
├── xgboost_traffic_model.pkl
├── scaler.pkl
├── yolov3.cfg
├── yolov3.weights
├── coco.names
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── predict.html
│   └── result.html
│
├── static/
│   ├── image/
│   └── uploaded.jpg
│
└── README.md
```

---

## ⚙️ Working

### Image-Based Prediction

1. Upload a traffic image.
2. YOLOv3 detects vehicles (cars, buses, and trucks).
3. Vehicle count is calculated.
4. Traffic density is classified.
5. Signal timings are generated automatically.

### Manual Prediction

Users can enter:

- Day of Week
- Car Count
- Bike Count
- Bus Count
- Truck Count
- Total Vehicle Count

The XGBoost model predicts the traffic level and generates optimized traffic signal timings.

---

## 🚦 Traffic Levels

- 🟢 Low
- 🟡 Normal
- 🟠 High
- 🔴 Heavy

---

## 🚥 Signal Timing

| Traffic Level | Green | Yellow | Red |
|---------------|-------|--------|-----|
| Low | 20 sec | 5 sec | 95 sec |
| Normal | 40 sec | 5 sec | 75 sec |
| High | 70 sec | 5 sec | 45 sec |
| Heavy | 90 sec | 5 sec | 25 sec |

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Smart-Traffic-Prediction.git
```

### Install Dependencies

```bash
pip install flask
pip install numpy
pip install pandas
pip install opencv-python
pip install scikit-learn
pip install xgboost
pip install joblib
```

### Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📸 Screenshots

### Home Page
<img src="screenshots/home.png" width="700">

### Login Page
<img src="screenshots/login.png" width="700">

### Prediction Page
<img src="screenshots/predict.png" width="700">

### Result Page
<img src="screenshots/result.png" width="700">

> Replace the images above with your own screenshots stored inside a `screenshots` folder.

---

## 🔮 Future Enhancements

- Live CCTV camera integration
- Google Maps API integration
- Real-time traffic monitoring
- Emergency vehicle priority detection
- Traffic analytics dashboard
- IoT-enabled smart traffic signal control

---

## 👩‍💻 Author

**Shreya S**

Java Full Stack Developer

- GitHub: https://github.com/Shreya492004

---

## 📄 License

This project is developed for educational and academic purposes.
