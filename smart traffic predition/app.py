   
from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import cv2
import os

app = Flask(__name__)

# ---------------- LOAD ML MODEL ----------------
model = joblib.load("xgboost_traffic_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- LOAD YOLO ----------------
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# ---------------- HOME ----------------
@app.route('/')
def index():
    return render_template("index.html")

# ---------------- LOGIN ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == "jershika" and password == "project":
            return redirect(url_for('predict'))
        else:
            return "Invalid Username or Password"

    return render_template("login.html")

# ---------------- OPEN CV FUNCTION (UPDATED) ----------------
def detect_traffic(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return 0

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    # ---- DETECTION ----
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Only strong vehicle detections
            if confidence > 0.6 and classes[class_id] in ["car", "bus", "truck"]:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # ---- REMOVE DUPLICATES (NMS) ----
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)

    vehicle_count = len(indexes)

    print("Detected vehicles:", vehicle_count)

    return vehicle_count

# ---------------- PREDICT ----------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'GET':
        return render_template("predict.html")

    if request.method == 'POST':

        # ---- IMAGE INPUT ----
        file = request.files.get('image')

        if file and file.filename != "":
            image_path = "static/uploaded.jpg"
            file.save(image_path)

            vehicle_count = detect_traffic(image_path)

            # ---- IMPROVED TRAFFIC LOGIC ----
            if vehicle_count <= 10:
                traffic = "Low"
                green_time = 20
            elif vehicle_count <= 25:
                traffic = "Medium"
                green_time = 40
            elif vehicle_count <= 45:
                traffic = "High"
                green_time = 70
            else:
                traffic = "Heavy"
                green_time = 90

            yellow_time = 5
            red_time = 120 - (green_time + yellow_time)

            return render_template(
                "result.html",
                traffic=traffic,
                green_time=green_time,
                yellow_time=yellow_time,
                red_time=red_time
            )

        # ---- MANUAL INPUT (ML MODEL) ----
        else:
            date = request.form['date']
            hour = int(request.form['hour'])
            minute = int(request.form['minute'])

            day_number = int(request.form['day'])
            car = float(request.form['car'])
            bike = float(request.form['bike'])
            bus = float(request.form['bus'])
            truck = float(request.form['truck'])

            if request.form['total_vehicle']:
                total_vehicles = float(request.form['total_vehicle'])
            else:
                total_vehicles = car + bike + bus + truck

            input_data = np.array([[day_number, car, bike, bus, truck, total_vehicles]])
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)[0]

            # ---- ML TRAFFIC LABEL ----
            if prediction == 0:
                traffic = "Low"
            elif prediction == 1:
                traffic = "Normal"
            elif prediction == 2:
                traffic = "High"
            else:
                traffic = "Heavy"

            # ---- SIGNAL LOGIC ----
            total_cycle = 120

            if traffic == "Low":
                green_time = 20
            elif traffic == "Normal":
                green_time = 40
            elif traffic == "High":
                green_time = 70
            else:
                green_time = 90

            yellow_time = 5
            red_time = total_cycle - (green_time + yellow_time)

            return render_template(
                "result.html",
                traffic=traffic,
                green_time=green_time,
                yellow_time=yellow_time,
                red_time=red_time
            )

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)