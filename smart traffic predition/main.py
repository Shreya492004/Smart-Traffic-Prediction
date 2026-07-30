import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load classes
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Load image
img = cv2.imread("C:/smart traffic predition/traffic.jpg")
height, width, _ = img.shape

# Convert image to blob
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get output layers
layer_names = net.getUnconnectedOutLayersNames()
outputs = net.forward(layer_names)

vehicle_count = 0

# Detect objects
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Detect only vehicles
        if confidence > 0.5 and classes[class_id] in ["car", "bus", "truck", "motorbike"]:
            vehicle_count += 1

print("Vehicle Count:", vehicle_count)

# Show image
cv2.imshow("Traffic Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
