import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import cvzone
from tracker import Tracker
import os
from datetime import datetime, timedelta

# Print current working directory
print("Current working directory:", os.getcwd())

# Change directory if needed
os.chdir('N:/v10/v10/yolov10')

# Verify the change
print("New working directory:", os.getcwd())

# Open the file
try:
    my_file = open("coco.txt", "r")
    print("File opened successfully")
except FileNotFoundError:
    print("File not found. Please check the file path.")


# Load the model and class list
model = YOLO("yolov10s.pt")
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Initialize the tracker
tracker = Tracker()

# Set the line position and offset
cy1 = 425
offset = 6

# Initialize variables
listcardown = []
count = 0

# Dictionary to track vehicle positions and timestamps
vehicle_status = {}

# Create directory to save images if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('sample.mp4')

while True:
    ret, frame = cap.read()
    count += 1
    if count % 2 != 0:
        continue
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 600))

    results = model(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
    
    bbox_idx = tracker.update(list)
    current_time = datetime.now()
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        if id not in vehicle_status:
            vehicle_status[id] = {"last_position": (cx, cy), "last_time": current_time, "status": "Free Flowing"}
        
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            if listcardown.count(id) == 0:
                listcardown.append(id)
                # Crop and save the car image
                car_image = frame[y3:y4, x3:x4]
                resized_car_image = cv2.resize(car_image, (300, 300))
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                cv2.imwrite(f"images/car_{timestamp}.jpg", resized_car_image)

        # Check vehicle movement status
        if vehicle_status[id]["last_position"] == (cx, cy):
            if current_time - vehicle_status[id]["last_time"] > timedelta(seconds=1):
                vehicle_status[id]["status"] = "Congestion"
        else:
            vehicle_status[id]["status"] = "Free Flowing"
            vehicle_status[id]["last_position"] = (cx, cy)
            vehicle_status[id]["last_time"] = current_time
        
        # Display the status
        status_text = vehicle_status[id]["status"]
        cvzone.putTextRect(frame, status_text, (x3, y3 - 20), 1, 1, colorR=(0, 255, 0) if status_text == "Free Flowing" else (0, 0, 255))

    cv2.line(frame, (13, 485), (966, 508), (255, 255, 255), 1)
    cardown = len(listcardown)
    cvzone.putTextRect(frame, f'Cardown:-{cardown}', (50, 60), 2, 2)
    
    cv2.imshow("RGB", frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
