import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import os
import numpy as np
import csv

model = YOLO('yolov8l.pt')

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

midpoint = width // 2

vehicle_classes = ['car', 'truck', 'bus']

def process_image(image, midpoint, model, vehicle_classes):
    results = model(image)
    lane1_vehicle_count = 0
    lane2_vehicle_count = 0

    for result in results:
        for box in result.boxes:
            cls_name = model.names[box.cls.item()]
            if cls_name in vehicle_classes:
                x1, _, x2, _ = map(int, box.xyxy.tolist()[0])
                center_x = (x1 + x2) // 2
                if center_x < midpoint:
                    lane1_vehicle_count += 1
                else:
                    lane2_vehicle_count += 1

    return lane1_vehicle_count, lane2_vehicle_count

stop_camera = False
start_time = time.time()
lane1_densities = []
lane2_densities = []
hourly_lane1_densities = []
hourly_lane2_densities = []

current_date = datetime.now().date()

start_hour = datetime.now().hour

time_periods = [f"{i:02d}:00-{i+1:02d}:00" for i in range(24)]
lane1_header = ["Date"] + time_periods
lane2_header = ["Date"] + time_periods

def classify_density(vehicle_count):
    if vehicle_count > 8:
        return "Heavy"
    elif vehicle_count > 4:
        return "Medium"
    else:
        return "Smooth"

def save_to_csv(filename, header, data):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists or os.path.getsize(filename) == 0:
            writer.writerow(header)
        writer.writerows(data)

try:
    while not stop_camera:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture image")
            break

        cv2.line(frame, (midpoint, 0), (midpoint, height), (0, 255, 0), 2)
        cv2.imshow('Frame', frame)

        if time.time() - start_time >= 10:
            start_time = time.time()
            datetime_now = datetime.now()

            lane1_vehicle_count, lane2_vehicle_count = process_image(frame, midpoint, model, vehicle_classes)

            lane1_density = classify_density(lane1_vehicle_count)
            lane2_density = classify_density(lane2_vehicle_count)

            lane1_densities.append(lane1_density)
            lane2_densities.append(lane2_density)

            print(datetime_now.strftime("%Y-%m-%d %H:%M:%S"))
            print("Lane 1 Vehicles:", lane1_vehicle_count, "Density:", lane1_density)
            print("Lane 2 Vehicles:", lane2_vehicle_count, "Density:", lane2_density)

            if len(lane1_densities) >= 2 and len(lane2_densities) >= 2:  # 120 * 30 seconds = 1 hour
                lane1_density_counts = {"Heavy": 0, "Medium": 0, "Smooth": 0}
                lane2_density_counts = {"Heavy": 0, "Medium": 0, "Smooth": 0}

                for density in lane1_densities:
                    lane1_density_counts[density] += 1

                for density in lane2_densities:
                    lane2_density_counts[density] += 1

                lane1_mean_density = max(lane1_density_counts, key=lane1_density_counts.get)
                lane2_mean_density = max(lane2_density_counts, key=lane2_density_counts.get)

                hourly_lane1_densities.append((start_hour, lane1_mean_density))
                hourly_lane2_densities.append((start_hour, lane2_mean_density))

                print("Lane 1 Mean Traffic Density after 1 Hour:", lane1_mean_density)
                print("Lane 2 Mean Traffic Density after 1 Hour:", lane2_mean_density)

                lane1_densities = []
                lane2_densities = []
                start_hour = (start_hour + 1) % 24  

                if datetime_now.date() != current_date:
                    lane1_row = [current_date] + [""] * 24
                    lane2_row = [current_date] + [""] * 24
                    for hour, density in hourly_lane1_densities:
                        lane1_row[hour + 1] = density
                    for hour, density in hourly_lane2_densities:
                        lane2_row[hour + 1] = density

                    save_to_csv('lane1_traffic_density.csv', lane1_header, [lane1_row])
                    save_to_csv('lane2_traffic_density.csv', lane2_header, [lane2_row])

                    current_date = datetime_now.date()
                    hourly_lane1_densities = []
                    hourly_lane2_densities = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit key pressed. Exiting...")
            stop_camera = True
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    if lane1_densities and lane2_densities:
        lane1_density_counts = {"Heavy": 0, "Medium": 0, "Smooth": 0}
        lane2_density_counts = {"Heavy": 0, "Medium": 0, "Smooth": 0}

        for density in lane1_densities:
            lane1_density_counts[density] += 1

        for density in lane2_densities:
            lane2_density_counts[density] += 1

        lane1_mean_density = max(lane1_density_counts, key=lane1_density_counts.get)
        lane2_mean_density = max(lane2_density_counts, key=lane2_density_counts.get)

        hourly_lane1_densities.append((start_hour, lane1_mean_density))
        hourly_lane2_densities.append((start_hour, lane2_mean_density))

        print("Lane 1 Mean Traffic Density:", lane1_mean_density)
        print("Lane 2 Mean Traffic Density:", lane2_mean_density)

    camera.release()
    cv2.destroyAllWindows()

    if hourly_lane1_densities and hourly_lane2_densities:
        lane1_row = [current_date] + [""] * 24
        lane2_row = [current_date] + [""] * 24
        for hour, density in hourly_lane1_densities:
            lane1_row[hour + 1] = density
        for hour, density in hourly_lane2_densities:
            lane2_row[hour + 1] = density

        save_to_csv('lane1_traffic_density.csv', lane1_header, [lane1_row])
        save_to_csv('lane2_traffic_density.csv', lane2_header, [lane2_row])
