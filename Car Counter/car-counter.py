import os
import math
import requests
from ultralytics import YOLO
import cv2
import cvzone

# Base URL and endpoint for fetching class names
baseURL = "https://098bf248-eb78-4902-b948-da19cc35a08f.mock.pstmn.io"
endpoint = "/classNames"

mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

try:
    response = requests.get(f'{baseURL}{endpoint}')
    response.raise_for_status()
    classNames = response.json()
except requests.exceptions.RequestException as e:
    print(f"Error fetching class names: {e}")
    classNames = []

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['YOLACT_BACKEND'] = 'mps'

cap = cv2.VideoCapture("../Videos/buslane.mp4")
model = YOLO('../yolo-weights/yolov8n.pt')

frame_skip = 2
frame_count = 0

try:
    while True:
        success, img = cap.read()
        if not success:
            break

        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)

        _, mask_binary = cv2.threshold(mask_resized, 1, 255, cv2.THRESH_BINARY)

        imgRegion = cv2.bitwise_and(img, img, mask=mask_binary)

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        results = model(imgRegion, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                label = "HULI KAMOTE! = 5k"
                color = (0, 0, 255)

                if classNames[cls] == "bus":
                    label = "bus"
                    color = (0, 255, 0)

                cvzone.cornerRect(img, (x1, y1, w, h), l=9, colorR=color)
                cvzone.putTextRect(img, f'{label} {conf}', (max(0, x1), max(35, y1)), scale=1.5, thickness=1, offset=3,
                                   colorR=color)

        cv2.imshow("Image", img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
