from ultralytics import YOLO
YOLO("yolov8n.pt").export(format="openvino", imgsz=320)