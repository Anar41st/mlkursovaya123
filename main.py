from ultralytics import YOLO
dataset_path = "C:/Users/nrbch/Downloads/carsv8"
yolov8_path = "yolov8.pt"
model = YOLO(yolov8_path)
results = model.train(data=f"{dataset_path}/data.yaml", epochs=30, batch=16)
model.save("carsv8.pt")