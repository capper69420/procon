from ultralytics import YOLO
# Load a pretrained YOLO11 model (e.g., small variant)
model = YOLO("yolo11s.pt")
# Train on custom dataset
results = model.train(
   data="path/to/data.yaml", # dataset config
   epochs=50,
   imgsz=640,
   plots=True
)