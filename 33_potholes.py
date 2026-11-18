import os
from ultralytics import YOLO

# Use absolute path to avoid YOLO looking in C:\Users\datasets
current_dir = os.getcwd()
dataset_path = os.path.join(current_dir, "pothole_dataset")

dataset_yaml = f"""
path: {dataset_path}
train: images/train 
val: images/val


#classes
names:
    0: pothole
"""


with open("potholes.yaml","w") as f:
    f.write(dataset_yaml)
    
print("config 'potholes.yaml' created")

print("Training trick")

model = YOLO('yolov8n.pt')
results = model.train(data='potholes.yaml',epochs =3,imgsz=640)
print("Test Training Complete")