import os
from ultralytics import YOLO

dataset_yaml = """
path: ./pothole_dataset
train: images/train 
val: images/val


#classes
names:
    0: pothole
"""