from ultralytics import YOLO
import cv2

print("loading the model")
model = YOLO('yolov8n.pt')

image_url = "https://ultralytics.com/images/bus.jpg"
print("scanning the image...")

results = model(image_url)

result = results[0]
print("\n Detections...")
print(f"Detected {len(result.boxes)} objects.")

save_path = 'bus_pred.jpg'
result.save(filename=save_path)

print(f"image saved to {save_path}")

try:
    img = cv2.imread(save_path)
    cv2.imshow("YOLO_VISION",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print("some error occured")