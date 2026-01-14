from ultralytics import YOLO
import cv2
import math

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

classNames = model.names
print("starting webcam press 'q' to quit")

while True:
    success,img = cap.read()
    if not success:
        break
    
    results = model(img,stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            
            conf = math.ceil(box.conf[0]*100)/100
            
            cls = int(box.cls[0])
            currentclass = classNames[cls]
            
            print(f"Detected: {currentclass}({conf})")
            cv2.putText(img,f"{currentclass}{conf}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            
    cv2.imshow('YOLO Webcam',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()