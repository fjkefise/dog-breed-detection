# api/models/detector.py
from ultralytics import YOLO
import cv2
import numpy as np

class DogBreedDetector:
    def __init__(self, model_path: str = "models/detection/best.pt"):
        self.model = YOLO(model_path)
        self.class_names = {
            0: 'Chihuahua',
            1: 'Japanese_spaniel',
            2: 'Maltese_dog',
            3: 'Pekinese'
        }
    
    def detect(self, image_bytes: bytes, conf_threshold: float = 0.5):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results = self.model(img, imgsz=256, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box, conf, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    breed_id = int(cls_id)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'breed_id': breed_id,
                        'breed_name': self.class_names.get(breed_id, f"class_{breed_id}"),
                        'width': x2 - x1,
                        'height': y2 - y1
                    })
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections
