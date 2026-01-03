# api/models/classifier.py
import onnxruntime as ort
import numpy as np
import cv2

class DogBreedClassifier:
    def __init__(self, model_path: str = "models/classification/model_dynamic_quant.onnx"):
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider', 'CUDAExecutionProvider'] 
            if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        self.class_names = [
            'Chihuahua',
            'Japanese_spaniel',
            'Maltese_dog',
            'Pekinese'
        ]
    
    def predict(self, image_bytes: bytes):
        input_tensor = self.preprocess(image_bytes)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        logits = outputs[0][0]
        class_id = int(np.argmax(logits))
        
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        all_breeds = []
        for i in range(4):
            all_breeds.append({
                'id': i,
                'name': self.class_names[i],
                'probability': float(probabilities[i]),
                'is_predicted': (i == class_id)
            })
        
        all_breeds.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'predicted_breed': {
                'id': class_id,
                'name': self.class_names[class_id],
                'probability': float(probabilities[class_id]),
                'confidence_percentage': round(float(probabilities[class_id]) * 100, 2)
            },
            'all_breeds': all_breeds,
            'top_3_breeds': all_breeds[:3]
        }
    
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, axis=0)
