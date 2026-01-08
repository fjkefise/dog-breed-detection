import onnxruntime as ort
import numpy as np
import cv2
import os

class DogBreedDetector:
    def __init__(self, model_path: str = None):
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            model_path = os.path.join(project_root, "models", "detection", "yolo11n_30_epochs_quantized.onnx")
        
        providers = ['CPUExecutionProvider']
        try:
            # Пробуем добавить GPU если доступно
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        except:
            pass
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.img_height = 256
        self.img_width = 256
        self.last_image_size = None
        self.class_names = ['Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese']

    def detect(self, image_bytes: bytes, conf_threshold: float = 0.5):
        input_tensor = self._preprocess(image_bytes)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        detections = self._postprocess(outputs[0], conf_threshold, self.last_image_size)
        return detections

    def _preprocess(self, image_bytes: bytes):
        """Препроцессинг изображения для YOLO модели 256x256"""
        # Декодируем изображение
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Не удалось декодировать изображение")
        
        original_shape = img.shape[:2]  # (height, width)
        
        # Конвертируем BGR в RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Изменяем размер с сохранением пропорций (letterbox)
        img_resized, ratio, pad = self._letterbox(img, (self.img_width, self.img_height))
        
        # Нормализация [0, 255] -> [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Транспонирование (H, W, C) -> (C, H, W)
        img_transposed = img_normalized.transpose(2, 0, 1)
        
        # Добавляем батч-размерность
        img_final = np.expand_dims(img_transposed, axis=0)
        
        # Сохраняем информацию для постпроцессинга
        self.last_image_size = (original_shape, ratio, pad)
        
        return img_final.astype(np.float32)

    def _letterbox(self, img, new_shape=(256, 256), color=(114, 114, 114)):
        """Letterbox изменение размера с сохранением пропорций"""
        shape = img.shape[:2] 
        
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        # Делим отступы пополам
        dw /= 2
        dh /= 2
        
        # Изменяем размер
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # Добавляем отступы
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img, r, (dw, dh)

    def _postprocess(self, output, conf_threshold, original_info):
        if output is None or output.size == 0:
            return []
        
        print(f"Формат вывода: {output.shape}")
        original_shape, ratio, pad = original_info
        original_h, original_w = original_shape
        
        detections = []
        
        if len(output.shape) == 3 and output.shape[0] == 1 and output.shape[1] == 8:
            predictions = output[0]  # [8, 1344]
            class_probs = predictions[4:8, :]
            max_conf = class_probs.max()
            
            # Пройдем по всем предсказаниям
            for i in range(predictions.shape[1]):
                pred = predictions[:, i]
                x_center, y_center, width, height = pred[0], pred[1], pred[2], pred[3]
                
                # Вероятности классов
                class_probs = pred[4:8]
                class_id = np.argmax(class_probs)
                confidence = class_probs[class_id]
                
                if confidence < conf_threshold:
                    continue
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                # Масштабируем к оригинальному размеру
                x1 = (x1 - pad[0]) / ratio
                y1 = (y1 - pad[1]) / ratio
                x2 = (x2 - pad[0]) / ratio
                y2 = (y2 - pad[1]) / ratio
                
                # Обрезаем
                x1 = max(0, min(int(x1), original_w))
                y1 = max(0, min(int(y1), original_h))
                x2 = max(0, min(int(x2), original_w))
                y2 = max(0, min(int(y2), original_h))
                
                box_width = x2 - x1
                box_height = y2 - y1  
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence),
                    'breed_id': int(class_id),
                    'breed_name': self.class_names[class_id],
                })
            
            if detections:
                boxes = np.array([d['bbox'] for d in detections])
                scores = np.array([d['confidence'] for d in detections])
                indices = self._nms(boxes, scores, 0.5)
                detections = [detections[i] for i in indices]
            
            return detections
        return []

    def _nms(self, boxes, scores, iou_threshold):
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
