import onnxruntime as ort
import numpy as np
import cv2
import os

class DogBreedClassifier:
    def __init__(self, model_path: str = None):
        # Автоматическое определение пути к модели
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            model_path = os.path.join(project_root, "models", "classification", "model_dynamic_quant.onnx")

        self.session = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )
        
        # Получаем информацию о входе и выходе
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, height, width]
        self.output_name = self.session.get_outputs()[0].name
        
        # Размеры изображения для модели
        self.img_height = self.input_shape[2] if len(self.input_shape) > 2 else 256
        self.img_width = self.input_shape[3] if len(self.input_shape) > 3 else 256
        
        # Названия классов
        self.class_names = [
            'Chihuahua',
            'Japanese_spaniel', 
            'Maltese_dog',
            'Pekinese'
        ]
    
    def predict(self, image_bytes: bytes):
        """Предсказание породы собаки"""
        try:
            # Препроцессинг изображения
            input_tensor = self._preprocess(image_bytes)
            
            # Инференс
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            
            # Постпроцессинг
            logits = outputs[0][0]
            class_id = int(np.argmax(logits))
            
            # Softmax для получения вероятностей
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            # Формируем список всех пород
            all_breeds = []
            for i, (name, prob) in enumerate(zip(self.class_names, probabilities)):
                all_breeds.append({
                    'id': i,
                    'name': name,
                    'probability': float(prob),
                    'is_predicted': (i == class_id)
                })
            
            # Сортируем по вероятности
            all_breeds.sort(key=lambda x: x['probability'], reverse=True)
            
            # Результат
            return {
                'predicted_breed': {
                    'id': class_id,
                    'name': self.class_names[class_id],
                    'probability': float(probabilities[class_id]),
                    'confidence_percentage': round(float(probabilities[class_id]) * 100, 2)
                },
                'all_breeds': all_breeds,
                'top_3_breeds': all_breeds[:3],
                'model_info': {
                    'input_size': f"{self.img_width}x{self.img_height}",
                    'model_type': 'quantized'
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Ошибка предсказания: {e}")
    
    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        """Препроцессинг изображения для модели"""
        # Декодируем изображение
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Не удалось декодировать изображение")
        
        # Конвертируем BGR в RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Изменяем размер
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        # Нормализация [0, 255] -> [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Транспонирование (H, W, C) -> (C, H, W)
        img = img.transpose(2, 0, 1)
        
        # Добавляем батч-размерность
        img = np.expand_dims(img, axis=0)
        
        return img.astype(np.float32)
