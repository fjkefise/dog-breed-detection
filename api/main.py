# api/main.py
from fastapi import FastAPI, File, UploadFile
import uvicorn
from api.models.classifier import DogBreedClassifier
from api.models.detector import DogBreedDetector

app = FastAPI(
    title="Dog Breed API",
    description="Простой API для работы с собаками",
    version="1.0.0"
)

classifier = DogBreedClassifier()
detector = DogBreedDetector()

@app.get("/")
def root():
    return {"message": "Dog Breed API"}

@app.post("/classify")
async def classify_dog(image: UploadFile = File(...)):
    contents = await image.read()
    result = classifier.predict(contents)
    return result

@app.post("/detect")
async def detect_dogs(image: UploadFile = File(...), confidence: float = 0.5):
    contents = await image.read()
    detections = detector.detect(contents, conf_threshold=confidence)
    return {"detections": detections, "count": len(detections)}

@app.post("/analyze")
async def analyze_image(image: UploadFile = File(...)):
    contents = await image.read()
    
    # Детекция
    detections = detector.detect(contents)
    
    # Классификация каждой детекции (просто для примера)
    analysis = []
    for det in detections:
        result = classifier.predict(contents)
        analysis.append({
            **det,
            "classification": result['predicted_breed']
        })
    
    return {"analysis": analysis, "total_dogs": len(analysis)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
