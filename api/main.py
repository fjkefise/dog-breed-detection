from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.models.classifier import DogBreedClassifier
from api.models.detector import DogBreedDetector

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Загрузка моделей...")
    app.state.classifier = DogBreedClassifier()
    app.state.detector = DogBreedDetector()
    print("Модели загружены успешно")
    yield
    print("Очистка ресурсов...")

app = FastAPI(
    title="Dog Breed API",
    description="API для классификации и детекции пород собак",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "index.html"
    )
        
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)

@app.post("/classify")
async def classify_dog(image: UploadFile = File(...)):
    if not image.content_type.startswith('image/'):
        raise HTTPException(400, "Нужно загрузить изображение")
    
    contents = await image.read()
    result = app.state.classifier.predict(contents)
    return result

@app.post("/detect")
async def detect_dogs(image: UploadFile = File(...), confidence: float = 0.5):
    if confidence < 0.1 or confidence > 1.0:
        raise HTTPException(400, "Confidence должен быть между 0.1 и 1.0")
    
    if not image.content_type.startswith('image/'):
        raise HTTPException(400, "Нужно загрузить изображение")
    
    contents = await image.read()
    detections = app.state.detector.detect(contents, conf_threshold=confidence)
    return {"detections": detections, "count": len(detections)}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
