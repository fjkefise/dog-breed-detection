# Dog Breed Detection

Веб-сервис для детекции и классификации пород собак на изображениях.

## Быстрый старт

```bash
# Собрать Docker образ
docker build -t dog_breed_project .

# Запустить (CPU)
docker run -p 8077:8077 dog_breed_project

# Запустить (GPU, если есть NVIDIA)
docker run --gpus all -p 8077:8077 dog_breed_project
