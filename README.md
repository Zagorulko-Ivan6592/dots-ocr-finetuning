# dots.ocr: Fine-tuning и Inference

Репозиторий для дообучения и использования модели [dots.ocr](https://github.com/rednote-hilab/dots.ocr) для OCR и парсинга документов на русском языке.

## Что сделано

### 1. Подготовка датасета
- Найден и подготовлен датасет PDF документов на русском языке
- Датасет содержит разнообразные документы: таблицы, текст, формулы, изображения

### 2. Дообучение модели dots.ocr
- Дообучение базовой модели `rednote-hilab/dots.ocr` с использованием LoRA (Low-Rank Adaptation)
- Обучение на русскоязычных PDF документах для улучшения качества OCR
- Объединение LoRA адаптеров с базовой моделью для получения единой дообученной модели
- Результат: дообученная модель готова к использованию

### 3. Оценка качества
- Проведена оценка базовой и дообученной модели на различных бенчмарках:
  - **OCRBench** - стандартный бенчмарк для OCR
  - **OCRBench v2** - обновленная версия OCRBench
  - **MWS-Vision-Bench** - русскоязычный бенчмарк для мультимодальных LLM
- Результаты показывают улучшение качества распознавания после дообучения

### 4. Публикация модели
- Дообученная модель опубликована на Hugging Face Hub
- **Модель доступна по адресу:** [normoldaki31/dots-ocr-finetuned](https://huggingface.co/normoldaki31/dots-ocr-finetuned)
- Модель можно использовать напрямую из Hugging Face Hub или скачать локально

### 5. Ноутбуки для инференса
- **`colab_inference_default.ipynb`** - инференс базовой модели dots.ocr
  - Обработка PDF документов
  - Извлечение текста, таблиц, формул
  - Восстановление Markdown и PDF из результатов
  
- **`colab_inference_merged.ipynb`** - инференс дообученной модели
  - Все возможности базовой модели
  - Улучшенная точность на специфических документах после дообучения

### Сравнение базовой и дообученной модели

![Benchmark Comparison](./benchmark_comparison.png)

**Результаты на различных бенчмарках:**

| Бенчмарк | Базовая модель | Дообученная модель | Улучшение |
|----------|----------------|-------------------|-----------|
| OCRBench | 0.560 | 0.577 | +3.04% |
| OCRBench v2 | 0.190 | 0.200 | +5.26% |
| MWS-Vision-Bench | 0.730 | 0.810 | +10.96% |



## Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Запуск vLLM сервера

vLLM позволяет ускорить инференс в 2-5 раз по сравнению с HuggingFace Transformers.

#### Для базовой модели:

```bash
vllm serve rednote-hilab/dots.ocr \
  --trust-remote-code \
  --async-scheduling \
  --gpu-memory-utilization 0.5 \
  --port 8000
```

#### Для дообученной модели:

```bash
vllm serve normoldaki31/dots-ocr-finetuned \
  --trust-remote-code \
  --async-scheduling \
  --gpu-memory-utilization 0.5 \
  --port 8001
```

**Примечание:** Модели автоматически скачаются из Hugging Face Hub при первом запуске.

#### Проверка работы сервера:

```bash
# Проверка здоровья сервера
curl http://localhost:8000/health

# Список доступных моделей
curl http://localhost:8000/v1/models
```

#### Запуск в фоне:

```bash
nohup vllm serve normoldaki31/dots-ocr-finetuned \
  --trust-remote-code \
  --async-scheduling \
  --gpu-memory-utilization 0.5 \
  --port 8001 > vllm.log 2>&1 &

# Просмотр логов
tail -f vllm.log
```

### Использование ноутбуков для инференса

1. Откройте ноутбук из папки `inference/`:
   - `colab_inference_default.ipynb` - для базовой модели
   - `colab_inference_merged.ipynb` - для дообученной модели

2. Установите путь к модели:
   ```python
   # Для базовой модели
   MODEL_PATH = "rednote-hilab/dots.ocr"
   
   # Для дообученной модели
   MODEL_PATH = "normoldaki31/dots-ocr-finetuned"
   ```

3. Если используете vLLM, установите:
   ```python
   USE_VLLM = True
   VLLM_SERVER_IP = "localhost"
   VLLM_SERVER_PORT = 8000  # или 8001 для дообученной модели
   ```

4. Запустите ячейки по порядку

## Оценка через бенчмарки

Для оценки производительности моделей используется форк [lmms-eval](https://github.com/Zagorulko-Ivan6592/lmms-eval) с поддержкой дополнительных бенчмарков.

### Установка lmms-eval

```bash
git clone https://github.com/Zagorulko-Ivan6592/lmms-eval
cd lmms-eval
pip install -e .
```

### Запуск бенчмарков

**1. Запустите vLLM сервер** (см. раздел "Запуск vLLM сервера" выше)

**2. Запустите оценку:**

Для базовой модели (порт 8000):
```bash
python -m lmms_eval \
  --model openai_compatible \
  --model_args "model_version=rednote-hilab/dots.ocr,base_url=http://localhost:8000/v1,api_key=dummy-key" \
  --tasks ocrbench \
  --batch_size 1 \
  --log_samples \
  --output_path ./results/dots_ocr_base_ocrbench
```

Для дообученной модели (порт 8001):
```bash
python -m lmms_eval \
  --model openai_compatible \
  --model_args "model_version=normoldaki31/dots-ocr-finetuned,base_url=http://localhost:8001/v1,api_key=dummy-key" \
  --tasks ocrbench \
  --batch_size 1 \
  --log_samples \
  --output_path ./results/dots_ocr_finetuned_ocrbench
```

### Доступные бенчмарки

- `ocrbench` - OCRBench (стандартный бенчмарк для OCR)
- `ocrbench_v2` - OCRBench v2 (обновленная версия)
- `mws_vision_bench_validation` - MWS-Vision-Bench (русскоязычный бенчмарк)
