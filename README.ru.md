# brain-mri-segmentation

[![CI](https://img.shields.io/github/actions/workflow/status/kiselyovd/brain-mri-segmentation/ci.yml?branch=main&label=CI&style=for-the-badge&logo=github)](https://github.com/kiselyovd/brain-mri-segmentation/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-526CFE?style=for-the-badge&logo=materialformkdocs&logoColor=white)](https://kiselyovd.github.io/brain-mri-segmentation/)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kiselyovd/brain-mri-segmentation/badges/coverage.json&style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/kiselyovd/brain-mri-segmentation/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![HF Model](https://img.shields.io/badge/🤗%20HF%20Hub-model-FFD21E?style=for-the-badge)](https://huggingface.co/kiselyovd/brain-mri-segmentation)

Бинарная сегментация опухолей головного мозга на МРТ-срезах — дообученный **SegFormer-B2** в роли основной модели и самописный **U-Net** как воспроизводимый baseline, обученные на датасете LGG (TCGA) Матеуша Буды с разбиением по пациентам, исключающим утечку данных.

**English:** [README.md](README.md) · **Docs:** [kiselyovd.github.io/brain-mri-segmentation](https://kiselyovd.github.io/brain-mri-segmentation/) · **Модель:** [kiselyovd/brain-mri-segmentation](https://huggingface.co/kiselyovd/brain-mri-segmentation)

## Датасет

[LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) от Матеуша Буды на Kaggle — 110 пациентов, 3 929 парных FLAIR-срезов с бинарными масками опухолей из The Cancer Genome Atlas (TCGA). `src/brain_mri_segmentation/data/prepare.py` выполняет **разбиение 80/10/10 на уровне пациентов** (88/11/11 пациентов), так что один и тот же пациент не попадает в несколько партиций.

Итоговое количество срезов: **3 133 train / 409 val / 387 test**.

## Результаты

Метрики на тестовом сплите после полного обучения (проставить из `reports/metrics.json`):

| Модель | Dice | IoU | Pixel Accuracy |
|---|---|---|---|
| **SegFormer-B2** (основная) | **65.5%** | **66.2%** | **99.73%** |
| U-Net 4-уровневый baseline | 51.9% | 57.7% | 99.66% |

Полный отчёт по срезам лежит в `reports/metrics.json` после запуска evaluation.

## Быстрый старт

```bash
# 1. Установка
uv sync --all-groups

# 2. Скопировать датасет из Kaggle в data/raw/ (один раз)
bash scripts/sync_data.sh /path/to/lgg-mri-segmentation

# 3. Сформировать обработанные сплиты
uv run python -m brain_mri_segmentation.data.prepare --raw data/raw --out data/processed

# 4. Обучение (основная модель, на GPU)
make train

# 5. Оценка на test-сплите
make evaluate

# 6. Локальный сервинг
make serve
# или
docker compose up api
```

## Полные команды обучения

**Основная — SegFormer-B2:**

```bash
uv run python -m brain_mri_segmentation.training.train experiment=sota
```

**Baseline — U-Net (4 уровня, 32→256 каналов):**

```bash
uv run python -m brain_mri_segmentation.training.train \
  model=baseline \
  trainer.max_epochs=30 \
  trainer.output_dir=artifacts/baseline
```

Каждый запуск логируется в MLflow в `./mlruns/`; просмотр: `mlflow ui --backend-store-uri ./mlruns`.

## Инференс

```python
from huggingface_hub import snapshot_download

from brain_mri_segmentation.inference.predict import load_model, predict

weights_dir = snapshot_download("kiselyovd/brain-mri-segmentation")
model = load_model(f"{weights_dir}/best.ckpt")
result = predict(model, "slice.tif")
print(f"Mask: {len(result['mask'])}×{len(result['mask'][0])}")
```

`result["mask"]` — двумерный бинарный массив (H × W), совмещённый с входным срезом.

## Сервинг

```bash
docker compose up api
curl -X POST -F "file=@slice.tif" http://localhost:8000/segment
```

Эндпоинты:

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/health` | Liveness-проба |
| `POST` | `/segment` | multipart TIFF/PNG → JSON-маска |
| `GET` | `/metrics` | Prometheus-метрики |

В каждом ответе есть заголовок `X-Request-ID` для корреляции логов.

## Структура проекта

```
src/brain_mri_segmentation/
├── data/           # MRIDataModule, MRIDataset, prepare.py (разбиение по пациентам)
├── models/         # factory.py, lightning_module.py, unet.py
├── training/       # Hydra-входная точка
├── evaluation/     # отчёт Dice / IoU / pixel accuracy
├── inference/      # load_model + predict
├── serving/        # FastAPI-приложение
└── utils/          # логирование, сиды, HF Hub
configs/            # Hydra-конфиги (data / model / trainer / experiment)
data/
├── raw/            # оригинальная выгрузка с Kaggle (изображения + маски по пациентам)
└── processed/      # сплиты train / val / test
docs/               # исходники MkDocs
tests/              # pytest
```

## Назначение

Исследовательский и образовательный проект. **Не является медицинским изделием.** Предсказания модели запрещается использовать для принятия клинических решений.

Известные ограничения и режимы отказа — в [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Цитирование

```bibtex
@software{kiselyov2026brainmri,
  author  = {Kiselyov, Daniil},
  title   = {brain-mri-segmentation: SegFormer-B2 brain tumor MRI segmentation},
  year    = {2026},
  url     = {https://github.com/kiselyovd/brain-mri-segmentation},
  version = {v0.1.0}
}
```

## Лицензия

MIT — см. [LICENSE](LICENSE).
