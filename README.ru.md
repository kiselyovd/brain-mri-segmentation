# Сегментация LGG MRI

[Модель и веса](https://huggingface.co/kiselyovd/brain-mri-segmentation) · [Полная документация / English](README.md) · [Контракт оценки](docs/EVALUATION.md)

SegFormer-B2 выделяет область FLAIR-аномалии на 2D MRI-срезе. Исходный датасет найден и скачан с Kaggle: 110 пациентов, 3929 пар изображение–маска. По коду подготовки восстановлен split **3133/409/387 срезов**, **88/11/11 пациентов**. Состав и хеши файлов сохранены в [manifest](reports/provenance/lgg-split-manifest.json).

Исправлен расчёт foreground-метрик. Новый replay опубликованных весов на 387 восстановленных test-срезах приведён в [таблице результатов](README.md#current-artifact-and-evaluation), которую CI генерирует из отчёта. Это проверенный запуск на восстановленном split. Исторический состав training-набора checkpoint документально не подтверждён, поэтому результат не называется независимым held-out benchmark.

[Полный отчёт](reports/replay-segformer-2026-10-15.json) содержит результаты по каждому пациенту и срезу, TP/FP/FN/TN, bootstrap-интервалы и идентификаторы модели, данных и кода. [Массивы предсказаний и масок](reports/replay-segformer-2026-10-15.npz) позволяют пересчитать показатели.

HF processor исправлен: 256×256,RGB/255, без ImageNet-нормализации и уменьшения меток. [Пример запуска](README.md#run-the-published-model) закреплён на конкретной ревизии; для точного tensor resize используется float32 RGB-массив и torchvision backend.

Старые 65,5/66,2 и 51,9/57,7 сохранены отдельно как legacy: evaluator передавал index-маски в one-hot контракт. Старый checkpoint собственного U-Net отсутствует; чужие веса автора датасета его не заменяют. Новый запуск baseline возможен [по инструкции](README.md#training-and-serving), но не считается восстановлением исторического эксперимента.

Код — MIT; данные и производные source manifests — CC-BY-NC-SA-4.0; upstream SegFormer имеет отдельные условия NVIDIA. [Лицензии](docs/LICENSES.md) · [Ограничения](docs/LIMITATIONS.md) · [Цитирование](CITATION.cff). Клиническая валидация не проводилась.
