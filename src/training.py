from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks

from .dataset import Brain_MRI_Dataset
from .losses import pixelwise_accuracy, mean_iou

from tqdm.notebook import tqdm, trange


def train_segmentation(model: nn.Module,
                       dataset: Brain_MRI_Dataset,
                       criterion: nn,
                       optimizer: torch.optim,
                       num_classes: int = 1,
                       batch_size: int = 10,
                       num_epochs: int = 10,
                       start_epoch: int = 0,
                       save_path: str | None = None,
                       current_pixelwise_accuracy: float = .0,
                       device: str | torch.device = "cpu"):
    """
    Функция обучения нейросети

    :param model: модель нейросети
    :param dataset: датасет для обучения
    :param criterion: функция потерь
    :param optimizer: оптимизатор
    :param num_classes: количество классов (необходимо для "Mean IoU")
    :param batch_size: размер батча
    :param num_epochs: количество эпох
    :param start_epoch: начальная эпоха
    :param save_path: путь для сохранения модели
    :param current_pixelwise_accuracy: текущая точность метрики "Pixelwise Acc"
    :param device: устройство, на котором будут происходить все вычисления
    """
    model.to(device)

    # Загрузчик данных для обучения и валидации
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))

    # Контроллеры скорости обучения
    scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=list(range(start_epoch, start_epoch + num_epochs, 10)),
                                                gamma=0.5)
    # Режим обучения модели
    model.train()
    for epoch in trange(start_epoch, start_epoch + num_epochs):
        print(f"Эпоха {epoch + 1}/{start_epoch + num_epochs}")
        # Переменная для подсчета точности и потерь
        epoch_train_loss = 0

        # Переменные для подсчета метрик сегментации "pixelwise Acc" и "Mean IoU"
        pixelwise_arr, iou = 0, 0
        # Процесс обучения
        for images, masks in tqdm(dataloader):
            # Обнуление градиента
            optimizer.zero_grad()

            images = images.to(device)
            masks = masks.to(device)

            # Предсказание
            outputs = model(images)['out']

            # Вызов функции потерь
            loss = criterion(outputs, masks)

            # Дифференцирование с учетом параметров
            loss.backward()
            # Шаг оптимизации
            optimizer.step()

            # Суммирование потерь
            epoch_train_loss += loss.item()

            for predicted, target in zip(outputs, masks):
                pixelwise_arr += pixelwise_accuracy(predicted, target)
                iou += mean_iou(predicted, target, num_classes)

        # Уменьшение скорости обучения
        scheduler1.step()
        scheduler2.step()

        # Вывод потерь и текущей скорости обучения
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        pixelwise = pixelwise_arr / len(dataset)
        mean_iou_value = iou / len(dataset)
        print(
            f"loss: {epoch_train_loss / len(dataset):.4f}\n"
            f"pixelwise Acc: {pixelwise:.4f}\n"
            f"Mean IoU: {mean_iou_value:.4f}\n"
            f"Скорость обучения: {lr}"
        )
        if current_pixelwise_accuracy < pixelwise:
            current_pixelwise_accuracy = pixelwise
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': criterion.state_dict(),
                'pixelwise': pixelwise,
                'Mean IoU': mean_iou_value
            }, save_path)
            print(f"Модель сохранена")


def test_segmentation(model: nn.Module,
                      image_preprocess: transforms.Compose,
                      image_path: str | Path,
                      proba_threshold: float = 0.7,
                      alpha: float = 0.9,
                      device: str | torch.device = "cpu"):
    """
    Функция тестирования нейросети

    :param model: модель нейросети
    :param image_preprocess: датасет для обучения
    :param image_path: путь до изображения
    :param proba_threshold: степень уверенности
    :param alpha: урвоень непрозрачности при наложении маски
    :param device: устройство, на котором будут происходить все вычисления

    :return: PIL.Image(оригинальное изображение), PIL.Image(предсказание), PIL.Image(наложенная маска)
    """
    model.eval()
    model.to(device)
                          
    # Загрузка изображения
    image = transforms.ToTensor()(Image.open(image_path).convert('RGB'))
    # Создание батча
    batch = transforms.Resize((256, 256))(image_preprocess(image)).unsqueeze(0).to(device)
    # Прогноз
    prediction = model(batch)["out"]
    # Булева маска
    bool_masks = prediction[0] > proba_threshold
    bool_masks = bool_masks.squeeze(1).to(device)
    # Изменение размера изображения
    image_size = bool_masks.shape[1:]
    mask = transforms.functional.resize(image, image_size).to(device)
    # Тензор наложенной маски
    draw_mask = draw_segmentation_masks(mask.type(torch.uint8), bool_masks, alpha=alpha)

    return to_pil_image(image), to_pil_image(prediction[0]), to_pil_image(draw_mask)
