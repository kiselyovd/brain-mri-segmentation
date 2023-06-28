from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


class Brain_MRI_Dataset(torch.utils.data.Dataset):
    """Класс загрузки и предобработки Brain MRI датасета"""

    def __init__(self,
                 path_array: list[str | Path],
                 image_preprocess: transforms.Compose,
                 augmented: bool = False,
                 device: str | torch.device = "cpu"
                 ):
        """
        Инициализация класса датасета

        :param X_path_array: список путей до изображений
        :param y_path: путь до папки с масками
        :param image_preprocess: функция предобработки изображения
        :param augmented: аугментирование данных
        """
        self.path_array = path_array
        self.image_preprocess = image_preprocess
        self.augmented = augmented  # Не реализовано
        self.device = device

    def __len__(self):
        """Метод возвращения длины датасета"""
        return len(self.path_array)

    def __getitem__(self, idx: int):
        """
        Метод возвращения элемента по индексу

        :param idx: индекс списка ссылок на архив

        :return: tensor(изображение), tensor(маска)
        """
        # Путь до элемента
        image_path = self.path_array[idx]

        # Чтение изображения и маски
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(image_path.parent / f"{image_path.stem}_mask.tif").convert('L')

        # Предобработка изображения
        image = transforms.Resize((256, 256))(self.image_preprocess(image))
        # Предобработка маски
        mask = transforms.ToTensor()(transforms.functional.resize(mask, (256, 256)))

        return image.to(self.device), mask.to(self.device)
