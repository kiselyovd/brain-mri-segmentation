import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import segmentation

from .config import *


def valid_model_name(model_name: str):
    """
    Функция проверки поддержки модели

    :param model_name: название модели
    """
    assert model_name in segmentation_models_names, f"'{model_name}'не поддерживается. Доступные модели: {segmentation_models_names}"


def get_image_segmentation_model(name: str,
                                 pretrained: bool = True,
                                 freeze_weight: bool = False,
                                 num_classes: int = 1):
    """
    Функция получения сконфигурированного сегментатора изображений

    :param name: название модели
    :param pretrained: загружать веса Imagenet1K
    :param freeze_weight: замораживать все слои, кроме последнего
    :param num_classes: количество выходных классов

    :return: tuple(модель, трансформер изображения)
    """
    valid_model_name(name)

    models_package = segmentation.__dict__

    last_layer_name = last_layer_replace_dict[name]
    import_name = get_models_dict[name]
    weight_name = get_weights_dict[f"{import_name}_weights"]

    weights = models_package[weight_name].DEFAULT

    model = models_package[import_name](weights=weights if pretrained else None)

    if freeze_weight:
        for param in model.parameters():
            param.requiresGrad = False

    if name == "lraspp_mobilenet_v3_large":
        in_channels_low = model.classifier.low_classifier.in_channels
        model.classifier.low_classifier = nn.Conv2d(in_channels=in_channels_low,
                                                    out_channels=num_classes,
                                                    kernel_size=(1, 1),
                                                    stride=(1, 1))

        in_channels_high = model.classifier.high_classifier.in_channels
        model.classifier.high_classifier = nn.Conv2d(in_channels=in_channels_high,
                                                     out_channels=num_classes,
                                                     kernel_size=(1, 1),
                                                     stride=(1, 1))

        return model, weights.transforms()

    if pretrained:
        for layer in last_layer_name:
            if "." not in layer:
                in_features = model._modules[layer].in_features
                model._modules[layer] = nn.Linear(in_features=in_features, out_features=num_classes)

            else:
                last_layer_arr = layer.split(".")

                try:
                    last_layer_arr[1] = int(last_layer_arr[1])
                except ValueError:
                    last_layer_arr[1] = 0

                in_channels = model._modules[last_layer_arr[0]][last_layer_arr[1]].in_channels
                model._modules[last_layer_arr[0]][last_layer_arr[1]] = nn.Conv2d(in_channels=in_channels,
                                                                                 out_channels=num_classes,
                                                                                 kernel_size=(1, 1),
                                                                                 stride=(1, 1))
    else:
        last_layer_name = last_layer_name[0]

        if "." not in last_layer_name:
            in_features = model._modules[last_layer_name].in_features
            model._modules[last_layer_name] = nn.Linear(in_features=in_features, out_features=num_classes)

        else:
            last_layer_arr = last_layer_name.split(".")

            try:
                last_layer_arr[1] = int(last_layer_arr[1])
            except ValueError:
                last_layer_arr[1] = 0

            in_channels = model._modules[last_layer_arr[0]][last_layer_arr[1]].in_channels
            model._modules[last_layer_arr[0]][last_layer_arr[1]] = nn.Conv2d(in_channels=in_channels,
                                                                             out_channels=num_classes,
                                                                             kernel_size=(1, 1),
                                                                             stride=(1, 1))
    return model, weights.transforms()
