import torch
from torch import nn


def pixelwise_accuracy(predicted, target):
    """
    Функция подсчета метрики "pixelwise Acc"
    """
    # Округление предсказаний
    predicted = torch.round(predicted)

    # Приведение к типу LongTensor
    predicted = predicted.type(torch.LongTensor)
    target = target.type(torch.LongTensor)

    # Подсчет пиксельной точности
    correct_pixels = torch.sum(predicted == target)
    total_pixels = predicted.numel()
    accuracy = correct_pixels.item() / total_pixels

    return accuracy


def mean_iou(predicted, target, num_classes):
    """
    Функция подсчета метрики "Mean IoU"
    """
    # Округление предсказаний
    predicted = torch.round(predicted)

    # Приведение к типу LongTensor
    predicted = predicted.type(torch.LongTensor)
    target = target.type(torch.LongTensor)

    iou_values = []
    for cls in range(num_classes):
        # Создание бинарных масок для класса
        predicted_mask = (predicted == cls)
        target_mask = (target == cls)

        intersection = torch.logical_and(predicted_mask, target_mask).sum()
        union = torch.logical_or(predicted_mask, target_mask).sum()

        # Исключение деления на 0
        iou = torch.where(union != 0, intersection.float() / union.float(), torch.tensor(0.0))
        iou_values.append(iou.item())

    mean_iou = sum(iou_values) / num_classes

    return mean_iou


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predicted, target):
        smooth = 1e-5
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target)
        loss = 1 - (2 * intersection + smooth) / (union + smooth)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predicted, target):
        eps = 1e-7
        predicted = predicted.clamp(eps, 1. - eps)
        focal_weights = torch.pow(1 - predicted, self.gamma)
        loss = -self.alpha * target * torch.log(predicted) * focal_weights - (1 - self.alpha) * (
                    1 - target) * torch.log(1 - predicted) * focal_weights
        return loss.mean()


class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, predicted, target):
        eps = 1e-7
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target) - intersection
        loss = 1 - (intersection + eps) / (union + eps)
        return loss
