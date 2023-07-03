# Название доступных моделей для сегментации
segmentation_models_names = [
    'deeplabv3_mobilenet_v3_large',
    'deeplabv3_resnet50',
    'deeplabv3_resnet101',
    'fcn_resnet50',
    'fcn_resnet101',
    'lraspp_mobilenet_v3_large'
]

# Ключи на последний слой модели
last_layer_replace_dict = {
    'deeplabv3_mobilenet_v3_large': ['classifier.4', 'aux_classifier.4'],
    'deeplabv3_resnet50': ['classifier.4', 'aux_classifier.4'],
    'deeplabv3_resnet101': ['classifier.4', 'aux_classifier.4'],
    'fcn_resnet50': ['classifier.4', 'aux_classifier.4'],
    'fcn_resnet101': ['classifier.4', 'aux_classifier.4'],
    'lraspp_mobilenet_v3_large': ['classifier.low_classifier', 'classifier.high_classifier']
}

# Название модели при импорте
get_models_dict = {
    'deeplabv3_mobilenet_v3_large': 'deeplabv3_mobilenet_v3_large',
    'deeplabv3_resnet50': 'deeplabv3_resnet50',
    'deeplabv3_resnet101': 'deeplabv3_resnet101',
    'fcn_resnet50': 'fcn_resnet50',
    'fcn_resnet101': 'fcn_resnet101',
    'lraspp_mobilenet_v3_large': 'lraspp_mobilenet_v3_large',
}

# Название весов при импорте
get_weights_dict = {
    'deeplabv3_mobilenet_v3_large_weights': 'DeepLabV3_MobileNet_V3_Large_Weights',
    'deeplabv3_resnet50_weights': 'DeepLabV3_ResNet50_Weights',
    'deeplabv3_resnet101_weights': 'DeepLabV3_ResNet101_Weights',
    'fcn_resnet50_weights': 'FCN_ResNet50_Weights',
    'fcn_resnet101_weights': 'FCN_ResNet101_Weights',
    'lraspp_mobilenet_v3_large_weights': 'LRASPP_MobileNet_V3_Large_Weights',
}
