import torchvision
from torchvision.models.detection.ssdlite import SSDLiteHead
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation

def get_model(num_classes, pretrained=True, image_size=320):
    """
    Создает модель SSDLite320 MobileNetV3 Large.
    Если pretrained=True, загружает веса COCO и заменяет голову.
    Если pretrained=False, создает модель с архитектурой MobileNetV3-Large (DEFAULT config)
    и затем также заменяет ее стандартную голову на голову для num_classes.
    Это гарантирует идентичность backbone при загрузке кастомных весов.
    """
    print_prefix_original_head = ""
    if pretrained:
        weights_source = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights_source)
        print_prefix_original_head = "COCO model.head"
    else:
        weights_for_arch_consistency = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights_for_arch_consistency)
        print_prefix_original_head = "DEFAULT arch model.head"

    if model.head is None or not hasattr(model.head, 'classification_head') or \
       not hasattr(model.head.classification_head, 'module_list'):
        raise ValueError(f"Оригинальная голова модели ({print_prefix_original_head}) или ее компоненты не инициализированы.")

    original_cls_head = model.head.classification_head
    in_channels_for_new_head = []
    for seq_module in original_cls_head.module_list:
        if not isinstance(seq_module[0], Conv2dNormActivation):
            raise ValueError(
                f"Ожидался Conv2dNormActivation как первый блок в {print_prefix_original_head}, но получен {type(seq_module[0])}"
            )
        conv_norm_act_block = seq_module[0]
        actual_conv_layer = None
        if hasattr(conv_norm_act_block, '0') and isinstance(conv_norm_act_block[0], nn.Conv2d):
             actual_conv_layer = conv_norm_act_block[0]
        else: 
            for child in conv_norm_act_block.modules():
                if isinstance(child, nn.Conv2d):
                    actual_conv_layer = child
                    break
        if actual_conv_layer is None:
            raise ValueError(f"Не удалось найти nn.Conv2d внутри Conv2dNormActivation блока ({print_prefix_original_head}): {conv_norm_act_block}")
        in_channels_for_new_head.append(actual_conv_layer.in_channels)
    
    num_anchors = []
    if hasattr(model.anchor_generator, 'num_anchors_per_location') and \
       callable(model.anchor_generator.num_anchors_per_location):
        num_anchors = model.anchor_generator.num_anchors_per_location()
        if not isinstance(num_anchors, list) or not all(isinstance(na, int) for na in num_anchors):
             num_anchors = [] 
        elif len(num_anchors) != len(in_channels_for_new_head):
            num_anchors = [] 
    
    if not num_anchors: 
        num_anchors = [6] * len(in_channels_for_new_head)
            
    norm_layer_for_head = nn.BatchNorm2d
    new_head = SSDLiteHead(in_channels=in_channels_for_new_head, 
                           num_anchors=num_anchors, 
                           num_classes=num_classes,
                           norm_layer=norm_layer_for_head)
    model.head = new_head
    return model 