from .ENet import ENet
from .RAUNet import RAUNet
from .resnet18 import Resnet18
from .mobilenetv2 import MobileNetV2
from .dinov3_vision_transformer import build_dinov3_base_primus_multiscale


def get_model(model_name: str, channels: int, **kwargs):
    assert model_name.lower() in ['enet', 'raunet', 'dinov3_vit']
    if model_name.lower() == 'raunet':
        model = RAUNet(num_classes=channels)
    elif model_name.lower() == 'enet':
        model = ENet(num_classes=channels)
    elif model_name.lower() == 'dinov3_vit':
        model = build_dinov3_base_primus_multiscale(
            num_classes=channels,
            checkpoint_path = kwargs["checkpoint_path"],
        )
    elif model_name.lower() == 'resnet':
        model = Resnet18(num_classes=channels)
    elif model_name.lower() == 'mobilenetv2':
        model = MobileNetV2(num_classes=channels)
    return model