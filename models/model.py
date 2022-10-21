from .LNG_Transformer import LNGTransformer
from .Swin_Transformer import SwinTransformer
from .Resnet import ResNet, BasicBlock, Bottleneck
from .Vision_Transformer import VisionTransformer


# LSG_Transformer
def LNG_T(num_classes=100, stages=[1, 1, 2, 1], dim=96, num_heads=[3, 6, 12, 24],**kwargs):
    return LNGTransformer(in_chans=3, dims=[dim, dim*2, dim*4, dim*8], patch_size=4, window_size=7, stages = stages, num_heads = num_heads, num_classes=num_classes)

def LNG_S(num_classes=100, stages=[1, 1, 2, 1], dim=128, num_heads=[4, 8, 16, 32],**kwargs):
    return LNGTransformer(in_chans=3, dims=[dim, dim*2, dim*4, dim*8], patch_size=4, window_size=7, stages = stages, num_heads = num_heads, num_classes=num_classes)

def LNG_B(num_classes=100, stages=[1, 1, 6, 1], dim=128, num_heads=[4, 8, 16, 32],**kwargs):
    return LNGTransformer(in_chans=3, dims=[dim, dim*2, dim*4, dim*8], patch_size=4, window_size=7, stages = stages, num_heads = num_heads, num_classes=num_classes)

# resnet

def resnet_50(num_classes=1000, include_top=True, **kwargs):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet_101(num_classes=1000, include_top=True, **kwargs):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

# vision transformer
def ViT_B(num_classes: int = 1000, **kwargs):
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=None, num_classes=num_classes)
    return model

def ViT_L(num_classes: int = 1000, **kwargs):
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, representation_size=None, num_classes=num_classes)
    return model



def Swin_T(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3, patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), num_classes=num_classes, **kwargs)
    return model

def Swin_S(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3, patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), num_classes=num_classes, **kwargs)
    return model

def Swin_B(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), num_classes=num_classes, **kwargs)
    return model