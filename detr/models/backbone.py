# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import segmentation_models_pytorch as smp
from util.misc import NestedTensor, is_main_process
import matplotlib.pyplot as plt
from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

#
# def replace_bn_with_frozen_bn(module):
#     """
#     递归地将 module 中所有 nn.BatchNorm2d 替换为 FrozenBatchNorm2d，
#     并将源 BN 的参数拷贝到新的 FrozenBatchNorm2d。
#     """
#     for name, child in list(module.named_children()):
#         # 如果发现了一个普通的 BatchNorm2d，就替换
#         if isinstance(child, nn.BatchNorm2d):
#             num_features = child.num_features
#             # 用自定义的 FrozenBatchNorm2d 替换
#             frozen_bn = FrozenBatchNorm2d(num_features=num_features, eps=child.eps)
#
#             # 将原 BN 参数复制过去（如果你想“冻结”，也可以选择不复制 weight/bias，而只复制 running_mean/var）
#             with torch.no_grad():
#                 # 如果原 BN 有 weight / bias，就复制
#                 if child.affine:
#                     frozen_bn.weight.copy_(child.weight.data)
#                     frozen_bn.bias.copy_(child.bias.data)
#
#                 # 复制运行时的 mean, var
#                 frozen_bn.running_mean.copy_(child.running_mean.data)
#                 frozen_bn.running_var.copy_(child.running_var.data)
#
#             # 然后把当前模块的这个子模块替换掉
#             setattr(module, name, frozen_bn)
#         else:
#             # 如果不是 BatchNorm2d，就继续递归
#             replace_bn_with_frozen_bn(child)

def replace_bn_with_frozen_bn(module):
    """
    递归替换 module 中的所有 nn.BatchNorm2d 为 FrozenBatchNorm2d。
    你可以根据需求来保留或修改。
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            # 用 FrozenBatchNorm2d 的通道数替换
            num_features = child.num_features
            frozen = FrozenBatchNorm2d(num_features)
            setattr(module, name, frozen)
        else:
            replace_bn_with_frozen_bn(child)
    return module

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:  # return_interm_layers = False
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels


    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        # print("is_main_process() is",is_main_process())
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    # for attr in vars(args):
    #     print(f"{attr}: {getattr(args, attr)}")
    # exit()
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0   # train_backbone = 1
    return_interm_layers = args.masks       # return_interm_layers = False
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation) # dilation = False
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

def build_CBAM_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0   # train_backbone = 1
    return_interm_layers = args.masks       # return_interm_layers = False
    backbone = BackCBAMbone(args.backbone, train_backbone, return_interm_layers, args.dilation) # dilation = False
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

def build_CBAM_backbone_mask(args, window_pos):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0   # train_backbone = 1
    return_interm_layers = args.masks       # return_interm_layers = False
    # backbone = BackCBAMbone(args.backbone, train_backbone, return_interm_layers, args.dilation) # dilation = False
    # backbone = BackCBAMbonewithMask(args.backbone, train_backbone, return_interm_layers, args.dilation, ) # dilation = False
    # window_pos = (0, 0, 640, 120)
    backbone = BackCBAMbonewithMask(
        args.backbone,
        train_backbone=train_backbone,
        return_interm_layers=return_interm_layers,
        dilation=args.dilation,
        window_pos=window_pos,
        visualize=False  # 启用可视化
    )

    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

#
# def build_tactile_backbone(args):
#     position_embedding = build_position_encoding(args)
#     train_backbone = args.lr_backbone > 0
#     return_interm_layers = args.masks
#     backbone = Backbone('resnet34', train_backbone, return_interm_layers, args.dilation)
#     model = Joiner(backbone, position_embedding)
#     model.num_channels = backbone.num_channels
#     return model


def build_tactile_backbone(args):

    position_embedding = build_position_encoding(args)
    # net = smp.DeepLabV3Plus(
    net = smp.Unet(
        encoder_name='resnet18',     # 假设 args.backbone 指定类似 'resnet50'、'resnet101' 等resnet34
        encoder_depth=5,
        encoder_weights="imagenet",     # 若不需要预训练权重可改成 None
        in_channels=3,
        classes=512,                    # 输出通道数为 512
        activation=None,               # 若需要激活函数可自行指定
    )

    net = replace_bn_with_frozen_bn(net)

    class TactileBackbone(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.num_channels = 512  # 与 classes=512 对应

        def forward(self, x: torch.Tensor):
            # net(x) 返回 shape = [B, 512, H, W] 的 Tensor
            out = self.net(x)
            # out = F.interpolate(out, size=(240, 320), mode='bilinear', align_corners=False)  512 15 20
            # out = F.interpolate(out, size=(120, 160), mode='bilinear', align_corners=False)
            # out = F.interpolate(out, size=(60, 80), mode='bilinear', align_corners=False)
            # out = F.interpolate(out, size=(30, 40), mode='bilinear', align_corners=False)
            # out = F.interpolate(out, size=(15, 20), mode='bilinear', align_corners=False)
            # conv_downsample = nn.Sequential(
            #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # 卷积操作（保持通道数和大部分特征）
            #     # nn.ReLU(),  # 激活函数（可选）
            #     nn.AdaptiveAvgPool2d((15, 20))  # 自适应平均池化，将空间维度调整到 (15, 20)
            # )
            # # print('input',out.shape)
            # conv_downsample = conv_downsample.cuda()
            # out = conv_downsample(out)
            # print('output',out.shape)
            # 动态计算目标尺寸 (原始尺寸的 1/32)
            input_height, input_width = out.shape[2], out.shape[3]
            output_height, output_width = input_height // 32, input_width // 32
            adaptive_pool = nn.AdaptiveAvgPool2d((output_height, output_width))
            out = adaptive_pool(out)

            return {"out": out}

    backbone = TactileBackbone(net)  # replace_bn_with_frozen_bn(model.encoder)
    # 使用和原始 build_backbone 类似的 Joiner 进行打包
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module (Channel + Spatial Attention)"""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x) * x  # Apply channel attention
        x = self.spatial_attention(x) * x  # Apply spatial attention
        return x


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.global_avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.global_max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2  # To maintain the same spatial size
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Channel-wise max pooling
        out = torch.cat([avg_out, max_out], dim=1)  # Concatenate along the channel dimension
        out = self.conv(out)
        return self.sigmoid(out)


class BackCBAMbone(BackboneBase):
    """ResNet backbone with CBAM and frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)

        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

        # 根据 return_interm_layers 的值设置 CBAM 模块
        if return_interm_layers:
            # 如果返回所有中间层，定义每一层的 CBAM 模块
            self.cbam_layers = nn.ModuleDict({
                "0": CBAM(64),    # layer1 输出通道数
                "1": CBAM(128),   # layer2 输出通道数
                "2": CBAM(256),   # layer3 输出通道数
                "3": CBAM(512),   # layer4 输出通道数
            })
        else:
            # 如果只返回最后一层，定义一个 CBAM 模块
            self.cbam_layers = nn.ModuleDict({
                "0": CBAM(512),   # layer4 输出通道数
            })

    def forward(self, x):
        # 获取 intermediate features
        features = self.body(x)
        enhanced_features = {}
        for layer_name, feature_map in features.items():
            # 根据当前层的名称调用对应的 CBAM 模块
            if layer_name in self.cbam_layers:
                enhanced_features[layer_name] = self.cbam_layers[layer_name](feature_map)
            else:
                enhanced_features[layer_name] = feature_map
        return enhanced_features

# class BackCBAMbonewithMask(BackboneBase):
#     """ResNet backbone with CBAM, frozen BatchNorm, and window-based attention."""
#     def __init__(self, name: str,
#                  train_backbone: bool,
#                  return_interm_layers: bool,
#                  dilation: bool,
#                  window_pos: tuple = None,
#                  visualize: bool = False):
#         """
#         Args:
#             name (str): ResNet model name (e.g., 'resnet18', 'resnet50').
#             train_backbone (bool): Whether to train the backbone.
#             return_interm_layers (bool): Whether to return intermediate layers.
#             dilation (bool): Use dilated convolutions or not.
#             window_pos (tuple): The fixed window position as (x1, y1, x2, y2).
#             visualize (bool): Whether to visualize feature maps and masks.
#         """
#         backbone = getattr(torchvision.models, name)(
#             replace_stride_with_dilation=[False, False, dilation],
#             pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
#
#         num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
#         super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
#
#         # 根据 return_interm_layers 的值设置 CBAM 模块
#         if return_interm_layers:
#             # 如果返回所有中间层，定义每一层的 CBAM 模块
#             self.cbam_layers = nn.ModuleDict({
#                 "0": CBAM(64),    # layer1 输出通道数
#                 "1": CBAM(128),   # layer2 输出通道数
#                 "2": CBAM(256),   # layer3 输出通道数
#                 "3": CBAM(512),   # layer4 输出通道数
#             })
#         else:
#             # 如果只返回最后一层，定义一个 CBAM 模块
#             self.cbam_layers = nn.ModuleDict({
#                 "0": CBAM(512),   # layer4 输出通道数
#             })
#
#         # 保存窗口位置和可视化标志
#         self.window_pos = window_pos
#         self.visualize = visualize
#
#     def create_window_mask(self, feature_map, window_pos):
#         """
#         Create a mask with a fixed window region.
#
#         Args:
#             feature_map (torch.Tensor): Input feature map of shape (B, C, H, W).
#             window_pos (tuple): Fixed window position (x1, y1, x2, y2).
#
#         Returns:
#             torch.Tensor: A mask of shape (B, 1, H, W) with 1s in the window region and 0s elsewhere.
#         """
#         b, _, h, w = feature_map.size()
#         x1, y1, x2, y2 = window_pos
#         # Normalize window position to feature map size
#         x1 = int(x1 * w / 640)  # Scale to feature map width
#         x2 = int(x2 * w / 640)
#         y1 = int(y1 * h / 480)  # Scale to feature map height
#         y2 = int(y2 * h / 480)
#
#         mask = torch.zeros((b, 1, h, w), device=feature_map.device)
#         mask[:, :, y1:y2, x1:x2] = 1.0  # Set the window region to 1
#         return mask
#
#     def visualize_features(self, feature_map, mask, layer_name, window_pos, input_image):
#         """
#         Visualize feature map as heatmap and overlay mask along with the original input image.
#
#         Args:
#             feature_map (torch.Tensor): The feature map to visualize (B, C, H, W).
#             mask (torch.Tensor): The mask to overlay (B, 1, H, W).
#             layer_name (str): Name of the layer for labeling.
#             window_pos (tuple): Original window position (x1, y1, x2, y2).
#             input_image (torch.Tensor): The original input image (B, 3, H, W).
#         """
#         # Take the first sample and average across channels for feature map
#         feature_map = feature_map[0].mean(dim=0).detach().cpu().numpy()
#         mask = mask[0, 0].detach().cpu().numpy()
#
#         # Normalize feature map to [0, 1]
#         normalized_feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
#
#         # Prepare the original input image for visualization
#         input_image = input_image[0].permute(1, 2, 0).detach().cpu().numpy()  # (H, W, C)
#         input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())  # Normalize to [0, 1]
#
#         # Plot original image and heatmap side by side
#         fig, ax = plt.subplots(1, 2, figsize=(15, 10))
#
#         # Display the original input image
#         ax[0].imshow(input_image)
#         ax[0].set_title("Original Input Image")
#         ax[0].axis("off")
#
#         # Display the feature map heatmap
#         ax[1].imshow(normalized_feature_map, cmap='viridis', alpha=0.8)
#         ax[1].set_title(f"Feature Map Visualization: {layer_name}")
#         ax[1].axis("off")
#
#         # Overlay mask on the heatmap
#         h, w = mask.shape
#         x1, y1, x2, y2 = window_pos
#         x1, x2 = int(x1 * w / 640), int(x2 * w / 640)
#         y1, y2 = int(y1 * h / 480), int(y2 * h / 480)
#         ax[1].add_patch(
#             plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', lw=2, label='Window Mask')
#         )
#         ax[1].legend(loc='upper right')
#
#         # Show the plots
#         plt.show()
#         # print("normalized_feature_map.shape", normalized_feature_map.shape)
#
#     def forward(self, x):
#         # 获取 intermediate features
#         features = self.body(x)
#         enhanced_features = {}
#
#         for layer_name, feature_map in features.items():
#             if layer_name in self.cbam_layers:
#                 # 获取当前层的 CBAM 模块
#                 cbam = self.cbam_layers[layer_name]
#
#                 # 如果指定了窗口位置，则生成窗口 mask
#                 if self.window_pos is not None:
#                     mask = self.create_window_mask(feature_map, self.window_pos)
#                     # 融合窗口区域的显式引导和 CBAM
#                     alpha = 0.5  # 可根据任务需求调整
#                     feature_map_with_mask = feature_map * (1 + mask * alpha)
#                     enhanced_features[layer_name] = cbam(feature_map_with_mask)
#
#                     # 可视化特征图和 mask
#                     if self.visualize:
#                         # print("feature_map_with_mask", feature_map_with_mask.size())
#                         # print("mask", mask.size())
#                         # print("x", x.size())
#                         self.visualize_features(feature_map_with_mask, mask, layer_name, self.window_pos, x)
#                 else:
#                     # 如果没有指定窗口位置，则正常使用 CBAM
#                     enhanced_features[layer_name] = cbam(feature_map)
#             else:
#                 enhanced_features[layer_name] = feature_map
#         return enhanced_features

class BackCBAMbonewithMask(BackboneBase):
    """ResNet backbone with CBAM, frozen BatchNorm, and window-based attention."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 window_pos: tuple = None,
                 visualize: bool = False,
                 gaussian_kernel_size: int = 15,
                 sigma: float = 5):
        """
        Args:
            name (str): ResNet model name (e.g., 'resnet18', 'resnet50').
            train_backbone (bool): Whether to train the backbone.
            return_interm_layers (bool): Whether to return intermediate layers.
            dilation (bool): Use dilated convolutions or not.
            window_pos (tuple): The fixed window position as (x1, y1, x2, y2).
            visualize (bool): Whether to visualize feature maps and masks.
            gaussian_kernel_size (int): Kernel size for Gaussian smoothing.
            sigma (float): Standard deviation for Gaussian smoothing.
        """
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)

        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

        # 根据 return_interm_layers 的值设置 CBAM 模块
        if return_interm_layers:
            self.cbam_layers = nn.ModuleDict({
                "0": CBAM(64),    # layer1 输出通道数
                "1": CBAM(128),   # layer2 输出通道数
                "2": CBAM(256),   # layer3 输出通道数
                "3": CBAM(512),   # layer4 输出通道数
            })
        else:
            self.cbam_layers = nn.ModuleDict({
                "0": CBAM(512),   # layer4 输出通道数
            })

        # 保存窗口位置和可视化标志
        self.window_pos = window_pos
        self.visualize = visualize
        self.gaussian_kernel_size = gaussian_kernel_size
        self.sigma = sigma

        # 定义 alpha 为可学习参数
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def create_window_mask(self, feature_map, window_pos):
        """
        Create a mask with a fixed window region and apply Gaussian smoothing.

        Args:
            feature_map (torch.Tensor): Input feature map of shape (B, C, H, W).
            window_pos (tuple): Fixed window position (x1, y1, x2, y2).

        Returns:
            torch.Tensor: A smoothed mask of shape (B, 1, H, W).
        """
        b, _, h, w = feature_map.size()
        x1, y1, x2, y2 = window_pos
        # Normalize window position to feature map size
        x1 = int(x1 * w / 640)  # Scale to feature map width
        x2 = int(x2 * w / 640)
        y1 = int(y1 * h / 480)  # Scale to feature map height
        y2 = int(y2 * h / 480)

        mask = torch.zeros((b, 1, h, w), device=feature_map.device)
        mask[:, :, y1:y2, x1:x2] = 1.0  # Set the window region to 1

        # Apply Gaussian smoothing to the mask
        kernel_size = self.gaussian_kernel_size
        sigma = self.sigma
        gauss_kernel = self.create_gaussian_kernel(kernel_size, sigma, device=feature_map.device)
        smoothed_mask = F.conv2d(mask, gauss_kernel, padding=kernel_size // 2, groups=1)
        return smoothed_mask

    def create_gaussian_kernel(self, kernel_size, sigma, device):
        """
        Create a 2D Gaussian kernel for smoothing.

        Args:
            kernel_size (int): Size of the Gaussian kernel (odd number).
            sigma (float): Standard deviation of the Gaussian distribution.
            device (torch.device): The device to create the kernel on.

        Returns:
            torch.Tensor: A Gaussian kernel of shape (1, 1, kernel_size, kernel_size).
        """
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, kernel_size, kernel_size)

    def visualize_features(self, feature_map, mask, layer_name, window_pos, input_image):
        """
        Visualize feature map as heatmap and overlay mask along with the original input image.

        Args:
            feature_map (torch.Tensor): The feature map to visualize (B, C, H, W).
            mask (torch.Tensor): The mask to overlay (B, 1, H, W).
            layer_name (str): Name of the layer for labeling.
            window_pos (tuple): Original window position (x1, y1, x2, y2).
            input_image (torch.Tensor): The original input image (B, 3, H, W).
        """
        # Visualization code remains unchanged
        # ...

    def forward(self, x):
        # 获取 intermediate features
        features = self.body(x)
        enhanced_features = {}

        for layer_name, feature_map in features.items():
            if layer_name in self.cbam_layers:
                # 获取当前层的 CBAM 模块
                cbam = self.cbam_layers[layer_name]

                # 如果指定了窗口位置，则生成窗口 mask
                if self.window_pos is not None:
                    mask = self.create_window_mask(feature_map, self.window_pos)
                    # 融合窗口区域的显式引导和 CBAM
                    # print("self.alpha is",self.alpha)
                    feature_map_with_mask = feature_map * (1 + mask * self.alpha)
                    enhanced_features[layer_name] = cbam(feature_map_with_mask)

                    # 可视化特征图和 mask
                    if self.visualize:
                        self.visualize_features(feature_map_with_mask, mask, layer_name, self.window_pos, x)
                else:
                    # 如果没有指定窗口位置，则正常使用 CBAM
                    enhanced_features[layer_name] = cbam(feature_map)
            else:
                enhanced_features[layer_name] = feature_map
        return enhanced_features