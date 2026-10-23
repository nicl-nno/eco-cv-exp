"""ResNet34 U-Net compatible with segmentation_models_pytorch checkpoints."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet34_Weights, convnext_tiny, resnet34


class Encoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        self.conv1, self.bn1, self.relu, self.maxpool = net.conv1, net.bn1, net.relu, net.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]


class DecoderBlock(nn.Module):
    def __init__(self, cin: int, skip: int, cout: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(cin + skip, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        return self.conv2(self.conv1(x))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # ``blocks`` preserves the naming used by segmentation_models_pytorch.
        self.blocks = nn.ModuleList([
            DecoderBlock(512, 256, 256),
            DecoderBlock(256, 128, 128),
            DecoderBlock(128, 64, 64),
            DecoderBlock(64, 64, 32),
            DecoderBlock(32, 0, 16),
        ])

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        skips = list(reversed(features[:-1]))
        for index, block in enumerate(self.blocks):
            x = block(x, skips[index] if index < len(skips) else None)
        return x


class ResNet34UNet(nn.Module):
    """U-Net from the article: ImageNet ResNet34 encoder and a five-stage decoder."""
    def __init__(self, classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.encoder = Encoder(pretrained)
        self.decoder = Decoder()
        self.segmentation_head = nn.Sequential(nn.Conv2d(16, classes, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.segmentation_head(self.decoder(features))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class SEDecoder(Decoder):
    def __init__(self):
        super().__init__()
        self.attention = nn.ModuleList(SqueezeExcitation(c) for c in (256, 128, 64, 32, 16))

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        skips = list(reversed(features[:-1]))
        for index, (block, attention) in enumerate(zip(self.blocks, self.attention)):
            x = attention(block(x, skips[index] if index < len(skips) else None))
        return x


class ResNet34UNetSE(ResNet34UNet):
    """U-Net with channel attention after every decoder stage."""
    def __init__(self, classes: int = 3, pretrained: bool = True):
        super().__init__(classes, pretrained)
        self.decoder = SEDecoder()


def conv_norm_relu(cin: int, cout: int, kernel: int, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, padding=(kernel // 2) * dilation,
                  dilation=dilation, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class ASPPContext(nn.Module):
    """Residual multi-scale context at the 1/32 encoder resolution."""
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([
            conv_norm_relu(512, 128, 1),
            conv_norm_relu(512, 128, 3, dilation=2),
            conv_norm_relu(512, 128, 3, dilation=4),
            conv_norm_relu(512, 128, 3, dilation=6),
        ])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(512, 128, 1), nn.ReLU(inplace=True))
        self.project = conv_norm_relu(640, 512, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        context = self.project(torch.cat([*(branch(x) for branch in self.branches), pooled], dim=1))
        return x + self.scale * context


class ResNet34UNetASPP(ResNet34UNet):
    """U-Net with a multi-scale context module before the decoder."""
    def __init__(self, classes: int = 3, pretrained: bool = True):
        super().__init__(classes, pretrained)
        self.context = ASPPContext()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        features[-1] = self.context(features[-1])
        return self.segmentation_head(self.decoder(features))


class ResNet34FPN(nn.Module):
    """Top-down feature pyramid with lateral fusion at all encoder scales."""
    def __init__(self, classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.encoder = Encoder(pretrained)
        self.lateral = nn.ModuleList(nn.Conv2d(c, 128, 1) for c in (64, 64, 128, 256, 512))
        self.smooth = conv_norm_relu(128, 128, 3)
        self.segmentation_head = nn.Conv2d(128, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pyramid = self.lateral[-1](features[-1])
        for index in range(len(features) - 2, -1, -1):
            pyramid = self.lateral[index](features[index]) + F.interpolate(
                pyramid, size=features[index].shape[-2:], mode="bilinear", align_corners=False,
            )
        logits = self.segmentation_head(self.smooth(pyramid))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class ConvNeXtTinyUNet(nn.Module):
    """ImageNet-pretrained ConvNeXt Tiny with a native-resolution U-Net decoder."""
    def __init__(self, classes: int = 3, pretrained: bool = True):
        super().__init__()
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        self.encoder = backbone.features
        self.decoder = nn.ModuleList([
            DecoderBlock(768, 384, 256),  # /32 -> /16
            DecoderBlock(256, 192, 128),  # /16 -> /8
            DecoderBlock(128, 96, 64),    # /8 -> /4
            DecoderBlock(64, 0, 32),      # /4 -> /2
            DecoderBlock(32, 0, 16),      # /2 -> native
        ])
        self.segmentation_head = nn.Conv2d(16, classes, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for index, stage in enumerate(self.encoder):
            x = stage(x)
            if index in (1, 3, 5, 7):
                features.append(x)
        x = features[-1]
        for block, skip in zip(self.decoder, (features[-2], features[-3], features[-4], None, None)):
            x = block(x, skip)
        return self.segmentation_head(x)


class ConvNeXtTinyUNetSE(ConvNeXtTinyUNet):
    """Same ImageNet encoder and decoder widths, with channel attention on skips."""
    def __init__(self, classes: int = 3, pretrained: bool = True):
        super().__init__(classes, pretrained)
        self.attention = nn.ModuleList(SqueezeExcitation(c) for c in (256, 128, 64, 32, 16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for index, stage in enumerate(self.encoder):
            x = stage(x)
            if index in (1, 3, 5, 7):
                features.append(x)
        x = features[-1]
        for block, attention, skip in zip(
            self.decoder, self.attention, (features[-2], features[-3], features[-4], None, None)
        ):
            x = attention(block(x, skip))
        return self.segmentation_head(x)


class ConvNeXtTinyFPN(nn.Module):
    """Same pretrained encoder, lightweight top-down pyramid instead of U-Net skips."""
    def __init__(self, classes: int = 3, pretrained: bool = True):
        super().__init__()
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        self.encoder = backbone.features
        self.lateral = nn.ModuleList(nn.Conv2d(c, 128, 1) for c in (96, 192, 384, 768))
        self.smooth = conv_norm_relu(128, 128, 3)
        self.segmentation_head = nn.Conv2d(128, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = []
        for index, stage in enumerate(self.encoder):
            x = stage(x)
            if index in (1, 3, 5, 7):
                features.append(x)
        pyramid = self.lateral[-1](features[-1])
        for index in range(len(features) - 2, -1, -1):
            pyramid = self.lateral[index](features[index]) + F.interpolate(
                pyramid, size=features[index].shape[-2:], mode="bilinear", align_corners=False,
            )
        return F.interpolate(self.segmentation_head(self.smooth(pyramid)),
                             size=input_size, mode="bilinear", align_corners=False)


ARCHITECTURES = {
    "resnet34_unet": ResNet34UNet,
    "resnet34_unet_se": ResNet34UNetSE,
    "resnet34_unet_aspp": ResNet34UNetASPP,
    "resnet34_fpn": ResNet34FPN,
    "convnext_tiny_unet": ConvNeXtTinyUNet,
    "convnext_tiny_unet_se": ConvNeXtTinyUNetSE,
    "convnext_tiny_fpn": ConvNeXtTinyFPN,
}


def build_model(architecture: str = "resnet34_unet", *, pretrained: bool = True) -> nn.Module:
    try:
        model_class = ARCHITECTURES[architecture]
    except KeyError as exc:
        raise ValueError(f"Unknown architecture {architecture!r}; choose from {sorted(ARCHITECTURES)}") from exc
    return model_class(pretrained=pretrained)
