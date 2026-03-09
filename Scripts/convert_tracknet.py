"""
Convert TrackNet (yastrebksv variant) from PyTorch to CoreML .mlpackage

Architecture: 18 ConvBlocks (Conv2d+ReLU+BN) with 3 MaxPool + 3 Upsample
Input:  (1, 9, 360, 640)  — 3 consecutive RGB frames stacked
Output: (1, 1, 360, 640)  — single-channel heatmap (weighted sum of 256-class softmax)

The 256-class softmax is collapsed into a single channel by computing:
   heatmap[y,x] = sum(c * P(c, y, x) for c in 0..255)
This produces a "expected intensity" map where the ball pixel has value ~200-255
and background pixels have value ~0.
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module):
    def __init__(self, input_channels=9, out_channels=256):
        super().__init__()
        self.conv1 = ConvBlock(input_channels, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)
        self.conv18 = ConvBlock(64, out_channels)
        self.softmax = nn.Softmax(dim=1)

        # Channel weights for weighted sum: [0, 1, 2, ..., 255]
        self.register_buffer(
            'channel_weights',
            torch.arange(out_channels, dtype=torch.float32).view(1, out_channels, 1, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.up1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.up2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.up3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.softmax(x)  # (1, 256, 360, 640)

        # Collapse 256 channels → single heatmap via weighted sum
        # heatmap[y,x] = sum(c * P(c,y,x)) for c in 0..255
        heatmap = (x * self.channel_weights).sum(dim=1, keepdim=True)  # (1, 1, 360, 640)
        return heatmap


def convert(weights_path: str, output_path: str):
    model = BallTrackerNet()
    state_dict = torch.load(weights_path, map_location="cpu")
    # Load only the conv weights (channel_weights is a buffer, not in state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"Loaded weights from {weights_path}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    example_input = torch.rand(1, 9, 360, 640)

    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
        out = traced(example_input)
        print(f"Output shape: {out.shape}")  # Should be (1, 1, 360, 640)

    # Verify with real-like input
    with torch.no_grad():
        test_out = model(example_input)
        print(f"Output range: [{test_out.min().item():.3f}, {test_out.max().item():.3f}]")

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="input_frames",
            shape=(1, 9, 360, 640),
            dtype=np.float32,
        )],
        outputs=[ct.TensorType(name="heatmap", dtype=np.float32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
    )

    mlmodel.short_description = "TrackNet: Tennis Ball Tracker"
    mlmodel.input_description["input_frames"] = (
        "3 consecutive RGB frames stacked (1x9x360x640), normalized [0,1]"
    )
    mlmodel.output_description["heatmap"] = (
        "Single-channel heatmap (1x1x360x640), value = expected intensity 0-255"
    )

    mlmodel.save(output_path)
    print(f"Saved CoreML model: {output_path}")


if __name__ == "__main__":
    convert(
        "model_best.pt",
        "TennisVision/Models/BallTracker.mlpackage",
    )
