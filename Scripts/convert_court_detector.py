"""
Convert TennisCourtDetector from PyTorch to CoreML .mlpackage

Architecture: Same 18-ConvBlock U-Net as TrackNet, but:
  - Input:  (1, 3, 360, 640) — single RGB frame
  - Output: (1, 14, 360, 640) — 14 heatmaps (one per court keypoint), sigmoid applied

The 15th channel (court center, training-only) is discarded.
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


class CourtDetectorNet(nn.Module):
    def __init__(self, input_channels=3, out_channels=15):
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
        self.sigmoid = nn.Sigmoid()

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
        x = self.sigmoid(x)        # Apply sigmoid (not in original forward)
        x = x[:, :14, :, :]        # Drop 15th channel (court center)
        return x


def convert(weights_path: str, output_path: str):
    model = CourtDetectorNet()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded weights from {weights_path}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    example_input = torch.rand(1, 3, 360, 640)

    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
        out = traced(example_input)
        print(f"Output shape: {out.shape}")  # Should be (1, 14, 360, 640)
        print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="input_frame",
            shape=(1, 3, 360, 640),
            dtype=np.float32,
        )],
        outputs=[ct.TensorType(name="keypoint_heatmaps", dtype=np.float32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
    )

    mlmodel.short_description = "Tennis Court Detector: 14 keypoint heatmaps"
    mlmodel.input_description["input_frame"] = (
        "Single RGB frame (1x3x360x640), normalized [0,1]"
    )
    mlmodel.output_description["keypoint_heatmaps"] = (
        "14 keypoint heatmaps (1x14x360x640), sigmoid applied, find peak per channel"
    )

    mlmodel.save(output_path)
    print(f"Saved CoreML model: {output_path}")


if __name__ == "__main__":
    convert(
        "model_tennis_court_det.pt",
        "TennisVision/Models/CourtDetector.mlpackage",
    )
