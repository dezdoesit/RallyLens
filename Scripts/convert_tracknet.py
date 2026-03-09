"""
Convert TrackNet (yastrebksv variant) from PyTorch to CoreML .mlpackage

Architecture: 18 ConvBlocks (Conv2d+BN+ReLU) with 3 MaxPool + 3 Upsample
Input:  (1, 9, 360, 640)  — 3 consecutive RGB frames stacked
Output: (1, 256, 360, 640) — 256-class per-pixel classification
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
    """
    TrackNet variant used by yastrebksv.
    18 conv blocks, U-Net style encoder-decoder.
    Output: 256-class per-pixel classification.
    """

    def __init__(self, input_channels=9, out_channels=256):
        super().__init__()
        # Encoder
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

        # Bottleneck
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)

        # Decoder
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

        # Output: 256-class per-pixel
        self.conv18 = ConvBlock(64, out_channels)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder
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

        # Bottleneck
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        # Decoder
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
        x = self.softmax(x)
        return x


def convert(weights_path: str, output_path: str):
    model = BallTrackerNet()

    # Load weights
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded weights from {weights_path}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Trace with example input: 3 frames x 3 channels = 9 channels, 360x640
    example_input = torch.rand(1, 9, 360, 640)

    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)

    # Verify trace output
    with torch.no_grad():
        out = traced(example_input)
        print(f"Output shape: {out.shape}")  # Expected: (1, 256, 360, 640)

    # Convert to CoreML
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

    # Add metadata
    mlmodel.short_description = "TrackNet: Tennis Ball Tracker (256-class per-pixel)"
    mlmodel.input_description["input_frames"] = (
        "3 consecutive RGB frames stacked (1x9x360x640), normalized [0,1]"
    )
    mlmodel.output_description["heatmap"] = (
        "Per-pixel classification heatmap (1x256x360x640), softmax applied"
    )

    mlmodel.save(output_path)
    print(f"Saved CoreML model: {output_path}")


if __name__ == "__main__":
    convert(
        "model_best.pt",
        "TennisVision/Models/BallTracker.mlpackage",
    )
