"""
Test TrackNet ball detection on a video file.

1. Loads the PyTorch model (BallTrackerNet with 18 ConvBlocks)
2. Extracts 3 consecutive frames from the test video
3. Resizes to 640x360, normalizes to [0,1], stacks as 9-channel input
4. Runs PyTorch inference
5. Runs CoreML inference (converts model if .mlpackage doesn't exist)
6. Compares outputs and saves debug visualizations
"""

import os
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn as nn

# ─── Project paths ────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "model_best.pt")
VIDEO_PATH = os.path.join(PROJECT_ROOT, "Videos_20260309092950.mp4")
COREML_PATH = os.path.join(PROJECT_ROOT, "TennisVision/Models/BallTracker.mlpackage")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Model Architecture (matches convert_tracknet.py exactly) ─────
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
    TrackNet variant (yastrebksv).
    18 conv blocks, U-Net encoder-decoder.
    Output: 256-class per-pixel classification (softmax).
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


# ─── Utility functions ────────────────────────────────────────────

def extract_frames(video_path, start_frame=100, count=3):
    """Extract `count` consecutive frames starting at `start_frame`."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for i in range(count):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {start_frame + i}")
        frames.append(frame)
    cap.release()
    print(f"Extracted frames {start_frame}-{start_frame + count - 1}, "
          f"original size: {frames[0].shape[1]}x{frames[0].shape[0]}")
    return frames


def preprocess_frames(frames, target_w=640, target_h=360):
    """Resize, normalize, and stack 3 frames into a (1, 9, 360, 640) tensor."""
    processed = []
    for frame in frames:
        # Resize
        resized = cv2.resize(frame, (target_w, target_h))
        # BGR -> RGB, normalize to [0, 1]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # HWC -> CHW
        chw = np.transpose(rgb, (2, 0, 1))  # (3, 360, 640)
        processed.append(chw)
    # Stack: (9, 360, 640)
    stacked = np.concatenate(processed, axis=0)
    # Add batch dimension: (1, 9, 360, 640)
    tensor = torch.from_numpy(stacked).unsqueeze(0)
    print(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"  Value range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    return tensor


def decode_output(output_tensor):
    """
    Decode the 256-class per-pixel output.

    The model outputs (1, 256, 360, 640) with softmax applied.
    For each pixel, the 256 channels represent a classification:
      - Class 0 = no ball
      - Classes 1-255 = ball present with varying "intensity" / confidence encoding

    Strategy:
      1. argmax across 256 channels -> pixel-wise class map (360, 640)
      2. Create heatmap: pixels with class > 0 are "ball detected"
      3. Find peak location
    """
    # output shape: (1, 256, 360, 640)
    output = output_tensor.squeeze(0).numpy()  # (256, 360, 640)

    # argmax per pixel
    argmax_map = np.argmax(output, axis=0)  # (360, 640), values in [0, 255]

    # Confidence: max probability across 256 classes for each pixel
    max_prob = np.max(output, axis=0)  # (360, 640)

    # Heatmap: sum of probabilities for non-zero classes (ball classes)
    # This gives us P(ball) = 1 - P(class_0) for each pixel
    ball_prob_map = 1.0 - output[0]  # (360, 640)

    # Statistics about the argmax map
    nonzero_count = np.count_nonzero(argmax_map)
    print(f"\n--- Output Analysis ---")
    print(f"Output shape: {output_tensor.shape}")
    print(f"Argmax map: unique classes = {np.unique(argmax_map).tolist()[:20]}{'...' if len(np.unique(argmax_map)) > 20 else ''}")
    print(f"Pixels with class > 0 (ball detected): {nonzero_count} / {argmax_map.size} "
          f"({100.0 * nonzero_count / argmax_map.size:.4f}%)")

    # Find peak in ball probability map
    peak_idx = np.unravel_index(np.argmax(ball_prob_map), ball_prob_map.shape)
    peak_y, peak_x = peak_idx
    peak_confidence = ball_prob_map[peak_y, peak_x]
    peak_class = argmax_map[peak_y, peak_x]

    print(f"\nBall probability map:")
    print(f"  Max P(ball) = {peak_confidence:.6f} at pixel ({peak_x}, {peak_y})")
    print(f"  Argmax class at peak = {peak_class}")
    print(f"  Mean P(ball) = {ball_prob_map.mean():.6f}")

    # Also check: what is the max probability value among non-zero argmax pixels?
    if nonzero_count > 0:
        nonzero_mask = argmax_map > 0
        ball_region_probs = max_prob[nonzero_mask]
        print(f"  Non-zero region max_prob: mean={ball_region_probs.mean():.6f}, "
              f"max={ball_region_probs.max():.6f}")

        # Weighted centroid of ball region
        ys, xs = np.where(nonzero_mask)
        weights = ball_prob_map[nonzero_mask]
        if weights.sum() > 0:
            cx = np.average(xs, weights=weights)
            cy = np.average(ys, weights=weights)
            print(f"  Weighted centroid: ({cx:.1f}, {cy:.1f})")
        else:
            cx, cy = float(peak_x), float(peak_y)
    else:
        cx, cy = float(peak_x), float(peak_y)
        print("  No ball pixels detected (all class 0).")

    return {
        "argmax_map": argmax_map,
        "ball_prob_map": ball_prob_map,
        "max_prob_map": max_prob,
        "peak_x": int(peak_x),
        "peak_y": int(peak_y),
        "peak_confidence": float(peak_confidence),
        "peak_class": int(peak_class),
        "centroid_x": float(cx),
        "centroid_y": float(cy),
        "nonzero_pixel_count": int(nonzero_count),
    }


def save_visualization(original_frame, result, output_path, title="PyTorch"):
    """Save debug visualization with detected ball marked on the original frame."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    h_target, w_target = 360, 640
    frame_resized = cv2.resize(original_frame, (w_target, h_target))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{title} TrackNet Inference Results", fontsize=14, fontweight='bold')

    # 1. Original frame with detected ball
    ax = axes[0, 0]
    ax.imshow(frame_rgb)
    if result["nonzero_pixel_count"] > 0:
        # Draw centroid
        ax.plot(result["centroid_x"], result["centroid_y"], 'o',
                color='yellow', markersize=15, markeredgecolor='white', markeredgewidth=2)
        ax.annotate(f'Ball ({result["centroid_x"]:.0f}, {result["centroid_y"]:.0f})\n'
                    f'conf={result["peak_confidence"]:.4f}',
                    xy=(result["centroid_x"], result["centroid_y"]),
                    xytext=(result["centroid_x"] + 30, result["centroid_y"] - 30),
                    color='yellow', fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='yellow'))
        # Draw peak
        ax.plot(result["peak_x"], result["peak_y"], 'x',
                color='red', markersize=12, markeredgewidth=2)
    ax.set_title("Original + Detected Ball")
    ax.axis('off')

    # 2. Ball probability heatmap
    ax = axes[0, 1]
    im = ax.imshow(result["ball_prob_map"], cmap='hot', vmin=0, vmax=max(0.01, result["ball_prob_map"].max()))
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f'P(ball) heatmap (max={result["ball_prob_map"].max():.6f})')
    ax.axis('off')

    # 3. Argmax class map
    ax = axes[1, 0]
    im = ax.imshow(result["argmax_map"], cmap='viridis')
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f'Argmax class map (nonzero pixels: {result["nonzero_pixel_count"]})')
    ax.axis('off')

    # 4. Overlay: frame + heatmap
    ax = axes[1, 1]
    ax.imshow(frame_rgb)
    ax.imshow(result["ball_prob_map"], cmap='hot', alpha=0.5,
              vmin=0, vmax=max(0.01, result["ball_prob_map"].max()))
    if result["nonzero_pixel_count"] > 0:
        ax.plot(result["centroid_x"], result["centroid_y"], 'o',
                color='cyan', markersize=12, markeredgecolor='white', markeredgewidth=2)
    ax.set_title("Overlay: Frame + Heatmap")
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {output_path}")


def test_multiple_frame_ranges(model, video_path):
    """Test on multiple frame ranges to find the best detection."""
    print("\n" + "=" * 70)
    print("Testing multiple frame ranges to find ball detections...")
    print("=" * 70)

    frame_starts = [50, 100, 200, 500, 800, 1000, 1500, 2000, 2500, 3000]
    best_result = None
    best_confidence = 0
    best_frame_start = 0
    best_frames = None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    for start in frame_starts:
        if start + 3 > total_frames:
            continue
        try:
            frames = extract_frames(video_path, start_frame=start, count=3)
            input_tensor = preprocess_frames(frames)

            with torch.no_grad():
                output = model(input_tensor)

            result = decode_output(output)

            if result["peak_confidence"] > best_confidence:
                best_confidence = result["peak_confidence"]
                best_result = result
                best_frame_start = start
                best_frames = frames

            print(f"  Frames {start}-{start+2}: P(ball)={result['peak_confidence']:.6f}, "
                  f"nonzero={result['nonzero_pixel_count']}")
        except Exception as e:
            print(f"  Frames {start}: ERROR - {e}")

    print(f"\nBest detection at frames {best_frame_start}-{best_frame_start+2}: "
          f"P(ball)={best_confidence:.6f}")

    return best_result, best_frame_start, best_frames


# ─── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("TrackNet Ball Detection Test")
    print("=" * 70)

    # 1. Load model
    print("\n[1] Loading PyTorch model...")
    model = BallTrackerNet()
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"    Model loaded: {param_count:,} parameters")

    # 2. Extract frames 100-102
    print("\n[2] Extracting frames 100-102 from video...")
    frames = extract_frames(VIDEO_PATH, start_frame=100, count=3)

    # 3. Preprocess
    print("\n[3] Preprocessing frames...")
    input_tensor = preprocess_frames(frames)

    # 4. PyTorch inference
    print("\n[4] Running PyTorch inference...")
    t0 = time.perf_counter()
    with torch.no_grad():
        output_pt = model(input_tensor)
    t_pt = time.perf_counter() - t0
    print(f"    PyTorch inference time: {t_pt * 1000:.1f} ms")

    # 5. Decode output
    print("\n[5] Decoding PyTorch output...")
    result_pt = decode_output(output_pt)

    # 6. Visualization for frames 100-102
    print("\n[6] Saving PyTorch visualization...")
    save_visualization(
        frames[2],  # Use the last (most recent) frame
        result_pt,
        os.path.join(OUTPUT_DIR, "tracknet_pytorch_frames100.png"),
        title="PyTorch (frames 100-102)"
    )

    # 7. Test multiple frame ranges to find good detections
    best_result, best_start, best_frames = test_multiple_frame_ranges(model, VIDEO_PATH)
    if best_frames is not None:
        save_visualization(
            best_frames[2],
            best_result,
            os.path.join(OUTPUT_DIR, "tracknet_pytorch_best.png"),
            title=f"PyTorch Best (frames {best_start}-{best_start+2})"
        )

    # 8. CoreML conversion and test
    print("\n" + "=" * 70)
    print("[7] CoreML Conversion and Comparison")
    print("=" * 70)

    try:
        import coremltools as ct

        # Convert if .mlpackage doesn't exist
        if not os.path.exists(COREML_PATH):
            print("    CoreML model not found. Converting from PyTorch...")
            os.makedirs(os.path.dirname(COREML_PATH), exist_ok=True)

            example_input = torch.rand(1, 9, 360, 640)
            with torch.no_grad():
                traced = torch.jit.trace(model, example_input)

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
            mlmodel.short_description = "TrackNet: Tennis Ball Tracker (256-class per-pixel)"
            mlmodel.save(COREML_PATH)
            print(f"    Saved CoreML model: {COREML_PATH}")
        else:
            print(f"    Found existing CoreML model: {COREML_PATH}")

        # Load CoreML model
        print("    Loading CoreML model...")
        mlmodel = ct.models.MLModel(COREML_PATH)

        # Print model spec info
        spec = mlmodel.get_spec()
        print(f"    CoreML spec version: {spec.specificationVersion}")
        for inp in spec.description.input:
            print(f"    Input: {inp.name}")
        for out in spec.description.output:
            print(f"    Output: {out.name}")

        # Run CoreML inference using the same input
        print("    Running CoreML inference...")
        input_np = input_tensor.numpy()
        t0 = time.perf_counter()
        coreml_pred = mlmodel.predict({"input_frames": input_np})
        t_cml = time.perf_counter() - t0
        print(f"    CoreML inference time: {t_cml * 1000:.1f} ms")

        # Get CoreML output
        coreml_output = coreml_pred["heatmap"]  # numpy array
        print(f"    CoreML output shape: {coreml_output.shape}")
        coreml_tensor = torch.from_numpy(coreml_output)

        # Decode CoreML output
        print("\n    Decoding CoreML output...")
        result_cml = decode_output(coreml_tensor)

        # Compare PyTorch vs CoreML
        print("\n--- PyTorch vs CoreML Comparison ---")
        pt_out = output_pt.numpy()
        cml_out = coreml_output

        # Ensure shapes match
        if pt_out.shape == cml_out.shape:
            abs_diff = np.abs(pt_out - cml_out)
            print(f"    Max absolute difference: {abs_diff.max():.8f}")
            print(f"    Mean absolute difference: {abs_diff.mean():.8f}")
            print(f"    Outputs match (atol=1e-4): {np.allclose(pt_out, cml_out, atol=1e-4)}")
            print(f"    Outputs match (atol=1e-3): {np.allclose(pt_out, cml_out, atol=1e-3)}")
            print(f"    Outputs match (atol=1e-2): {np.allclose(pt_out, cml_out, atol=1e-2)}")
        else:
            print(f"    Shape mismatch: PyTorch {pt_out.shape} vs CoreML {cml_out.shape}")

        print(f"\n    PyTorch peak: ({result_pt['peak_x']}, {result_pt['peak_y']}) "
              f"conf={result_pt['peak_confidence']:.6f}")
        print(f"    CoreML  peak: ({result_cml['peak_x']}, {result_cml['peak_y']}) "
              f"conf={result_cml['peak_confidence']:.6f}")

        # Save CoreML visualization
        save_visualization(
            frames[2],
            result_cml,
            os.path.join(OUTPUT_DIR, "tracknet_coreml_frames100.png"),
            title="CoreML (frames 100-102)"
        )

    except Exception as e:
        print(f"    CoreML test failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model: BallTrackerNet ({param_count:,} params)")
    print(f"Video: {VIDEO_PATH}")
    print(f"Input shape: (1, 9, 360, 640)")
    print(f"Output shape: (1, 256, 360, 640)")
    print(f"PyTorch inference time: {t_pt * 1000:.1f} ms")
    print(f"Frames 100-102 detection:")
    print(f"  Peak pixel: ({result_pt['peak_x']}, {result_pt['peak_y']})")
    print(f"  Peak confidence (P(ball)): {result_pt['peak_confidence']:.6f}")
    print(f"  Peak class: {result_pt['peak_class']}")
    print(f"  Centroid: ({result_pt['centroid_x']:.1f}, {result_pt['centroid_y']:.1f})")
    print(f"  Ball pixels: {result_pt['nonzero_pixel_count']}")
    if best_result:
        print(f"Best detection across video (frames {best_start}-{best_start+2}):")
        print(f"  Peak confidence: {best_result['peak_confidence']:.6f}")
        print(f"  Centroid: ({best_result['centroid_x']:.1f}, {best_result['centroid_y']:.1f})")
        print(f"  Ball pixels: {best_result['nonzero_pixel_count']}")
    print(f"\nOutput files saved in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
