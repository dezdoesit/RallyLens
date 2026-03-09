"""
Convert WASB (HRNet) Tennis Ball Tracker from PyTorch to CoreML .mlpackage

Architecture: HRNet with parallel multi-resolution branches
Input:  (1, 9, 288, 512)  — 3 consecutive RGB frames stacked
Output: (1, 1, 288, 512)  — single-channel heatmap (max of 3 output frames)

The model outputs 3 heatmaps (one per input frame). We take the last frame's
heatmap as the detection output, then apply sigmoid.
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super().__init__()
        self.num_inchannels = num_inchannels
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM))
        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([self._make_one_branch(i, block, num_blocks, num_channels) for i in range(num_branches)])

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        fuse_layers = []
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.num_inchannels[j], self.num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(self.num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(self.num_inchannels[j], self.num_inchannels[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(self.num_inchannels[i])))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(self.num_inchannels[j], self.num_inchannels[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(self.num_inchannels[j]),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

# WASB config matching wasb.yaml
WASB_CFG = {
    'frames_in': 3,
    'frames_out': 3,
    'out_scales': [0],
    'MODEL': {
        'EXTRA': {
            'FINAL_CONV_KERNEL': 1,
            'PRETRAINED_LAYERS': ['*'],
            'STEM': {'INPLANES': 64, 'STRIDES': [1, 1]},
            'STAGE1': {'NUM_MODULES': 1, 'NUM_BRANCHES': 1, 'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': [1], 'NUM_CHANNELS': [32], 'FUSE_METHOD': 'SUM'},
            'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [2, 2], 'NUM_CHANNELS': [16, 32], 'FUSE_METHOD': 'SUM'},
            'STAGE3': {'NUM_MODULES': 1, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [2, 2, 2], 'NUM_CHANNELS': [16, 32, 64], 'FUSE_METHOD': 'SUM'},
            'STAGE4': {'NUM_MODULES': 1, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [2, 2, 2, 2], 'NUM_CHANNELS': [16, 32, 64, 128], 'FUSE_METHOD': 'SUM'},
            'DECONV': {'NUM_DECONVS': 0, 'KERNEL_SIZE': [], 'NUM_BASIC_BLOCKS': 2},
        }
    }
}


class HRNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._frames_in = cfg['frames_in']
        self._frames_out = cfg['frames_out']
        self._out_scales = cfg['out_scales']
        self._stem_strides = cfg['MODEL']['EXTRA']['STEM']['STRIDES']
        self._stem_inplanes = cfg['MODEL']['EXTRA']['STEM']['INPLANES']

        self.conv1 = nn.Conv2d(3*self._frames_in, self._stem_inplanes, 3, stride=self._stem_strides[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self._stem_inplanes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(self._stem_inplanes, self._stem_inplanes, 3, stride=self._stem_strides[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self._stem_inplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1
        s1 = cfg['MODEL']['EXTRA']['STAGE1']
        block = blocks_dict[s1['BLOCK']]
        self.layer1 = self._make_layer(block, self._stem_inplanes, s1['NUM_CHANNELS'][0], s1['NUM_BLOCKS'][0])
        stage1_out = block.expansion * s1['NUM_CHANNELS'][0]

        # Stage 2
        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_ch = [c * block.expansion for c in self.stage2_cfg['NUM_CHANNELS']]
        self.transition1 = self._make_transition_layer([stage1_out], num_ch)
        self.stage2, pre_ch = self._make_stage(self.stage2_cfg, num_ch)

        # Stage 3
        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_ch = [c * block.expansion for c in self.stage3_cfg['NUM_CHANNELS']]
        self.transition2 = self._make_transition_layer(pre_ch, num_ch)
        self.stage3, pre_ch = self._make_stage(self.stage3_cfg, num_ch)

        # Stage 4
        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_ch = [c * block.expansion for c in self.stage4_cfg['NUM_CHANNELS']]
        self.transition3 = self._make_transition_layer(pre_ch, num_ch)
        self.stage4, pre_ch = self._make_stage(self.stage4_cfg, num_ch, multi_scale_output=True)

        self.num_deconvs = cfg['MODEL']['EXTRA']['DECONV']['NUM_DECONVS']

        # Final layer
        kernel = cfg['MODEL']['EXTRA']['FINAL_CONV_KERNEL']
        self.final_layers = nn.ModuleList([
            nn.Conv2d(pre_ch[s], self._frames_out, kernel_size=kernel) for s in self._out_scales
        ])

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre, num_channels_cur):
        transition_layers = []
        for i in range(len(num_channels_cur)):
            if i < len(num_channels_pre):
                if num_channels_cur[i] != num_channels_pre[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre[i], num_channels_cur[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - len(num_channels_pre)):
                    inc = num_channels_pre[-1]
                    outc = num_channels_cur[i] if j == i - len(num_channels_pre) else inc
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inc, outc, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outc, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        block = blocks_dict[layer_config['BLOCK']]
        modules = []
        for i in range(layer_config['NUM_MODULES']):
            ms = True if (multi_scale_output or i < layer_config['NUM_MODULES'] - 1) else False
            modules.append(HighResolutionModule(
                layer_config['NUM_BRANCHES'], block, layer_config['NUM_BLOCKS'],
                num_inchannels, layer_config['NUM_CHANNELS'], layer_config['FUSE_METHOD'], ms))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            x_list.append(self.transition1[i](x) if self.transition1[i] is not None else x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            x_list.append(self.transition2[i](y_list[-1]) if self.transition2[i] is not None else y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            x_list.append(self.transition3[i](y_list[-1]) if self.transition3[i] is not None else y_list[i])
        y_list = self.stage4(x_list)

        # Only use scale 0 (highest resolution branch)
        out = self.final_layers[0](y_list[0])
        return out


class WASBWrapper(nn.Module):
    """Wraps HRNet to output a single sigmoid heatmap (last frame only)."""
    def __init__(self, hrnet):
        super().__init__()
        self.hrnet = hrnet
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.hrnet(x)        # (B, 3, H, W) — one heatmap per frame
        last = out[:, 2:3, :, :]   # Take last frame's heatmap: (B, 1, H, W)
        return self.sigmoid(last)


def convert(weights_path: str, output_path: str):
    model = HRNet(WASB_CFG)
    ckpt = torch.load(weights_path, map_location='cpu')
    sd = ckpt['model_state_dict']
    model.load_state_dict(sd)
    model.eval()

    print(f"Loaded WASB weights from {weights_path}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    wrapper = WASBWrapper(model)
    wrapper.eval()

    # Input: 3 frames × 3 RGB = 9 channels, 288×512
    example = torch.rand(1, 9, 288, 512)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
        out = traced(example)
        print(f"Output shape: {out.shape}")  # (1, 1, 288, 512)
        print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="input_frames",
            shape=(1, 9, 288, 512),
            dtype=np.float32,
        )],
        outputs=[ct.TensorType(name="heatmap", dtype=np.float32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
    )

    mlmodel.short_description = "WASB (HRNet): Tennis Ball Tracker — F1=94.0"
    mlmodel.input_description["input_frames"] = (
        "3 consecutive RGB frames stacked (1x9x288x512), normalized [0,1]"
    )
    mlmodel.output_description["heatmap"] = (
        "Single-channel sigmoid heatmap (1x1x288x512)"
    )

    mlmodel.save(output_path)
    print(f"Saved CoreML model: {output_path}")


if __name__ == "__main__":
    convert(
        "wasb_tennis_best.pth.tar",
        "TennisVision/Models/BallTracker.mlpackage",
    )
