# Copyright 2026 Zijiang Yang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn

try:
    from torchvision.models import (
        ConvNeXt_Base_Weights,
        ConvNeXt_Large_Weights,
        ConvNeXt_Small_Weights,
        ConvNeXt_Tiny_Weights,
        convnext_base,
        convnext_large,
        convnext_small,
        convnext_tiny,
    )
    _HAS_TV_WEIGHTS = True
except ImportError:
    from torchvision.models import (
        convnext_base,
        convnext_large,
        convnext_small,
        convnext_tiny,
    )
    ConvNeXt_Base_Weights = None
    ConvNeXt_Large_Weights = None
    ConvNeXt_Small_Weights = None
    ConvNeXt_Tiny_Weights = None
    _HAS_TV_WEIGHTS = False


_VARIANTS = {
    "tiny": (convnext_tiny, ConvNeXt_Tiny_Weights, [96, 192, 384, 768]),
    "small": (convnext_small, ConvNeXt_Small_Weights, [96, 192, 384, 768]),
    "base": (convnext_base, ConvNeXt_Base_Weights, [128, 256, 512, 1024]),
    "large": (convnext_large, ConvNeXt_Large_Weights, [192, 384, 768, 1536]),
    't': (convnext_tiny, ConvNeXt_Tiny_Weights, [96, 192, 384, 768]),
    's': (convnext_small, ConvNeXt_Small_Weights, [96, 192, 384, 768]),
    'b': (convnext_base, ConvNeXt_Base_Weights, [128, 256, 512, 1024]),
    'l': (convnext_large, ConvNeXt_Large_Weights, [192, 384, 768, 1536]),
}


class ConvNeXt(nn.Module):
    def __init__(
        self,
        variant="tiny",
        out_stages=(2, 3, 4),
        pretrain=True,
        disable_cudnn_benchmark=False,
        use_channels_last=False,
        *args,
        **kwargs
    ):
        super(ConvNeXt, self).__init__()
        assert variant in _VARIANTS
        assert set(out_stages).issubset((1, 2, 3, 4))

        model_fn, weights_enum, stage_channels = _VARIANTS[variant]
        if _HAS_TV_WEIGHTS and pretrain:
            backbone = model_fn(weights=weights_enum.IMAGENET1K_V1)
        else:
            try:
                backbone = model_fn(pretrained=pretrain)
            except TypeError:
                backbone = model_fn(weights=None)

        self.out_stages = out_stages
        self.out_channels = stage_channels
        self.features = backbone.features
        self.disable_cudnn_benchmark = disable_cudnn_benchmark
        self.use_channels_last = use_channels_last
        if len(self.features) < 8:
            raise ValueError("Unexpected ConvNeXt features layout.")

        # TorchVision ConvNeXt layout: 0 stem, 1 stage1, 2 down1, 3 stage2,
        # 4 down2, 5 stage3, 6 down3, 7 stage4.
        self._stage_feature_indices = {1: 1, 2: 3, 3: 5, 4: 7}
        self._feature_to_stage = {v: k for k, v in self._stage_feature_indices.items()}

    def forward(self, x):
        if self.use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        prev_cudnn_benchmark = None
        if self.disable_cudnn_benchmark and torch.backends.cudnn.is_available():
            prev_cudnn_benchmark = torch.backends.cudnn.benchmark
            torch.backends.cudnn.benchmark = False

        outputs = []
        try:
            for idx, block in enumerate(self.features):
                x = block(x)
                stage = self._feature_to_stage.get(idx)
                if stage is not None and stage in self.out_stages:
                    outputs.append(x)
        finally:
            if prev_cudnn_benchmark is not None:
                torch.backends.cudnn.benchmark = prev_cudnn_benchmark
        return tuple(outputs)


if __name__ == "__main__":
    # Test the ConvNeXt backbone
    model = ConvNeXt(variant="small", out_stages=(2, 3, 4), pretrain=False)
    
    test_input = torch.randn(1, 3, 256, 256)
    outputs = model(test_input)
    print("Output feature shapes:")
    for i, output in enumerate(outputs):
        print(f"Stage {i + 1}: {output.shape}")