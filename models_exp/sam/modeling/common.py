# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type

# add by aaron
class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class LoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True, lora_layer=False, multi_branch=False):
        super().__init__()
        self.skip_connect = skip_connect
        self.multi_branch = multi_branch
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        if lora_layer:
            self.D_fc1 = LoRALinear(D_features, D_hidden_features, r=16)
            self.D_fc2 = LoRALinear(D_hidden_features, D_features, r=16)
        elif multi_branch:
            self.n_branch = 8
            each_branch = int(D_features / self.n_branch)
            D_hidden_features = int(each_branch * mlp_ratio)
            self.D_braunch1_fc1 = nn.Linear(each_branch, D_hidden_features)
            self.D_braunch2_fc1 = nn.Linear(each_branch, D_hidden_features)
            self.D_braunch3_fc1 = nn.Linear(each_branch, D_hidden_features)
            self.D_braunch4_fc1 = nn.Linear(each_branch, D_hidden_features)
            self.D_braunch5_fc1 = nn.Linear(each_branch, D_hidden_features)
            self.D_braunch6_fc1 = nn.Linear(each_branch, D_hidden_features)
            self.D_braunch7_fc1 = nn.Linear(each_branch, D_hidden_features)
            self.D_braunch8_fc1 = nn.Linear(each_branch, D_hidden_features)

            self.D_braunch1_fc2 = nn.Linear(D_hidden_features, each_branch)
            self.D_braunch2_fc2 = nn.Linear(D_hidden_features, each_branch)
            self.D_braunch3_fc2 = nn.Linear(D_hidden_features, each_branch)
            self.D_braunch4_fc2 = nn.Linear(D_hidden_features, each_branch)
            self.D_braunch5_fc2 = nn.Linear(D_hidden_features, each_branch)
            self.D_braunch6_fc2 = nn.Linear(D_hidden_features, each_branch)
            self.D_braunch7_fc2 = nn.Linear(D_hidden_features, each_branch)
            self.D_braunch8_fc2 = nn.Linear(D_hidden_features, each_branch)
        else:
            self.D_fc1 = nn.Linear(D_features, D_hidden_features)
            self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def block(self, x, fc1, fc2, skip_connect):
        xs = fc1(x)
        xs = self.act(xs)
        xs = fc2(xs)
        if skip_connect:
            x = x + xs
        else:
            x = xs
        return x
        
    def forward(self, x):
        # x is (BT, HW+1, D)  (1, 64, 64, 768)
        if self.multi_branch:
            x_b = torch.chunk(x, self.n_branch, dim=-1)
            x1 = self.block(x_b[0], self.D_braunch1_fc1, self.D_braunch1_fc2, self.skip_connect)
            x2 = self.block(x_b[1], self.D_braunch2_fc1, self.D_braunch2_fc2, self.skip_connect)
            x3 = self.block(x_b[2], self.D_braunch3_fc1, self.D_braunch3_fc2, self.skip_connect)
            x4 = self.block(x_b[3], self.D_braunch4_fc1, self.D_braunch4_fc2, self.skip_connect)
            x5 = self.block(x_b[4], self.D_braunch5_fc1, self.D_braunch5_fc2, self.skip_connect)
            x6 = self.block(x_b[5], self.D_braunch6_fc1, self.D_braunch6_fc2, self.skip_connect)
            x7 = self.block(x_b[6], self.D_braunch7_fc1, self.D_braunch7_fc2, self.skip_connect)
            x8 = self.block(x_b[7], self.D_braunch8_fc1, self.D_braunch8_fc2, self.skip_connect)

            return torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=-1)

        else:
            xs = self.D_fc1(x)
            xs = self.act(xs)
            xs = self.D_fc2(xs)
            if self.skip_connect:
                x = x + xs
            else:
                x = xs
            return x
    
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
