import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Union, Tuple, Optional


def sample(inps: Union[Tuple[torch.Tensor, ...], torch.Tensor], size: int):
    if isinstance(inps, torch.Tensor):
        assert len(inps.shape) == 3 and inps.shape[0] == 1
        inps = inps.squeeze(0).cpu()  # (seq_length, hidden_size)
        size = min(inps.shape[0], size)
        indices = np.random.choice(inps.shape[0], size, replace=False)
        indices = torch.from_numpy(indices)
        return inps[indices]
    else:
        return tuple(sample(x, size) for x in inps)


class Mask(nn.Module):
    min_s = -0.1
    max_s = 1.1
    eps = 1e-6
    magical_number = 0.8
    
    def __init__(self, features: int, repeat: int = 1) -> None:
        super().__init__()
        self.activate = nn.Parameter(torch.tensor(True), requires_grad=False)
        self.features = features * repeat
        self.repeat = repeat
        self.beta = 2. / 3.
        self.log_alpha = nn.Parameter(torch.zeros((features,)))
        self.sampler = torch.distributions.Uniform(self.eps, 1. - self.eps)
        self.set_params(10.0)
    
    def set_state(self, activate: bool):
        self.activate.copy_(activate)
    
    @torch.no_grad()
    def set_params(self, mean: float, indices: Optional[torch.LongTensor] = None):  # [-10, 10]
        if indices is None:
            self.log_alpha.normal_(mean=mean, std=1e-2)
        else:
            self.log_alpha[indices].normal_(mean=mean, std=1e-2)
    
    def L(self):
        log_alpha = self.log_alpha.repeat(self.repeat)
        x = (0 - self.min_s) / (self.max_s - self.min_s)
        logits = math.log(x) - math.log(1 - x)
        L = torch.sigmoid(log_alpha - logits * self.beta).clamp(min=self.eps, max=1-self.eps)
        if not self.activate.item():
            L = L.detach()
        return L

    def sample_z(self):  # z -> mask
        log_alpha = self.log_alpha.repeat(self.repeat)
        u = self.sampler.sample((self.features,)).type_as(log_alpha)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / self.beta)
        s_bar = s * (self.max_s - self.min_s) + self.min_s
        z = F.hardtanh(s_bar, min_val=0, max_val=1)
        return z
    
    def deterministic_z(self):
        sub_features = self.features // self.repeat
        Lc = self.L().sum() / self.repeat
        num_zeros = round(sub_features - Lc.item()) * self.repeat
        log_alpha = self.log_alpha.repeat(self.repeat)
        z = torch.sigmoid(log_alpha / self.beta * self.magical_number)
        if num_zeros > 0:
            _, indices = torch.topk(z, k=num_zeros, largest=False)
            z[indices] = 0
        return z
    
    def forward(self):
        if self.activate.item():
            if self.training:
                return self.sample_z()
            else:
                return self.deterministic_z()
        else:
            return self.deterministic_z().detach()

    @torch.no_grad()
    def parse(self):
        sub_features = self.features // self.repeat
        Lc = self.L().sum() / self.repeat
        num_zeros = round(sub_features - Lc.item())
        num_non_zeros = sub_features - num_zeros
        z = torch.sigmoid(self.log_alpha / self.beta * self.magical_number)
        
        if num_non_zeros < 1:
            num_non_zeros = 1

        indices = torch.topk(z, k=num_non_zeros).indices
        indices = torch.sort(indices).values
        if self.repeat > 1:  # shape: (num_heads, head_dim)
            z = z.repeat(self.repeat)
            indices = torch.concat(tuple(indices + i * sub_features for i in range(self.repeat)))
        return z[indices], indices





class LinearWithMaskBefore(nn.Linear):
    
    def __init__(self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        device=None, 
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.mask = Mask(in_features)
        self.kw_args = {
            "in_features": in_features,
            "out_features": out_features,
            "bias": bias,
            "device": device, 
            "dtype": dtype,
        }

    def super_forward(self, x: Tensor) -> Tensor:
        return super().forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x * self.mask()
        x = super().forward(x)
        return x

    def get_hook_fn(self, act_dict, name):
        def fn(module, inp, outp):
            assert len(inp[0].shape) == 3 and inp[0].shape[0] == 1
            inp: Tensor = inp[0].squeeze(0).cpu()
            act_values = inp.abs().max(dim=0).values
            filter_weights = module.weight.norm(dim=0)
            values = act_values * filter_weights
            if name not in act_dict:
                act_dict[name] = values
            else:
                act_dict[name] = torch.max(act_dict[name], values)            
        return fn

    def extract(self,
        indices: torch.Tensor,
        values: Optional[torch.Tensor] = None,
    ) -> nn.Linear:
        if values is None:
            values = torch.ones_like(indices, dtype=self.weight.dtype)
        values = torch.diag(values)

        self.kw_args["in_features"] = indices.shape[0]
        new_linear = nn.Linear(**self.kw_args)        
        new_linear.weight.copy_(self.weight[:, indices] @ values)
        if new_linear.bias is not None:
            new_linear.bias.copy_(self.bias)        
        return new_linear