import torch
from torch import nn
from functools import partial

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


class W8A8Linear(nn.Module):
    # --- Start of feature modification ---
    def __init__(self, in_features, out_features, bias=True, act_quant='per_token', quantize_output=False, nbits=None, weight_nbits=None, act_nbits=None):
        super().__init__()
        # Support backward compatibility: if nbits is provided, use it for both weight and activation
        if nbits is not None and weight_nbits is None and act_nbits is None:
            self.weight_nbits = nbits
            self.act_nbits = nbits
        else:
            # Use separate bit widths if provided, otherwise default to 8
            self.weight_nbits = weight_nbits if weight_nbits is not None else 8
            self.act_nbits = act_nbits if act_nbits is not None else 8

        # Keep nbits for backward compatibility
        self.nbits = nbits if nbits is not None else self.act_nbits
        # --- End of feature modification ---

        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randn(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)

        # --- Start of feature modification ---
        if act_quant == 'per_token':
            self.act_quant_name = 'per_token'
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=self.act_nbits)
        elif act_quant == 'per_tensor':
            self.act_quant_name = 'per_tensor'
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=self.act_nbits)
        else:
            raise ValueError(f'Invalid act_quant: {act_quant}')
        # --- End of feature modification ---

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    # --- Start of feature modification ---
    def from_float(module, weight_quant='per_channel', act_quant='per_token', quantize_output=False, nbits=None, weight_nbits=None, act_nbits=None):
        assert isinstance(module, torch.nn.Linear)

        # Support backward compatibility: if nbits is provided, use it for both weight and activation
        if nbits is not None and weight_nbits is None and act_nbits is None:
            _weight_nbits = nbits
            _act_nbits = nbits
        else:
            # Use separate bit widths if provided, otherwise default to 8
            _weight_nbits = weight_nbits if weight_nbits is not None else 8
            _act_nbits = act_nbits if act_nbits is not None else 8

        new_module = W8A8Linear(
            module.in_features, module.out_features, module.bias is not None,
            act_quant=act_quant, quantize_output=quantize_output,
            nbits=nbits, weight_nbits=_weight_nbits, act_nbits=_act_nbits)

        if weight_quant == 'per_channel':
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=_weight_nbits)
        elif weight_quant == 'per_tensor':
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=_weight_nbits)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        # --- End of feature modification ---
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        # --- Start of feature modification ---
        return f'W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name}, weight_nbits={self.weight_nbits}, act_nbits={self.act_nbits})'
        # --- End of feature modification ---
