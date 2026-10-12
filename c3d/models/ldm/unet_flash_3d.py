# unet.py
from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# ============================================================
# fp16_util.py (inlined)
# ============================================================

def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def get_param_groups(model: nn.Module):
    """Return parameter groups separated by fp16/fp32 (used in some optim wrappers)."""
    fp16_params, fp32_params = [], []
    for p in model.parameters():
        (fp16_params if p.dtype == th.float16 else fp32_params).append(p)
    return fp16_params, fp32_params


class MixedPrecisionTrainer:
    """
    Minimal mixed precision helper (kept for compatibility with unet_latent.py style).
    Not required to use the UNet itself.
    """

    def __init__(self, model: nn.Module, use_fp16: bool = False, fp16_scale_growth: float = 1e-3):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.loss_scale = 2.0 ** 15 if use_fp16 else 1.0

        if use_fp16:
            # Keep master params in fp32
            self.fp16_params, self.fp32_params = get_param_groups(model)
            self.master_params = [p.detach().float().clone().requires_grad_(True) for p in self.fp16_params]
        else:
            self.fp16_params, self.fp32_params = [], []
            self.master_params = []

    def master_params_to_model_params(self):
        if not self.use_fp16:
            return
        for p_model, p_master in zip(self.fp16_params, self.master_params):
            p_model.data.copy_(p_master.data)

    def model_params_to_master_params(self):
        if not self.use_fp16:
            return
        for p_model, p_master in zip(self.fp16_params, self.master_params):
            p_master.data.copy_(p_model.data)

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def backward(self, loss: th.Tensor):
        if self.use_fp16:
            (loss * self.loss_scale).backward()
        else:
            loss.backward()

    def optimize(self, opt: th.optim.Optimizer):
        if not self.use_fp16:
            opt.step()
            return True

        # Copy grads from model fp16 params to master fp32 params
        grads = []
        for p in self.fp16_params:
            if p.grad is None:
                grads.append(th.zeros_like(p.data, dtype=th.float32))
            else:
                grads.append(p.grad.detach().float())
        flat_grads = _flatten_dense_tensors(grads)
        if th.isfinite(flat_grads).all():
            # unscale
            flat_grads.div_(self.loss_scale)
            grads = _unflatten_dense_tensors(flat_grads, grads)
            for p_master, g in zip(self.master_params, grads):
                p_master.grad = g
            opt.step()
            self.master_params_to_model_params()
            # scale growth
            self.loss_scale *= (1.0 + self.fp16_scale_growth)
            return True
        else:
            # scale backoff
            self.loss_scale = max(self.loss_scale / 2.0, 1.0)
            self.zero_grad()
            return False


# ============================================================
# nn.py (inlined)
# ============================================================

class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims: int, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module.
    Note: even when the UNet is 2D/3D, attention uses 1D convs internally.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dims={dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims: int, *args, **kwargs):
    """Create a 1D, 2D, or 3D average pooling module."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dims={dims}")


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels: int):
    """GroupNorm with a safe number of groups."""
    max_groups = min(32, channels)
    # pick the largest divisor of channels not exceeding 32
    groups = 1
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            groups = g
            break
    return GroupNorm32(groups, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


# ============================================================
# UNet building blocks
# ============================================================

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        ...


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    Upsampling with nearest-neighbor + optional conv.
    For dims=3:
      - if resample_3d=True: upsamples (D, H, W) by 2.
      - else: upsamples (H, W) by 2 and keeps D unchanged (video-style).
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_3d: bool = True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.resample_3d = resample_3d
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            if self.resample_3d:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            else:
                x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Downsampling with stride-2 conv or avg-pool.
    For dims=3:
      - if resample_3d=True: downsamples (D, H, W) by 2.
      - else: downsamples (H, W) by 2 and keeps D unchanged (video-style).
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_3d: bool = True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.resample_3d = resample_3d
        if dims == 3:
            stride = (2, 2, 2) if resample_3d else (1, 2, 2)
        else:
            stride = 2
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        resample_3d: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.updown = up or down

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        if up:
            self.h_upd = Upsample(channels, False, dims, resample_3d=resample_3d)
            self.x_upd = Upsample(channels, False, dims, resample_3d=resample_3d)
        elif down:
            self.h_upd = Downsample(channels, False, dims, resample_3d=resample_3d)
            self.x_upd = Downsample(channels, False, dims, resample_3d=resample_3d)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        # If you want checkpointing, replace with:
        # return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)
        return self._forward(x, emb)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)



class QKVFlashAttention(nn.Module):
    """
    Flash/SDPA attention backend.

    Uses torch.nn.functional.scaled_dot_product_attention.
    On CUDA with fp16/bf16 and supported shapes, PyTorch dispatches to FlashAttention.
    Otherwise it falls back to the best available SDPA backend.
    No parameters are added, so old checkpoints remain compatible.
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)

        # qkv: [B, 3*H*C, L] -> q/k/v: [B, H, L, C]
        qkv = qkv.reshape(bs, self.n_heads, 3 * ch, length)
        q, k, v = qkv.split(ch, dim=2)

        q = q.transpose(-2, -1).contiguous()
        k = k.transpose(-2, -1).contiguous()
        v = v.transpose(-2, -1).contiguous()

        # SDPA uses scale=1/sqrt(ch), equivalent to the old double-sqrt scaling.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )  # [B, H, L, C]

        out = out.transpose(-2, -1).contiguous()  # [B, H, C, L]
        return out.reshape(bs, self.n_heads * ch, length)

class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        # Flash/SDPA attention; keeps architecture/checkpoints unchanged.
        self.attention = QKVFlashAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        # return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
        return self._forward(x)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x_in = x
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial) + 0.0 * x_in  # keep shape


# ============================================================
# Config + UNet
# ============================================================

@dataclass
class UnetConfig:
    """
    Configuration for a UNet.
    """
    image_size: int = 128
    in_channels: int = 64
    out_channels: int = 64

    model_channels: int = 64
    num_res_blocks: int = 4
    channel_mult: Tuple[int, ...] = (1, 2, 2, 2, 4)
    attention_resolutions: Tuple[int, ...] = (16,)

    dropout: float = 0.0
    conv_resample: bool = True
    dims: int = 3  # 2 or 3
    resample_3d: bool = True  # only relevant when dims=3

    use_fp16: bool = False
    use_checkpoint: bool = False

    num_heads: int = 4
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = True
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    decoder_skip: str = "add"
    skip_scale: float = 0.5


class UNetModel(nn.Module):
    """
    UNet for diffusion/flow models.

    This is the same UNet structure as unet_latent.py, with:
    - dims restricted to {2,3}
    - correct volumetric resampling for dims=3 via `resample_3d`
    - FIX: output conv uses current `ch` (not `input_ch`)
    """

    def __init__(
        self,
        image_size: int = 128,
        in_channels: int = 64,
        model_channels: int = 64,
        out_channels: int = 64,
        num_res_blocks: int = 4,
        attention_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.0,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 2, 4),
        conv_resample: bool = True,
        dims: int = 3,
        resample_3d: bool = True,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 4,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        decoder_skip: str = "add",
        skip_scale: float = 0.5,
        config: Optional[UnetConfig] = None,
        **kwargs,
    ):
        super().__init__()

        if config is not None:
            image_size = config.image_size
            in_channels = config.in_channels
            out_channels = config.out_channels
            model_channels = config.model_channels
            num_res_blocks = config.num_res_blocks
            attention_resolutions = config.attention_resolutions
            dropout = config.dropout
            channel_mult = config.channel_mult
            conv_resample = config.conv_resample
            dims = config.dims
            resample_3d = config.resample_3d
            use_checkpoint = config.use_checkpoint
            use_fp16 = config.use_fp16
            num_heads = config.num_heads
            num_head_channels = config.num_head_channels
            num_heads_upsample = config.num_heads_upsample
            use_scale_shift_norm = config.use_scale_shift_norm
            resblock_updown = config.resblock_updown
            use_new_attention_order = config.use_new_attention_order
            decoder_skip = config.decoder_skip
            skip_scale = config.skip_scale

        assert dims in (2, 3), "UNetModel supports dims=2 or dims=3 only."
        if decoder_skip not in ("add", "concat"):
            raise ValueError("decoder_skip must be 'add' or 'concat'")
        if not math.isfinite(skip_scale) or skip_scale <= 0:
            raise ValueError("skip_scale must be finite and positive")
        self.decoder_skip = decoder_skip
        self.skip_scale = float(skip_scale)

        self.config = config
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads if num_heads_upsample == -1 else num_heads_upsample
        self.resample_3d = resample_3d

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(self.num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        resample_3d=resample_3d,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            resample_3d=resample_3d,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch, resample_3d=resample_3d)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                resample_3d=resample_3d,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                resample_3d=resample_3d,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        self.skip_projections = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                # Keep the original per-block skip topology. Addition needs
                # equal channels; concatenation retains the original widths.
                self.skip_projections.append(
                    conv_nd(dims, ich, ch, 1) if decoder_skip == "add" and ich != ch else nn.Identity()
                )
                layers = [
                    ResBlock(
                        ch if decoder_skip == "add" else ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        resample_3d=resample_3d,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            resample_3d=resample_3d,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, resample_3d=resample_3d)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # FIX compared to your pasted code: use current `ch`, not `input_ch`.
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.skip_projections.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        self.skip_projections.apply(convert_module_to_f32)

    def forward(self, x: th.Tensor, timesteps: th.Tensor, y: Optional[th.Tensor] = None) -> th.Tensor:
        assert (y is not None) == (self.num_classes is not None), "must specify y iff class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)

        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module, projection in zip(self.output_blocks, self.skip_projections):
            skip = hs.pop()
            if h.shape[2:] != skip.shape[2:]:
                raise ValueError("Decoder/skip spatial shapes differ; use compatible latent dimensions")
            h = (h + projection(skip)) * self.skip_scale if self.decoder_skip == "add" else th.cat([h, skip], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        return self.out(h)


# ============================================================
# __main__ test (3D latents)
# ============================================================

if __name__ == "__main__":
    # Test on (B, 4, Z, 64, 64)
    # NOTE: pick a small Z here to avoid OOM in a quick unit test.
    # In training you can use your real Z (256/400/512) with batch_size=1.
    Z = 400

    model = UNetModel(
        image_size=64,
        in_channels=4,
        out_channels=4,
        model_channels=64,
        num_res_blocks=2,
        channel_mult=(1, 2, 3, 4),
        attention_resolutions=(8, 16),  # ds factors, i.e. spatial 8x8 and 4x4 for 64x64 input
        dims=3,
        resample_3d=True,
        num_heads=4,
        use_scale_shift_norm=True,
    )

    x = th.randn(1, 4, Z, 64, 64)
    t = th.randint(low=0, high=1000, size=(1,), dtype=th.long)
    out = model(x, t)

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(out.shape))
