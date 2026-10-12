#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn as nn
from c3d.models.checkpointing import block


# -----------------------------------------------------------------------------
# Device helpers
# -----------------------------------------------------------------------------
def resolve_device(device_arg: str) -> torch.device:
    s = str(device_arg).lower()
    if s in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def autocast_if_cuda(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()


# -----------------------------------------------------------------------------
# ND Factory
# -----------------------------------------------------------------------------
class NDOps:
    """
    Small factory that abstracts Conv / Norm / shape handling for 2D and 3D.
    """
    def __init__(self, dim: int):
        assert dim in (2, 3)
        self.dim = dim

        if dim == 2:
            self.Conv = nn.Conv2d
            self.BatchNorm = nn.BatchNorm2d
            self.InstanceNorm = nn.InstanceNorm2d
        else:
            self.Conv = nn.Conv3d
            self.BatchNorm = nn.BatchNorm3d
            self.InstanceNorm = nn.InstanceNorm3d

    def conv_k(self, in_ch, out_ch, k, stride=1, padding=0, groups=1):
        return self.Conv(
            in_ch, out_ch,
            kernel_size=k,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )

    def conv3(self, in_ch, out_ch, stride=1, groups=1):
        return self.conv_k(in_ch, out_ch, k=3, stride=stride, padding=1, groups=groups)

    def conv1(self, in_ch, out_ch, stride=1):
        return self.conv_k(in_ch, out_ch, k=1, stride=stride, padding=0)

    def make_norm_layer(self, norm: str, gn_groups: int):
        norm = norm.lower()
        if norm == "bn":
            return self.BatchNorm
        if norm == "in":
            return self.InstanceNorm
        if norm == "gn":
            def _gn(ch):
                g = min(int(gn_groups), int(ch))
                while ch % g != 0 and g > 1:
                    g -= 1
                return nn.GroupNorm(g, ch)
            return _gn
        raise ValueError("norm must be bn | in | gn")


# -----------------------------------------------------------------------------
# Activations
# -----------------------------------------------------------------------------
def make_act(act: str):
    act = act.lower()
    if act == "silu":
        return nn.SiLU(inplace=True)
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    raise ValueError("act must be silu | relu | gelu")


# -----------------------------------------------------------------------------
# Blocks
# -----------------------------------------------------------------------------
class ConvNormActND(nn.Module):
    def __init__(self, ops: NDOps, in_ch, out_ch, stride, norm_layer, act="silu"):
        super().__init__()
        self.conv = ops.conv3(in_ch, out_ch, stride=stride)
        self.norm = norm_layer(out_ch)
        self.act = make_act(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResLiteND(nn.Module):
    def __init__(self, ops: NDOps, ch, norm_layer, act="silu"):
        super().__init__()
        self.dw = ops.conv3(ch, ch, stride=1, groups=ch)
        self.n1 = norm_layer(ch)
        self.pw = ops.conv1(ch, ch)
        self.n2 = norm_layer(ch)
        self.act = make_act(act)

    def forward(self, x):
        y = self.act(self.n1(self.dw(x)))
        y = self.n2(self.pw(y))
        return self.act(x + y)


class ChannelMixerND(nn.Module):
    def __init__(self, ops: NDOps, ch, norm_layer, expansion=8, act="silu"):
        super().__init__()
        hidden = ch * expansion
        self.n0 = norm_layer(ch)
        self.fc1 = ops.conv1(ch, hidden)
        self.act = make_act(act)
        self.fc2 = ops.conv1(hidden, ch)

    def forward(self, x):
        y = self.fc1(self.n0(x))
        y = self.act(y)
        y = self.fc2(y)
        return x + y


class LatentAffineNormND(nn.Module):
    def __init__(self, dim: int, ch: int):
        super().__init__()
        shape = (1, ch) + (1,) * dim
        self.scale = nn.Parameter(torch.ones(*shape))
        self.bias = nn.Parameter(torch.zeros(*shape))

    def forward(self, x):
        return (x + self.bias) * self.scale


# -----------------------------------------------------------------------------
# ResNet ND
# -----------------------------------------------------------------------------
class BasicBlockND(nn.Module):
    expansion = 1

    def __init__(self, ops: NDOps, in_ch, out_ch, stride=1, norm_layer=None):
        super().__init__()
        self.conv1 = ops.conv3(in_ch, out_ch, stride=stride)
        self.bn1 = norm_layer(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = ops.conv3(out_ch, out_ch)
        self.bn2 = norm_layer(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                ops.conv1(in_ch, out_ch, stride=stride),
                norm_layer(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)


class ResNetND(nn.Module):
    def __init__(
        self,
        ops: NDOps,
        in_channels,
        layers=(2, 2),
        channels=(64, 64),
        strides=(1, 1),
        norm_layer=None,
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        cur = in_channels

        for n, ch, st in zip(layers, channels, strides):
            blocks = [BasicBlockND(ops, cur, ch, st, norm_layer)]
            cur = ch
            for _ in range(1, n):
                blocks.append(BasicBlockND(ops, cur, ch, 1, norm_layer))
            self.stages.append(nn.Sequential(*blocks))

        self.out_channels = cur

    def forward(self, x, checkpoint_blocks=False):
        for s in self.stages:
            for layer in s:x = block(layer,x,checkpoint_blocks)
        return x


# -----------------------------------------------------------------------------
# Encoder ND
# -----------------------------------------------------------------------------
class StridedEncND(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int = 1,
        base_ch: int = 8,
        latent_ch: int = 16,
        norm: str = "gn",
        gn_groups: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.ops = NDOps(dim)
        norm_layer = self.ops.make_norm_layer(norm, gn_groups)

        self.down = nn.Sequential(
            ResLiteND(self.ops, in_channels, norm_layer),
            ConvNormActND(self.ops, in_channels, base_ch, 2, norm_layer),
            ResLiteND(self.ops, base_ch, norm_layer),
            ConvNormActND(self.ops, base_ch, base_ch * 8, 2, norm_layer),
        )

        self.backbone = ResNetND(
            self.ops,
            in_channels=base_ch * 8,
            layers=(2, 2),
            channels=(base_ch * 8, base_ch * 8),
            strides=(1, 1),
            norm_layer=norm_layer,
        )
        feat_ch = self.backbone.out_channels

        self.mixer = ChannelMixerND(self.ops, feat_ch, norm_layer)
        self.affine = LatentAffineNormND(dim, feat_ch)

        self.mu_proj = self.ops.conv1(feat_ch, latent_ch)
        self.logvar_proj = self.ops.conv1(feat_ch, latent_ch)

    def forward(self, x, checkpoint_blocks=False):
        for layer in self.down:x = block(layer,x,checkpoint_blocks)
        x = self.backbone(x,checkpoint_blocks=checkpoint_blocks)
        x = block(self.mixer,x,checkpoint_blocks)
        x = block(self.affine,x,checkpoint_blocks)
        return block(self.mu_proj,x,checkpoint_blocks), block(self.logvar_proj,x,checkpoint_blocks)


# -----------------------------------------------------------------------------
# Model inspection utilities
# -----------------------------------------------------------------------------
def tensor_mem_mb(t: torch.Tensor):
    return t.numel() * t.element_size() / 1024**2


def register_hooks(model):
    rows = []
    hooks = []

    def hook_fn(name):
        def hook(m, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            inp = inp[0]
            rows.append({
                "name": name,
                "in_shape": tuple(inp.shape),
                "out_shape": tuple(out.shape),
                "act_mem": tensor_mem_mb(out),
                "param_mem": sum(
                    p.numel() * p.element_size() for p in m.parameters()
                ) / 1024**2
            })
        return hook

    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(hook_fn(name)))

    return rows, hooks


def print_report(rows):
    print("\n================ Layer Report ================")
    total_act = 0.0
    total_param = 0.0

    for r in rows:
        total_act += r["act_mem"]
        total_param += r["param_mem"]
        print(
            f"{r['name']:<40} "
            f"{str(r['in_shape']):>18} -> {str(r['out_shape']):>18} | "
            f"Act {r['act_mem']:7.2f} MB | "
            f"Param {r['param_mem']:7.2f} MB"
        )

    print("------------------------------------------------")
    print(f"Total activation memory : {total_act:.2f} MB")
    print(f"Total parameter memory  : {total_param:.2f} MB")
    print("================================================\n")


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------
def run_experiment(args):
    print(f"\n========== Running {args.dim}D Experiment ==========")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model = StridedEncND(
        dim=args.dim,
        in_channels=args.in_channels,
        base_ch=args.base_ch,
        latent_ch=args.latent_ch,
        norm=args.norm,
        gn_groups=args.gn_groups,
    ).to(device)

    if device.type == "cuda":
        model = model.half()

    spatial = (args.input_size,) * args.dim
    x = torch.randn(
        args.batch,
        args.in_channels,
        *spatial,
        device=device,
    )

    if device.type == "cuda":
        x = x.half()

    rows, hooks = register_hooks(model)

    with torch.no_grad():
        with autocast_if_cuda(device):
            mu, logvar = model(x)

    for h in hooks:
        h.remove()

    print(f"Output mu shape     : {tuple(mu.shape)}")
    print(f"Output logvar shape : {tuple(logvar.shape)}")
    print_report(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    # Core
    ap.add_argument("--dim", type=int, default=3, choices=[2, 3])
    ap.add_argument("--device", type=str, default="auto")

    # Data
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--in-channels", type=int, default=1)

    # Model
    ap.add_argument("--base-ch", type=int, default=8)
    ap.add_argument("--latent-ch", type=int, default=16)
    ap.add_argument("--norm", type=str, default="gn", choices=["bn", "in", "gn"])
    ap.add_argument("--gn-groups", type=int, default=8)

    # Experiment
    ap.add_argument("--run-both", default=True, action="store_true")

    args = ap.parse_args()

    if args.run_both:
        for d in (2, 3):
            args.dim = d
            run_experiment(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
