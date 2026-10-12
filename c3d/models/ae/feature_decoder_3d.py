#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Optional
from contextlib import nullcontext

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
# ND ops factory (ONLY changes conv/norm dimensionality)
# -----------------------------------------------------------------------------
class NDOps:
    def __init__(self, dim: int):
        assert dim in (2, 3), "dim must be 2 or 3"
        self.dim = dim
        if dim == 2:
            self.Conv = nn.Conv2d
            self.BatchNorm = nn.BatchNorm2d
            self.InstanceNorm = nn.InstanceNorm2d
        else:
            self.Conv = nn.Conv3d
            self.BatchNorm = nn.BatchNorm3d
            self.InstanceNorm = nn.InstanceNorm3d


# -----------------------------------------------------------------------------
# Config (same semantics as your 3D file, with added `dim`)
# -----------------------------------------------------------------------------
@dataclass
class StageCfg:
    out_channels: int
    num_blocks: int
    stride: int = 1              # keep 1 to preserve spatial size


@dataclass
class DecCNNConfig:
    dim: int                     # NEW: 2 or 3

    in_channels: int
    stages: List[StageCfg]       # full control
    out_channels: Optional[int] = None  # if None -> last stage channels

    block: Literal["basic", "bottleneck"] = "basic"
    norm: Literal["gn", "bn", "in"] = "gn"
    gn_groups: int = 8
    act: Literal["silu", "relu", "gelu"] = "silu"

    dropout: float = 0.0
    drop_path_rate: float = 0.0

    in_proj: bool = True         # 1x1 to first stage channels
    out_proj: bool = False       # 1x1 to out_channels (if set)
    residual_scale: float = 1.0  # can stabilize deep stacks


# -----------------------------------------------------------------------------
# Building blocks (same math/structure as your 3D version; only ND conv/norm)
# -----------------------------------------------------------------------------
def _make_norm(ops: NDOps, norm: str, ch: int, gn_groups: int) -> nn.Module:
    norm = norm.lower()
    if norm == "bn":
        return ops.BatchNorm(ch)
    if norm == "in":
        return ops.InstanceNorm(ch, affine=True)
    if norm == "gn":
        g = min(int(gn_groups), int(ch))
        while ch % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, ch)
    raise ValueError("norm must be gn|bn|in")


def _make_act(act: str) -> nn.Module:
    act = act.lower()
    if act == "silu":
        return nn.SiLU(inplace=True)
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    raise ValueError("act must be silu|relu|gelu")


class DropPath(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, device=x.device) < keep).to(x.dtype)
        return x * mask / keep


class BasicResBlockND(nn.Module):
    def __init__(
        self,
        ops: NDOps,
        in_ch,
        out_ch,
        stride,
        norm,
        gn_groups,
        act,
        dropout,
        drop_path,
        residual_scale=1.0,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.dp = DropPath(drop_path)

        self.conv1 = ops.Conv(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.n1 = _make_norm(ops, norm, out_ch, gn_groups)
        self.a1 = _make_act(act)

        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = ops.Conv(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.n2 = _make_norm(ops, norm, out_ch, gn_groups)

        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                ops.Conv(in_ch, out_ch, 1, stride=stride, bias=False),
                _make_norm(ops, norm, out_ch, gn_groups),
            )
        self.a2 = _make_act(act)

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        h = self.a1(self.n1(self.conv1(x)))
        h = self.do(h)
        h = self.n2(self.conv2(h))
        h = identity + self.dp(h) * self.residual_scale
        return self.a2(h)


class BottleneckResBlockND(nn.Module):
    expansion = 4

    def __init__(
        self,
        ops: NDOps,
        in_ch,
        out_ch,
        stride,
        norm,
        gn_groups,
        act,
        dropout,
        drop_path,
        residual_scale=1.0,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.dp = DropPath(drop_path)

        mid = out_ch
        out_exp = out_ch * self.expansion

        self.conv1 = ops.Conv(in_ch, mid, 1, bias=False)
        self.n1 = _make_norm(ops, norm, mid, gn_groups)
        self.a1 = _make_act(act)

        self.conv2 = ops.Conv(mid, mid, 3, stride=stride, padding=1, bias=False)
        self.n2 = _make_norm(ops, norm, mid, gn_groups)
        self.a2 = _make_act(act)

        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv3 = ops.Conv(mid, out_exp, 1, bias=False)
        self.n3 = _make_norm(ops, norm, out_exp, gn_groups)

        self.skip = None
        if stride != 1 or in_ch != out_exp:
            self.skip = nn.Sequential(
                ops.Conv(in_ch, out_exp, 1, stride=stride, bias=False),
                _make_norm(ops, norm, out_exp, gn_groups),
            )
        self.a3 = _make_act(act)

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        h = self.a1(self.n1(self.conv1(x)))
        h = self.a2(self.n2(self.conv2(h)))
        h = self.do(h)
        h = self.n3(self.conv3(h))
        h = identity + self.dp(h) * self.residual_scale
        return self.a3(h)


# -----------------------------------------------------------------------------
# Decoder backbone (ND)
# -----------------------------------------------------------------------------
class DecCNN(nn.Module):
    def __init__(self, cfg: DecCNNConfig):
        super().__init__()
        self.cfg = cfg
        assert len(cfg.stages) > 0

        ops = NDOps(cfg.dim)
        self.ops = ops

        if cfg.block == "basic":
            block_cls = BasicResBlockND
            out_mult = 1
        else:
            block_cls = BottleneckResBlockND
            out_mult = BottleneckResBlockND.expansion

        first_out = cfg.stages[0].out_channels
        self.in_proj = ops.Conv(cfg.in_channels, first_out, 1, bias=False) if cfg.in_proj else nn.Identity()

        total_blocks = sum(s.num_blocks for s in cfg.stages)
        dps = [cfg.drop_path_rate * i / max(1, total_blocks - 1) for i in range(total_blocks)]

        layers: List[nn.Module] = []
        cur = first_out
        dp_i = 0

        for s in cfg.stages:
            for b in range(s.num_blocks):
                stride = s.stride if b == 0 else 1
                layers.append(
                    block_cls(
                        ops=ops,
                        in_ch=cur,
                        out_ch=s.out_channels,
                        stride=stride,
                        norm=cfg.norm,
                        gn_groups=cfg.gn_groups,
                        act=cfg.act,
                        dropout=cfg.dropout,
                        drop_path=dps[dp_i],
                        residual_scale=cfg.residual_scale,
                    )
                )
                dp_i += 1
                cur = s.out_channels * out_mult

        self.net = nn.Sequential(*layers)

        out_ch = cfg.out_channels if cfg.out_channels is not None else cur
        self.out_proj = ops.Conv(cur, out_ch, 1, bias=False) if (cfg.out_proj or (out_ch != cur)) else nn.Identity()
        self.out_channels = out_ch

    def forward(self, z: torch.Tensor, checkpoint_blocks=False) -> torch.Tensor:
        h = block(self.in_proj,z,checkpoint_blocks)
        for layer in self.net:h = block(layer,h,checkpoint_blocks)
        h = block(self.out_proj,h,checkpoint_blocks)
        # GroupNorm/residual promotion can leave the coarse field in FP32.
        # Store that field in the active AMP dtype before all head gathers.
        if torch.is_autocast_enabled(h.device.type):
            h=h.to(torch.get_autocast_dtype(h.device.type))
        return h


# -----------------------------------------------------------------------------
# Testing / main
# -----------------------------------------------------------------------------
def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",")]


def build_cfg_from_args(args, dim: int) -> DecCNNConfig:
    stage_out = _parse_int_list(args.cnn_stage_out)
    stage_blocks = _parse_int_list(args.cnn_stage_blocks)
    stage_strides = _parse_int_list(args.cnn_stage_strides)

    if not (len(stage_out) == len(stage_blocks) == len(stage_strides)):
        raise ValueError(
            f"Stage lists must match lengths: "
            f"out({len(stage_out)}), blocks({len(stage_blocks)}), strides({len(stage_strides)})"
        )

    stages = [StageCfg(o, b, st) for o, b, st in zip(stage_out, stage_blocks, stage_strides)]

    out_channels = None if args.cnn_out_channels == -1 else int(args.cnn_out_channels)

    return DecCNNConfig(
        dim=dim,
        in_channels=args.in_channels,
        stages=stages,
        out_channels=out_channels,
        block=args.cnn_block,
        norm=args.cnn_norm,
        gn_groups=args.cnn_gn_groups,
        act=args.cnn_act,
        dropout=args.cnn_dropout,
        drop_path_rate=args.cnn_drop_path,
        in_proj=args.cnn_in_proj,
        out_proj=args.cnn_out_proj,
        residual_scale=args.cnn_residual_scale,
    )


def run_test(args, dim: int):
    device = resolve_device(args.device)
    cfg = build_cfg_from_args(args, dim=dim)
    model = DecCNN(cfg).to(device)

    if device.type == "cuda" and args.fp16:
        model = model.half()

    spatial = (args.spatial,) * dim
    z = torch.randn(args.batch, args.in_channels, *spatial, device=device)

    if device.type == "cuda" and args.fp16:
        z = z.half()

    with torch.no_grad():
        with autocast_if_cuda(device):
            y = model(z)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n===== DecCNN test: dim={dim} =====")
    print(f"device            : {device}  (fp16={bool(device.type=='cuda' and args.fp16)})")
    print(f"input  z shape    : {tuple(z.shape)}")
    print(f"output y shape    : {tuple(y.shape)}")
    print(f"model.out_channels: {model.out_channels}")
    print(f"#params           : {n_params:,}")


def main():
    import argparse

    ap = argparse.ArgumentParser("DecCNN ND test")

    # run control
    ap.add_argument("--dim", type=int, default=3, choices=[2, 3], help="Run a single dim (2 or 3)")
    ap.add_argument("--run-both", default=True, action="store_true", help="Run both 2D and 3D tests")

    # device / precision
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--fp16", action="store_true", help="Use fp16 on CUDA")

    # input
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--spatial", type=int, default=64, help="H=W(=D for 3D) of latent grid")
    ap.add_argument("--in-channels", dest="in_channels", type=int, default=16)

    # your existing knobs (kept same meaning)
    ap.add_argument("--cnn-stage-out", type=str, default="48,48")
    ap.add_argument("--cnn-stage-blocks", type=str, default="2,2")
    ap.add_argument("--cnn-stage-strides", type=str, default="1,1")

    ap.add_argument("--cnn-out-channels", type=int, default=-1, help="-1 = use last stage channels")
    ap.add_argument("--cnn-block", type=str, default="basic", choices=["basic", "bottleneck"])
    ap.add_argument("--cnn-norm", type=str, default="gn", choices=["gn", "bn", "in"])
    ap.add_argument("--cnn-gn-groups", type=int, default=8)
    ap.add_argument("--cnn-act", type=str, default="silu", choices=["silu", "relu", "gelu"])
    ap.add_argument("--cnn-dropout", type=float, default=0.0)
    ap.add_argument("--cnn-drop-path", type=float, default=0.0)
    ap.add_argument("--cnn-in-proj", action="store_true", default=True)
    ap.add_argument("--cnn-out-proj", action="store_true", default=False)
    ap.add_argument("--cnn-residual-scale", type=float, default=1.0)

    args = ap.parse_args()

    if args.run_both:
        run_test(args, dim=2)
        run_test(args, dim=3)
    else:
        run_test(args, dim=args.dim)


if __name__ == "__main__":
    main()
