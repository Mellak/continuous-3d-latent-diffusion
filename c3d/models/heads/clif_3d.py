#!/usr/bin/env python3
"""Coordinate-conditioned local convolutional image function for 3-D volumes.

This is a 3-D adaptation of the CLIF renderer used by INFD.  The expensive
feature decoder remains on the coarse latent grid.  For a requested dense
output patch, the renderer:

1. samples the nearest coarse feature at every target coordinate;
2. appends relative coordinate and target-cell-size channels;
3. predicts the patch jointly with a small 3-D convolutional network.

Dense volumes are rendered tile by tile with a halo, so no full-resolution
multi-channel feature tensor is materialized.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .geometry import sample_fp32
from c3d.models.checkpointing import block,run_checkpoint


Spatial3D = Tuple[int, int, int]


def _triple(value: int | Sequence[int]) -> Spatial3D:
    if isinstance(value, int):
        return (int(value), int(value), int(value))
    values = tuple(int(v) for v in value)
    if len(values) != 3:
        raise ValueError("Expected three spatial values (D,H,W).")
    return values  # type: ignore[return-value]


class ChannelNorm3D(nn.Module):
    """Normalize channels independently at each voxel; no spatial/tile statistics."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            y = F.layer_norm(x.float().movedim(1, -1), (x.shape[1],), self.weight.float(), self.bias.float(), 1e-5)
        return y.movedim(-1, 1).to(x.dtype)


def _make_norm(kind: str, channels: int, groups: int) -> nn.Module:
    kind = str(kind).lower()
    if kind == "channel":
        return ChannelNorm3D(channels)
    if kind == "none":
        return nn.Identity()
    if kind == "in":
        return nn.InstanceNorm3d(channels, affine=True)
    if kind == "bn":
        return nn.BatchNorm3d(channels)
    if kind == "gn":
        g = min(int(groups), int(channels))
        while g > 1 and channels % g != 0:
            g -= 1
        return nn.GroupNorm(g, channels)
    raise ValueError("norm must be one of: gn, bn, in, none")


def _make_act(kind: str) -> nn.Module:
    kind = str(kind).lower()
    if kind == "silu":
        return nn.SiLU(inplace=True)
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "gelu":
        return nn.GELU()
    raise ValueError("act must be one of: silu, relu, gelu")


class CLIFResBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        norm: str = "gn",
        gn_groups: int = 8,
        act: str = "silu",
        dropout: float = 0.0,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd.")
        pad = kernel_size // 2
        self.norm1 = _make_norm(norm, channels, gn_groups)
        self.act1 = _make_act(act)
        self.conv1 = nn.Conv3d(
            channels,
            channels,
            kernel_size,
            padding=pad,
            padding_mode=padding_mode,
        )
        self.norm2 = _make_norm(norm, channels, gn_groups)
        self.act2 = _make_act(act)
        self.dropout = nn.Dropout3d(float(dropout)) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(
            channels,
            channels,
            kernel_size,
            padding=pad,
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return x + h


@dataclass
class CLIFHeadConfig:
    dim: int = 3
    out_channels: int = 1
    hidden_channels: int = 64
    num_res_blocks: int = 2
    kernel_size: int = 3
    norm: str = "gn"
    gn_groups: int = 8
    act: str = "silu"
    dropout: float = 0.0
    use_cell: bool = True
    align_corners: bool = False
    padding_mode: str = "border"
    conv_padding_mode: str = "replicate"
    tile_size: Spatial3D = (64, 64, 64)
    halo: int = 8


class CLIFHead3D(nn.Module):
    """3-D convolutional local image function."""

    def __init__(self, in_channels: int, cfg: CLIFHeadConfig) -> None:
        super().__init__()
        if int(cfg.dim) != 3:
            raise ValueError("CLIFHead3D only supports dim=3.")
        self.in_channels = int(in_channels)
        self.cfg = cfg

        layout_channels = self.in_channels + 3 + (3 if cfg.use_cell else 0)
        pad = int(cfg.kernel_size) // 2
        self.conv_in = nn.Conv3d(
            layout_channels,
            int(cfg.hidden_channels),
            int(cfg.kernel_size),
            padding=pad,
            padding_mode=cfg.conv_padding_mode,
        )
        self.blocks = nn.ModuleList(
            [
                CLIFResBlock3D(
                    int(cfg.hidden_channels),
                    kernel_size=int(cfg.kernel_size),
                    norm=cfg.norm,
                    gn_groups=int(cfg.gn_groups),
                    act=cfg.act,
                    dropout=float(cfg.dropout),
                    padding_mode=cfg.conv_padding_mode,
                )
                for _ in range(int(cfg.num_res_blocks))
            ]
        )
        self.norm_out = _make_norm(cfg.norm, int(cfg.hidden_channels), int(cfg.gn_groups))
        self.act_out = _make_act(cfg.act)
        self.conv_out = nn.Conv3d(
            int(cfg.hidden_channels),
            int(cfg.out_channels),
            int(cfg.kernel_size),
            padding=pad,
            padding_mode=cfg.conv_padding_mode,
        )

    def get_last_layer_weight(self) -> torch.Tensor:
        return self.conv_out.weight

    @staticmethod
    def _feature_coord_grid(
        spatial: Spatial3D,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch: int,
    ) -> torch.Tensor:
        d, h, w = spatial
        z = (torch.arange(d, device=device, dtype=torch.float32) + 0.5) * (2.0 / d) - 1.0
        y = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * (2.0 / h) - 1.0
        x = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * (2.0 / w) - 1.0
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        grid = torch.stack((xx, yy, zz), dim=0).unsqueeze(0)
        return grid.expand(batch, -1, -1, -1, -1)

    def make_layout(
        self,
        feat: torch.Tensor,
        coord: torch.Tensor,
        cell: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build CLIF input layout.

        Args:
            feat:  [B,C,Df,Hf,Wf]
            coord: [B,Do,Ho,Wo,3], coordinate order (x,y,z), range [-1,1]
            cell:  same spatial shape as coord, normalized voxel size (x,y,z)
        """
        if feat.ndim != 5 or coord.ndim != 5 or coord.shape[-1] != 3:
            raise ValueError("Expected feat [B,C,D,H,W] and coord [B,D,H,W,3].")
        if feat.shape[0] != coord.shape[0]:
            raise ValueError("Feature and coordinate batch sizes differ.")
        if self.cfg.use_cell and cell is None:
            raise ValueError("CLIF is configured with use_cell=True but cell is None.")

        b, _, fd, fh, fw = feat.shape
        feat_coord = self._feature_coord_grid(
            (fd, fh, fw), device=feat.device, dtype=torch.float32, batch=b
        )

        q_feat = sample_fp32(
            feat,
            coord,
            mode="nearest",
            padding_mode=self.cfg.padding_mode,
            align_corners=self.cfg.align_corners,
        )
        q_coord = sample_fp32(
            feat_coord,
            coord,
            mode="nearest",
            padding_mode=self.cfg.padding_mode,
            align_corners=self.cfg.align_corners,
        )

        coord_cf = coord.permute(0, 4, 1, 2, 3).contiguous()
        rel_coord = coord_cf - q_coord
        scale = torch.tensor(
            [fw, fh, fd], device=feat.device, dtype=torch.float32
        ).view(1, 3, 1, 1, 1)
        rel_coord = rel_coord * scale

        layout = [q_feat, rel_coord]
        if self.cfg.use_cell:
            assert cell is not None
            rel_cell = cell.permute(0, 4, 1, 2, 3).contiguous() * scale
            layout.append(rel_cell)
        return torch.cat(layout, dim=1)

    def forward(
        self,
        feat: torch.Tensor,
        coord: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        checkpoint_blocks: bool = False,
    ) -> torch.Tensor:
        layout = run_checkpoint(self.make_layout,feat,coord,cell,enabled=checkpoint_blocks)
        h = block(self.conv_in,layout,checkpoint_blocks)
        for layer in self.blocks:h = block(layer,h,checkpoint_blocks)
        # Keep the last Conv3d eager. Adaptive GAN autograd.grad then needs only
        # its small patch input, without replaying the encoder or whole head.
        h=block(self.norm_out,h,checkpoint_blocks)
        return self.conv_out(self.act_out(h))

    @staticmethod
    def make_region_coord_cell(
        full_spatial: Spatial3D,
        start: Spatial3D,
        end: Spatial3D,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Coordinates for a rectangular region of a larger output grid."""
        d, h, w = full_spatial
        sd, sh, sw = start
        ed, eh, ew = end
        zi = torch.arange(sd, ed, device=device, dtype=torch.float32)
        yi = torch.arange(sh, eh, device=device, dtype=torch.float32)
        xi = torch.arange(sw, ew, device=device, dtype=torch.float32)
        z = (zi + 0.5) * (2.0 / d) - 1.0
        y = (yi + 0.5) * (2.0 / h) - 1.0
        x = (xi + 0.5) * (2.0 / w) - 1.0
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        coord = torch.stack((xx, yy, zz), dim=-1).unsqueeze(0)
        coord = coord.expand(batch, -1, -1, -1, -1).contiguous()
        cell = torch.empty_like(coord)
        cell[..., 0] = 2.0 / w
        cell[..., 1] = 2.0 / h
        cell[..., 2] = 2.0 / d
        return coord, cell

    def iter_render_tiles(
        self,
        feat: torch.Tensor,
        spatial: Spatial3D,
        *,
        tile_size: int | Sequence[int] | None = None,
        halo: Optional[int] = None,
    ) -> Generator[tuple[tuple[slice, slice, slice], torch.Tensor], None, None]:
        """Yield cropped output tiles while retaining differentiability."""
        if feat.ndim != 5:
            raise ValueError("Expected feature tensor [B,C,D,H,W].")
        spatial = _triple(spatial)
        tile = _triple(self.cfg.tile_size if tile_size is None else tile_size)
        halo_value = int(self.cfg.halo if halo is None else halo)
        if halo_value < 0:
            raise ValueError("halo must be non-negative.")

        if self.cfg.norm not in ("channel", "none"):
            raise ValueError("Tiled CLIF requires norm=channel or none. Spatial GN/IN/BN checkpoints need full-patch rendering; retrain a tile-local head.")
        required = (2 + 2 * int(self.cfg.num_res_blocks)) * (int(self.cfg.kernel_size)//2)
        if halo_value < required:
            raise ValueError(f"CLIF halo {halo_value} is smaller than receptive-field radius {required}.")
        if min(tile) < 1:
            raise ValueError("Tile dimensions must be positive")
        d, h, w = spatial
        td, th, tw = tile
        batch = feat.shape[0]

        for sd in range(0, d, td):
            ed = min(sd + td, d)
            for sh in range(0, h, th):
                eh = min(sh + th, h)
                for sw in range(0, w, tw):
                    ew = min(sw + tw, w)

                    hsd, hed = max(0, sd - halo_value), min(d, ed + halo_value)
                    hsh, heh = max(0, sh - halo_value), min(h, eh + halo_value)
                    hsw, hew = max(0, sw - halo_value), min(w, ew + halo_value)

                    coord, cell = self.make_region_coord_cell(
                        spatial,
                        (hsd, hsh, hsw),
                        (hed, heh, hew),
                        device=feat.device,
                        dtype=torch.float32,
                        batch=batch,
                    )
                    pred_halo = self.forward(feat, coord, cell if self.cfg.use_cell else None)

                    cd0, ch0, cw0 = sd - hsd, sh - hsh, sw - hsw
                    cd1, ch1, cw1 = cd0 + (ed - sd), ch0 + (eh - sh), cw0 + (ew - sw)
                    pred = pred_halo[:, :, cd0:cd1, ch0:ch1, cw0:cw1]
                    yield (slice(sd, ed), slice(sh, eh), slice(sw, ew)), pred

    def render(
        self,
        feat: torch.Tensor,
        spatial: Spatial3D,
        *,
        tile_size: int | Sequence[int] | None = None,
        halo: Optional[int] = None,
    ) -> torch.Tensor:
        """Render a dense volume tile by tile.

        The returned scalar output volume is materialized, but full-resolution
        hidden feature maps are never materialized.
        """
        d, h, w = _triple(spatial)
        out = feat.new_empty((feat.shape[0], int(self.cfg.out_channels), d, h, w))
        for (ds, hs, ws), tile in self.iter_render_tiles(
            feat, (d, h, w), tile_size=tile_size, halo=halo
        ):
            out[:, :, ds, hs, ws] = tile
        return out


__all__ = ["CLIFHeadConfig", "CLIFHead3D", "CLIFResBlock3D"]
