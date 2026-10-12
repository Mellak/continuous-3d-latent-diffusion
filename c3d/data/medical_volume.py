#!/usr/bin/env python3
"""Full-volume 3-D datasets used by the AE trainers.

MRI is normalized by foreground percentiles and returned at 256^3.
CT is clipped in HU and returned at 512^3. AutoPET discovery prefers CTres.nii.gz
and excludes PET/SUV/segmentation files.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler, Dataset


Spatial3D = Tuple[int, int, int]
SpacingXYZ = Tuple[float, float, float]


def parse_int3(value: str | Sequence[int] | int) -> Spatial3D:
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, str):
        values = [int(v.strip()) for v in value.replace("x", ",").split(",") if v.strip()]
    else:
        values = [int(v) for v in value]
    if len(values) == 1:
        return (values[0], values[0], values[0])
    if len(values) != 3:
        raise ValueError(f"Expected D,H,W, got {values}.")
    return tuple(values)  # type: ignore[return-value]


def parse_float3(value: str | Sequence[float] | None) -> Optional[SpacingXYZ]:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in ("", "none", "native"):
            return None
        values = [float(v.strip()) for v in value.replace("x", ",").split(",") if v.strip()]
    else:
        values = [float(v) for v in value]
    if len(values) != 3:
        raise ValueError("Spacing must be x,y,z in millimetres.")
    return tuple(values)  # type: ignore[return-value]


def parse_patterns(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


def _is_nifti(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def _is_supported(path: Path) -> bool:
    lower = path.name.lower()
    return _is_nifti(path) or lower.endswith((".npy", ".npz", ".pt", ".pth"))


def discover_files(
    root: str | Path,
    *,
    modality: str,
    patterns: Sequence[str] | None = None,
    file_list: str | Path | None = None,
) -> list[Path]:
    root = Path(root).expanduser()
    modality = modality.lower()
    selected: list[Path] = []

    if file_list:
        for line in Path(file_list).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                selected.append(Path(line.split("\t")[-1]).expanduser())
    elif root.is_file():
        selected = [root]
    else:
        if not root.is_dir():
            raise FileNotFoundError(f"Data root does not exist: {root}")
        patterns = list(patterns or [])
        if patterns:
            for pattern in patterns:
                selected.extend(root.rglob(pattern))
        else:
            selected = [p for p in root.rglob("*") if p.is_file() and _is_supported(p)]

    output: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(selected):
        path = path.resolve()
        if path in seen or not path.is_file() or not _is_supported(path):
            continue
        name = path.name.lower()
        if any(token in name for token in ("seg", "mask", "label")):
            continue
        if modality == "ct" and any(token in name for token in ("pet", "suv")):
            continue
        if modality == "ct" and not patterns:
            # AutoPET's CT file is normally CTres.nii.gz. Array files remain accepted.
            if not ("ct" in name or name.endswith((".npy", ".npz", ".pt", ".pth"))):
                continue
        seen.add(path)
        output.append(path)

    if not output:
        raise RuntimeError(
            f"No {modality.upper()} volumes found in {root}. "
            "Set --file-pattern when the filenames are non-standard."
        )
    return output


def load_volume(path: str | Path) -> tuple[np.ndarray, SpacingXYZ]:
    """Return float32 volume in D,H,W order and physical spacing in x,y,z."""
    path = Path(path)
    lower = path.name.lower()

    if _is_nifti(path):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("NIfTI input requires nibabel inside the SIF.") from exc
        image = nib.as_closest_canonical(nib.load(str(path)))
        xyz = np.asarray(image.get_fdata(dtype=np.float32))
        if xyz.ndim != 3:
            raise ValueError(f"Expected a 3-D NIfTI volume, got {xyz.shape}: {path}")
        spacing_xyz = tuple(float(v) for v in image.header.get_zooms()[:3])
        return np.ascontiguousarray(np.transpose(xyz, (2, 1, 0))), spacing_xyz

    if lower.endswith(".npy"):
        value = np.load(path, allow_pickle=False)
    elif lower.endswith(".npz"):
        archive = np.load(path, allow_pickle=False)
        if not archive.files:
            raise ValueError(f"Empty NPZ: {path}")
        value = archive[archive.files[0]]
    else:
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            for key in ("volume", "image", "ct", "mri", "data"):
                if key in value:
                    value = value[key]
                    break
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()

    array = np.asarray(value)
    array = np.squeeze(array)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D array after squeezing, got {array.shape}: {path}")
    return np.ascontiguousarray(array.astype(np.float32)), (1.0, 1.0, 1.0)


def resample_to_spacing(
    volume_dhw: np.ndarray,
    source_spacing_xyz: SpacingXYZ,
    target_spacing_xyz: Optional[SpacingXYZ],
) -> np.ndarray:
    if target_spacing_xyz is None:
        return volume_dhw
    sx, sy, sz = source_spacing_xyz
    tx, ty, tz = target_spacing_xyz
    factors_dhw = (sz / tz, sy / ty, sx / tx)
    new_shape = tuple(
        max(1, int(round(size * factor)))
        for size, factor in zip(volume_dhw.shape, factors_dhw)
    )
    if new_shape == tuple(volume_dhw.shape):
        return volume_dhw
    x = torch.from_numpy(np.ascontiguousarray(volume_dhw))[None, None].float()
    x = F.interpolate(x, size=new_shape, mode="trilinear", align_corners=False)
    return x[0, 0].numpy()


def center_crop_or_pad(volume: np.ndarray, target: Spatial3D, pad_value: float) -> np.ndarray:
    slices = []
    for current, wanted in zip(volume.shape, target):
        start = max(0, (current - wanted) // 2)
        slices.append(slice(start, min(current, start + wanted)))
    result = volume[tuple(slices)]

    pads = []
    for current, wanted in zip(result.shape, target):
        missing = max(0, wanted - current)
        before = missing // 2
        pads.append((before, missing - before))
    if any(before or after for before, after in pads):
        result = np.pad(result, pads, mode="constant", constant_values=float(pad_value))
    if tuple(result.shape) != tuple(target):
        raise RuntimeError(f"Crop/pad failed: got {result.shape}, expected {target}.")
    return np.ascontiguousarray(result)


def preprocess_ct(
    volume: np.ndarray,
    spacing_xyz: SpacingXYZ,
    *,
    target_shape: Spatial3D,
    target_spacing_xyz: Optional[SpacingXYZ],
    min_hu: float,
    max_hu: float,
    pad_hu: float,
) -> np.ndarray:
    volume = np.nan_to_num(volume, nan=pad_hu, posinf=max_hu, neginf=min_hu)
    volume = resample_to_spacing(volume, spacing_xyz, target_spacing_xyz)
    volume = center_crop_or_pad(volume, target_shape, pad_hu)
    volume = np.clip(volume, min_hu, max_hu)
    volume = 2.0 * (volume - min_hu) / max(max_hu - min_hu, 1e-6) - 1.0
    return volume.astype(np.float32, copy=False)


def preprocess_mri(
    volume: np.ndarray,
    spacing_xyz: SpacingXYZ,
    *,
    target_shape: Spatial3D,
    target_spacing_xyz: Optional[SpacingXYZ],
    percentile_low: float,
    percentile_high: float,
    foreground_threshold: float,
) -> np.ndarray:
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    volume = resample_to_spacing(volume, spacing_xyz, target_spacing_xyz)
    foreground = volume[np.abs(volume) > float(foreground_threshold)]
    values = foreground if foreground.size >= 32 else volume.reshape(-1)
    low, high = np.percentile(values, [percentile_low, percentile_high]).astype(np.float64)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(volume.min()), float(volume.max())
    if high <= low:
        high = low + 1.0
    volume = np.clip(volume, low, high)
    volume = 2.0 * (volume - low) / (high - low) - 1.0
    volume = center_crop_or_pad(volume, target_shape, -1.0)
    return volume.astype(np.float32, copy=False)


class MedicalVolumeDataset3D(Dataset):
    def __init__(
        self,
        *,
        root: str | Path,
        modality: str,
        target_shape: Spatial3D,
        target_spacing_xyz: Optional[SpacingXYZ],
        patterns: Sequence[str] | None = None,
        file_list: str | Path | None = None,
        augment_flip: bool = True,
        ct_min_hu: float = -1000.0,
        ct_max_hu: float = 1500.0,
        ct_pad_hu: float = -1000.0,
        mri_percentile_low: float = 1.0,
        mri_percentile_high: float = 99.0,
        mri_foreground_threshold: float = 0.0,
    ) -> None:
        modality = modality.lower()
        if modality not in ("ct", "mri"):
            raise ValueError("modality must be ct or mri.")
        self.modality = modality
        self.target_shape = tuple(int(v) for v in target_shape)
        self.target_spacing_xyz = target_spacing_xyz
        self.augment_flip = bool(augment_flip)
        self.ct_min_hu = float(ct_min_hu)
        self.ct_max_hu = float(ct_max_hu)
        self.ct_pad_hu = float(ct_pad_hu)
        self.mri_percentile_low = float(mri_percentile_low)
        self.mri_percentile_high = float(mri_percentile_high)
        self.mri_foreground_threshold = float(mri_foreground_threshold)
        self.files = discover_files(
            root,
            modality=modality,
            patterns=patterns,
            file_list=file_list,
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]
        volume, spacing_xyz = load_volume(path)
        if self.modality == "ct":
            volume = preprocess_ct(
                volume,
                spacing_xyz,
                target_shape=self.target_shape,
                target_spacing_xyz=self.target_spacing_xyz,
                min_hu=self.ct_min_hu,
                max_hu=self.ct_max_hu,
                pad_hu=self.ct_pad_hu,
            )
        else:
            volume = preprocess_mri(
                volume,
                spacing_xyz,
                target_shape=self.target_shape,
                target_spacing_xyz=self.target_spacing_xyz,
                percentile_low=self.mri_percentile_low,
                percentile_high=self.mri_percentile_high,
                foreground_threshold=self.mri_foreground_threshold,
            )

        tensor = torch.from_numpy(volume).unsqueeze(0)
        rng=random.Random(int(getattr(self,"augmentation_seed",0))+int(getattr(self,"epoch",0))*1000003+index)
        if self.augment_flip:
            for axis in (1, 2, 3):
                if rng.random() < 0.5:
                    tensor = torch.flip(tensor, dims=(axis,))
        return {"volume": tensor.contiguous(), "path": str(path)}


def make_dataloader(
    *,
    root: str | Path,
    modality: str,
    target_shape: Spatial3D,
    target_spacing_xyz: Optional[SpacingXYZ],
    patterns: Sequence[str] | None,
    file_list: str | Path | None,
    batch_size: int,
    num_workers: int,
    augment_flip: bool,
    ct_min_hu: float = -1000.0,
    ct_max_hu: float = 1500.0,
    ct_pad_hu: float = -1000.0,
    mri_percentile_low: float = 1.0,
    mri_percentile_high: float = 99.0,
    mri_foreground_threshold: float = 0.0,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    dataset = MedicalVolumeDataset3D(
        root=root,
        modality=modality,
        target_shape=target_shape,
        target_spacing_xyz=target_spacing_xyz,
        patterns=patterns,
        file_list=file_list,
        augment_flip=augment_flip,
        ct_min_hu=ct_min_hu,
        ct_max_hu=ct_max_hu,
        ct_pad_hu=ct_pad_hu,
        mri_percentile_low=mri_percentile_low,
        mri_percentile_high=mri_percentile_high,
        mri_foreground_threshold=mri_foreground_threshold,
    )
    workers = max(0, int(num_workers))
    sampler = DistributedSampler(dataset, num_replicas=int(world_size), rank=int(rank), shuffle=True, drop_last=False) if distributed else None
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


__all__ = [
    "parse_int3",
    "parse_float3",
    "parse_patterns",
    "MedicalVolumeDataset3D",
    "make_dataloader",
]
