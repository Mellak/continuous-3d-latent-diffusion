"""Shared helpers for the inference scripts and notebooks."""
from __future__ import annotations
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch

from c3d.data.medical_volume import center_crop_or_pad, load_volume, preprocess_ct, preprocess_mri


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def autocast(device: torch.device, precision: str = 'bf16'):
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    enabled = device.type == 'cuda' and precision in ('bf16', 'fp16')
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def load_input(path, ae_cfg: dict, resample: bool = True, preprocessed: bool = False) -> torch.Tensor:
    """Read a NIfTI/NumPy/PyTorch volume and apply the model's preprocessing.

    ``preprocessed=True`` means the volume is already in [-1, 1] on the
    training grid; it is then only centre-cropped/padded to the native shape.
    Returns a [1, 1, D, H, W] tensor in [-1, 1].
    """
    volume, spacing = load_volume(path)               # NIfTI -> D,H,W = z,y,x
    if str(path).lower().endswith(('.nii', '.nii.gz')) and ae_cfg.get('array_axes') == 'xyz':
        volume = np.ascontiguousarray(volume.transpose(2, 1, 0))   # model trained on x,y,z arrays
        spacing = tuple(spacing)
    pre = ae_cfg['preprocessing']
    shape = tuple(ae_cfg['native_shape'])
    if preprocessed:
        return torch.from_numpy(center_crop_or_pad(volume.astype('float32'), shape, -1.0))[None, None]
    target = ae_cfg.get('voxel_spacing_xyz_mm') if resample else None
    target = tuple(target) if target else None
    if ae_cfg['modality'] == 'ct':
        x = preprocess_ct(volume, spacing, target_shape=shape, target_spacing_xyz=target,
                          min_hu=pre['ct_min_hu'], max_hu=pre['ct_max_hu'], pad_hu=pre['ct_pad_hu'])
    else:
        x = preprocess_mri(volume, spacing, target_shape=shape, target_spacing_xyz=target,
                           percentile_low=pre['mri_percentile_low'], percentile_high=pre['mri_percentile_high'],
                           foreground_threshold=pre['mri_foreground_threshold'])
    return torch.from_numpy(x)[None, None]


def to_output_units(x: torch.Tensor, ae_cfg: dict) -> np.ndarray:
    """[-1, 1] → HU for CT; MRI stays in the normalised [-1, 1] range."""
    v = x[0, 0].float().cpu().numpy()
    if ae_cfg['modality'] == 'ct':
        lo, hi = ae_cfg['preprocessing']['ct_min_hu'], ae_cfg['preprocessing']['ct_max_hu']
        v = (np.clip(v, -1, 1) + 1) * 0.5 * (hi - lo) + lo
    return v.astype(np.float32)


def save_nifti(volume_dhw: np.ndarray, path, spacing_xyz=None, array_axes: str = 'zyx'):
    """Write a model-grid array as NIfTI with a diagonal affine (voxel spacing).

    ``array_axes`` is the axis order of the model's arrays: 'zyx' (CT) or 'xyz' (MRI).
    """
    import nibabel as nib
    if array_axes == 'xyz':
        volume_dhw = np.transpose(volume_dhw, (2, 1, 0))
    spacing = spacing_xyz or (1.0, 1.0, 1.0)
    affine = np.diag([float(spacing[0]), float(spacing[1]), float(spacing[2]), 1.0])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.transpose(volume_dhw, (2, 1, 0)), affine), str(path))


# Anatomical name of the central slice along array axes 0, 1, 2. CT arrays are stored
# superior-inferior first; the prepared IXI MRI arrays are stored left-right first.
PLANES = {'ct': ('axial', 'coronal', 'sagittal'), 'mri': ('sagittal', 'coronal', 'axial')}


def three_planes(volumes: dict, path=None, title: str = '', modality: str = 'ct'):
    """Central slices along the three array axes; one column per volume ([1,1,D,H,W] tensors in [-1, 1])."""
    import matplotlib
    if path is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(volumes)
    fig, axes = plt.subplots(3, len(names), figsize=(3.2 * len(names), 9.6), squeeze=False)
    for j, name in enumerate(names):
        v = volumes[name][0, 0].float().cpu()
        d, h, w = v.shape
        planes = (v[d // 2], v[:, h // 2, :].flip(0), v[:, :, w // 2].flip(0))
        for i, (img, plane) in enumerate(zip(planes, PLANES.get(modality, ('axis 0', 'axis 1', 'axis 2')))):
            axes[i, j].imshow(((img.clamp(-1, 1) + 1) * 0.5).numpy(), cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            axes[i, j].set_title(f'{name} · {plane}', fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig


def psnr(a: torch.Tensor, b: torch.Tensor, data_range: float = 2.0) -> float:
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return float('inf') if mse == 0 else 10 * np.log10(data_range ** 2 / mse)


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    from c3d.losses.ssim_3d import ssim3d
    return float(ssim3d(a.float(), b.float()).mean())


@contextmanager
def measure(device: torch.device, result: dict):
    """Wall time and peak GPU memory of the enclosed block."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    yield result
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        result['peak_gib'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
    result['seconds'] = time.perf_counter() - start
