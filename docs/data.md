# Data

No images are distributed with this repository.

## CT (AutoPET)

- Source: AutoPET / FDG-PET-CT-Lesions on The Cancer Imaging Archive (TCIA). Use the CT series only.
- Preparation used for the released models: resample to **1 × 1 × 1.5 mm** (x, y, z), then centre-crop or pad to **512³** (pad value −1000 HU).
- At load time the code clips to [−1000, 1500] HU and scales linearly to [−1, 1] (`configs/ae/ct_*.yaml`, `data:`).
- Volumes can also be resampled on the fly: set `data.target_spacing: [1.0, 1.0, 1.5]`.

## MRI (IXI)

- Source: IXI dataset, T1-weighted volumes (https://brain-development.org/ixi-dataset/).
- Volumes are centre-cropped or padded to **256³**. Intensities are clipped to the 1st–99th percentiles of the foreground and scaled to [−1, 1].

## Folder layout

`--data <folder>` is searched recursively for `.nii`, `.nii.gz`, `.npy`, `.npz`, `.pt` and `.pth` volumes. Files whose names contain `seg`, `mask` or `label` are skipped. For CT, NIfTI files must have `ct` in the name (AutoPET: `CTres.nii.gz`), and PET/SUV files are skipped. Use `--set "data.patterns=['*my_ct*.nii.gz']"` for other names. NIfTI files are reoriented to the closest canonical orientation, and their voxel spacing is read from the header. NumPy and PyTorch arrays are read as D × H × W.

A validation split (10 % of the files, fixed seed) is made automatically. If a patient has several scans, give an explicit patient-level split with `data.file_list`.
