import torch
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from matplotlib import pyplot as plt

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    beta = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(beta, 1e-4, 0.999)

def load_nifti_slice(path: Path, slice_axis: int, slice_idx: int,
                     target_hw: tuple) -> np.ndarray:
    """
    Load one 2-D slice from a 3-D NIfTI volume.
    Returns float32 array of shape (H, W), z-score normalised to [-1, 1].
    """
    img = nib.load(str(path))
    vol = img.get_fdata(dtype=np.float32)

    # Extract slice
    if slice_axis == 0:
        sl = vol[slice_idx, :, :]
    elif slice_axis == 1:
        sl = vol[:, slice_idx, :]
    else:
        sl = vol[:, :, slice_idx]

    # Resize to target spatial size if needed
    H, W = target_hw
    if sl.shape != (H, W):
        from PIL import Image as PILImage
        pil = PILImage.fromarray(sl).resize((W, H), PILImage.BILINEAR)
        sl = np.array(pil, dtype=np.float32)

    # Z-score then clip to [-1, 1]  (same range as your synthetic test)
    std = sl.std()
    if std > 1e-6:
        sl = (sl - sl.mean()) / std
    sl = np.clip(sl, -3, 3) / 3.0   # now in [-1, 1]
    return sl

def load_session_tensor(row: pd.Series, data_root: Path,
                        sequences: list, slice_axis: int,
                        slice_idx: int, target_hw: tuple) -> torch.Tensor:
    """
    Load all requested sequences for one session.
    Returns tensor of shape (1, C, H, W)  where C = len(sequences).
    Missing sequences are filled with zeros.
    """
    H, W = target_hw
    channels = []
    for seq in sequences:
        fname = row.get(seq)
        if fname and pd.notna(fname):
            fpath = data_root / fname
            if fpath.exists():
                sl = load_nifti_slice(fpath, slice_axis, slice_idx, target_hw)
                channels.append(sl)
                continue
        channels.append(np.zeros((H, W), dtype=np.float32))

    arr = np.stack(channels, axis=0)
    return torch.from_numpy(arr).unsqueeze(0)

def sample_and_save(model, gt_frames, lt_seq, img_size, sequences, out_dir, T):
    B = 1
    C = len(sequences)
    H = W = img_size
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        gen = model.sample(shape=(B, C, H, W), T=T, lt_seq=lt_seq)

    n_visits = len(gt_frames)
    n_rows = C * 2   # GT row + Gen row per channel

    fig, axes = plt.subplots(n_rows, n_visits,
                             figsize=(n_visits * 1.8, n_rows * 1.8))
    if n_visits == 1:
        axes = axes[:, np.newaxis]

    for c_idx, seq_name in enumerate(sequences):
        row_gt  = c_idx * 2
        row_gen = c_idx * 2 + 1

        for v in range(n_visits):
            # Ground truth
            gt_img = gt_frames[v][0, c_idx].cpu().numpy()
            axes[row_gt, v].imshow(gt_img, cmap='gray', vmin=-1, vmax=1)
            axes[row_gt, v].axis('off')
            if v == 0:
                axes[row_gt, v].set_ylabel(f'GT {seq_name}', fontsize=7)
            if row_gt == 0:
                axes[row_gt, v].set_title(f'v{v}', fontsize=7)

            # Generated
            gen_img = gen[v][0, c_idx].cpu().numpy()
            axes[row_gen, v].imshow(gen_img, cmap='gray', vmin=-1, vmax=1)
            axes[row_gen, v].axis('off')
            if v == 0:
                axes[row_gen, v].set_ylabel(f'Gen {seq_name}', fontsize=7)

    plt.suptitle('RDDIM — Ground Truth vs Generated (MRI)', fontsize=10)
    plt.tight_layout()
    grid_path = out_dir / 'sample_grid.png'
    plt.savefig(grid_path, dpi=150)
    plt.close()
    print(f"Saved sample grid → {grid_path}")