import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def load_session_tensor(row: pd.Series, data_root: Path, seqs: list, hw: tuple) -> torch.Tensor:
    """
    Load one session row → tensor of shape (1, C, H, W) in [-1, 1].
    """
    frames = []
    for seq in seqs:
        img_path = data_root / row[seq]
        img = plt.imread(img_path)
        if img.ndim == 3:
            img = img[..., 0]  # Convert RGBA to grayscale if needed
        img_resized = np.array(Image.fromarray(img).resize((hw[1], hw[0]), resample=Image.BILINEAR))
        frames.append(img_resized)
    tensor = torch.from_numpy(np.stack(frames)).unsqueeze(0).float()  # (1, C, H, W)
    tensor = (tensor / 127.5) - 1.0  # Rescale to [-1, 1]
    return tensor

def sample_and_save(model, gt_frames, lt_seq, img_size, sequences, out_dir, T, compare=True, pre_images=None, pre_times=None):
    """
    Generate and save samples. If compare is True, display comparison with ground truth in a grid.
    If compare is False, do not expect GT images and save each generated image as a separate file.
    """
    B = 1
    C = len(sequences)
    if isinstance(img_size, (tuple, list)):
        H, W = img_size
    else:
        H = W = img_size
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
    with torch.no_grad():
        gen = model.sample(shape=(B, C, H, W), T=T, lt_seq=lt_seq, pre_images=pre_images, pre_times=pre_times, device=device)

    if compare:
        if gt_frames is None or (isinstance(gt_frames, (list, tuple)) and len(gt_frames) == 0):
            print("No ground truth frames provided for comparison.")
            return
        n_visits = len(gt_frames)
        n_rows = C * 2   # GT row + Gen row per channel
        fig, axes = plt.subplots(n_rows, n_visits, figsize=(n_visits * 1.8, n_rows * 1.8))
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
        jpg_path = out_dir / 'samples.jpg'
        plt.savefig(grid_path, dpi=150)
        plt.savefig(jpg_path, dpi=150, format='jpg')
        plt.close()
        print(f"Saved sample grid → {grid_path}")
        print(f"Saved sample grid as JPEG → {jpg_path}")
    else:
        n_visits = len(gen)
        for v in range(n_visits):
            for c_idx, seq_name in enumerate(sequences):
                gen_img = gen[v][0, c_idx].cpu().numpy()
                # Rescale from [-1, 1] to [0, 255] for saving
                img_uint8 = ((gen_img + 1) * 127.5).clip(0, 255).astype(np.uint8)
                from PIL import Image
                im = Image.fromarray(img_uint8)
                fname = out_dir / f'gen_{seq_name}_v{v}.png'
                im.save(fname)
                print(f"Saved generated image → {fname}")