# %%
# train_synthetic.py
#
# Drop-in replacement for train.py that loads the synthetic PNG dataset
# exported by the HTML generator instead of NIfTI + Excel files.
#
# Expected dataset layout (produced by "Export dataset" in the HTML tool):
#
#   training_data/
#   ├── session_table.csv          ← columns: session_date, FLAIR, POST, PRE, T2
#   └── images/
#       └── P001/
#           ├── 2023-01-15/
#           │   ├── FLAIR_T01.png
#           │   ├── POST_T01.png
#           │   ├── PRE_T01.png
#           │   └── T2_T01.png
#           └── ...
#
# Usage:
#   python train_synthetic.py
#   python train_synthetic.py --data training_data/ --epochs 200 --device cuda

import argparse
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from RDDPM import RDDPM
from utils import generate

torch.backends.cudnn.benchmark = True

sequences       = ['FLAIR', 'POST', 'PRE', 'T2']
BASE_DIM         = 64
GRU_LAYERS       = 6
RES_BLOCKS       = 3
T                = 1000
BETA_SCHEDULE    = 'linear'
EPOCHS           = 10000
LEARNING_RATE    = 1e-4
H = W            = 96
min_visits       = 5
subsample_visits = True
log_every        = 50
save_every       = 200
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path       = 'training_data/'

checkpoint_dir  = Path("./checkpoints")
output_dir      = Path("./output")
output_dir.mkdir(exist_ok=True)
checkpoint_dir.mkdir(exist_ok=True)

csv_path = Path(data_path) / "session_table.csv"
assert csv_path.exists(), f"session_table.csv not found at {csv_path}"

session_table = pd.read_csv(csv_path)
session_table = session_table.sort_values("session_date").reset_index(drop=True)

print(f"Found {len(session_table)} sessions in session_table.csv")
print(session_table[["session_date"] + [s for s in sequences if s in session_table.columns]].to_string())


def load_session_frame(row: pd.Series, data_root: Path, seqs: list, hw: tuple) -> torch.Tensor:
    """
    Load one session row → tensor of shape (1, C, H, W) in [-1, 1].

    For each sequence, reads the PNG at data_root / row[seq].
    If a sequence column is missing or the file doesn't exist the channel
    is filled with zeros (matches the NaN-handling in LongitudinalMRIDataset).
    """
    channels = []
    for seq in seqs:
        img_path = data_root / row[seq] if seq in row and pd.notna(row[seq]) else None
        if img_path is not None and img_path.exists():
            img = Image.open(img_path).convert("L")          # grayscale
            img = img.resize((hw[1], hw[0]), Image.BILINEAR) # W, H order for PIL
            arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [0,255] → [-1,1]
        else:
            arr = np.zeros(hw, dtype=np.float32)
        channels.append(torch.from_numpy(arr))

    frame = torch.stack(channels, dim=0)   # (C, H, W)
    return frame.unsqueeze(0)              # (1, C, H, W)


def _load_png_01(path: Path, hw: tuple) -> np.ndarray:
    """Load PNG as float image in [0, 1] with fixed output size."""
    img = Image.open(path).convert("L")
    img = img.resize((hw[1], hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    mse = float(np.mean((gt - pred) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def save_comparison_table(
    generated_results: dict,
    patient_sessions: pd.DataFrame,
    data_root: Path,
    seqs: list,
    hw: tuple,
    n_input: int,
    out_csv: Path,
    out_summary_csv: Path,
):
    """
    Save per-frame GT-vs-generated comparison rows and per-sequence summary.
    """
    rows = []

    for seq in seqs:
        gen_paths = generated_results.get(seq, [])
        for i, gen_path_str in enumerate(gen_paths):
            visit_idx = n_input + i
            session_date = ""
            gt_path = None

            if visit_idx < len(patient_sessions):
                row = patient_sessions.iloc[visit_idx]
                session_date = str(row.get("session_date", ""))
                gt_rel = row.get(seq)
                if isinstance(gt_rel, str) and gt_rel:
                    gt_path = data_root / gt_rel

            gen_path = Path(gen_path_str)
            record = {
                "visit_idx": visit_idx,
                "session_date": session_date,
                "sequence": seq,
                "gt_path": str(gt_path) if gt_path is not None else "",
                "gen_path": str(gen_path),
                "mae": np.nan,
                "mse": np.nan,
                "psnr": np.nan,
            }

            if gt_path is not None and gt_path.exists() and gen_path.exists():
                gt = _load_png_01(gt_path, hw)
                pred = _load_png_01(gen_path, hw)
                mae = float(np.mean(np.abs(gt - pred)))
                mse = float(np.mean((gt - pred) ** 2))
                record.update({
                    "mae": mae,
                    "mse": mse,
                    "psnr": _psnr(gt, pred),
                })

            rows.append(record)

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(out_csv, index=False)

    if not cmp_df.empty:
        summary_df = (
            cmp_df.groupby("sequence", as_index=False)[["mae", "mse", "psnr"]]
            .mean(numeric_only=True)
            .sort_values("sequence")
        )
    else:
        summary_df = pd.DataFrame(columns=["sequence", "mae", "mse", "psnr"])
    summary_df.to_csv(out_summary_csv, index=False)

    return cmp_df, summary_df


data_root = Path(data_path)
frames = [
    load_session_frame(row, data_root, sequences, (H, W))
    for _, row in session_table.iterrows()
]
n_visits = len(frames)

# lt_seq: integer indices 0 … n_visits-1, shape (n_visits,)
# This is the longitudinal time index tensor used by RDDPM.train_step
lt_seq = torch.arange(n_visits, dtype=torch.long)

print(f"\nDataset loaded: {n_visits} visits")
print(f"Frame shape:    {frames[0].shape}  (1, C, H, W)")
print(f"Sequences:      {sequences}")

# ---------------------------------------------------------------------------
# ── Model ────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
model = RDDPM(
    input_size=(H, W),
    n_channels=len(sequences),
    base_dim=BASE_DIM,
    gru_n_layers=GRU_LAYERS,
    n_res_blocks=RES_BLOCKS,
    T=T,
    beta_schedule=BETA_SCHEDULE,
)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {total_params:,}")

start_epoch = 0
best_loss   = float('inf')

latest_ckpt = checkpoint_dir / "latest.pth"
if latest_ckpt.exists():
    print(f"Resuming from {latest_ckpt}")
    ckpt = torch.load(latest_ckpt, map_location="cpu")
    try:
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt['epoch'] + 1
        best_loss   = ckpt.get('best_loss', float('inf'))
        print(f"  → Resuming at epoch {start_epoch}, best loss {best_loss:.4f}")
    except RuntimeError as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting fresh training instead.")
        start_epoch = 0
        best_loss   = float('inf')
else:
    print("No checkpoint found. Starting fresh training.")

# ---------------------------------------------------------------------------
# ── Training loop ────────────────────────────────────────────────────────
# Identical structure to train.py — only the data source changed.
# ---------------------------------------------------------------------------
loss_history = []
total_start  = time.perf_counter()

print(f"\nTraining for {EPOCHS} epochs...")
print(f"  Sequences  : {sequences}")
print(f"  Image size : {H}x{W}")
print(f"  Device     : {DEVICE}")
print()

model  = model.to(DEVICE)
lt_seq = lt_seq.to(DEVICE)

try:
    for epoch in range(start_epoch, EPOCHS + start_epoch):
        epoch_start = time.perf_counter()

        if subsample_visits and n_visits > min_visits:
            n_use   = random.randint(min_visits, n_visits)
            indices = sorted(random.sample(range(n_visits), n_use))
            sub_lt  = torch.tensor(indices, dtype=torch.long, device=DEVICE)
        else:
            indices = list(range(n_visits))
            sub_lt  = lt_seq

        # Move only the selected frames to the target device to save memory
        sub_frames = [frames[i].to(DEVICE) for i in indices]

        loss = model.train_step(sub_frames, sub_lt, device=DEVICE)
        loss_history.append(loss)

        del sub_frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        epoch_elapsed = time.perf_counter() - epoch_start

        if epoch % log_every == 0 or epoch == (EPOCHS + start_epoch - 1):
            avg_recent = np.mean(loss_history[-50:])
            print(f"  Epoch {epoch:4d}/{EPOCHS + start_epoch}  "
                  f"loss={loss:.4f}  avg50={avg_recent:.4f}  "
                  f"({epoch_elapsed:.2f}s/epoch)")

        if epoch % save_every == 0:
            if loss < best_loss:
                best_loss = loss
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'best_loss':   best_loss,
                'hyperparams': {
                    'base_dim': BASE_DIM,
                    'gru_layers': GRU_LAYERS,
                    'res_blocks': RES_BLOCKS,
                    'T': T,
                    'beta_schedule': BETA_SCHEDULE,
                    'size': (H, W),
                    'sequences': sequences,
                },
            }, checkpoint_dir / f"{epoch}.pth")

except KeyboardInterrupt:
    print("\nTraining interrupted. Saving current state...")

# Always save latest regardless of how loop ended
torch.save({
    'epoch':       epoch,
    'model_state': model.state_dict(),
    'best_loss':   best_loss,
    'hyperparams': {
        'base_dim': BASE_DIM,
        'gru_layers': GRU_LAYERS,
        'res_blocks': RES_BLOCKS,
        'T': T,
        'beta_schedule': BETA_SCHEDULE,
        'size': (H, W),
        'sequences': sequences,
    },
}, checkpoint_dir / "latest.pth")

total_elapsed = time.perf_counter() - total_start
print(f"\nTraining complete in {total_elapsed:.1f}s")
print(f"Best loss: {best_loss:.4f}")

# ---------------------------------------------------------------------------
# ── Loss curve ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 3))
plt.plot(loss_history, linewidth=0.8, alpha=0.6, label='per-epoch loss')
if len(loss_history) >= 10:
    smooth = pd.Series(loss_history).rolling(10).mean()
    plt.plot(smooth, linewidth=2, label='10-epoch avg')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('RDDIM training loss — synthetic longitudinal MRI')
plt.legend()
plt.tight_layout()
loss_plot_path = output_dir / 'loss_curve.png'
plt.savefig(loss_plot_path, dpi=150)
plt.close()
print(f"Saved loss curve → {loss_plot_path}")

# ---------------------------------------------------------------------------
# ── Sampling ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
print("\nSampling after training...")

PREDICT_PATIENT_ID = "P001"
PREDICT_N_INPUT = min(5, n_visits)
PREDICT_N_PREDICT = max(0, n_visits - PREDICT_N_INPUT)

# Wrap the trained model in the generate() API format
model_bundle = {
    "model":     model,
    "sequences": sequences,
    "img_size":  (H, W),
    "T":         T,
    "device":    DEVICE,
}

results = generate(
    model=model_bundle,
    data_path=data_path,
    patient_id=PREDICT_PATIENT_ID,
    n_input=PREDICT_N_INPUT,
    n_predict=PREDICT_N_PREDICT,
    out_dir=str(output_dir),
)

print("\nGenerated files:")
for seq, paths in results.items():
    print(f"  {seq}: {len(paths)} frames")
    for p in paths:
        print(f"    {p}")

# ---------------------------------------------------------------------------
# ── GT vs Generated Comparison Table ──────────────────────────────────────
# ---------------------------------------------------------------------------
patient_sessions = session_table[
    session_table[sequences[0]].astype(str).str.contains(PREDICT_PATIENT_ID, na=False)
].sort_values("session_date").reset_index(drop=True)

comparison_csv = output_dir / "comparison_table.csv"
comparison_summary_csv = output_dir / "comparison_summary.csv"

cmp_df, summary_df = save_comparison_table(
    generated_results=results,
    patient_sessions=patient_sessions,
    data_root=data_root,
    seqs=sequences,
    hw=(H, W),
    n_input=PREDICT_N_INPUT,
    out_csv=comparison_csv,
    out_summary_csv=comparison_summary_csv,
)

print(f"\nSaved comparison table → {comparison_csv}")
print(f"Saved comparison summary → {comparison_summary_csv}")
if not summary_df.empty:
    print("\nPer-sequence mean metrics:")
    print(summary_df.to_string(index=False))