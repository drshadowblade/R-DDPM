import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from PIL import Image as PILImage

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
    arr = np.stack(frames)
    arr_max = arr.max()
    if arr_max <= 1.0:
        arr = (arr * 2.0) - 1.0
    else:
        arr = (arr / 127.5) - 1.0
    tensor = torch.from_numpy(arr).unsqueeze(0).float()
    return tensor

# ---------------------------------------------------------------------------
# Internal helpers — backend guy doesn't need to touch these
# ---------------------------------------------------------------------------

def _load_png_as_tensor(img_path: Path, target_hw: tuple) -> "torch.Tensor":
    """Load a single PNG → (1, H, W) float tensor in [-1, 1]."""
    img = plt.imread(str(img_path))
    if img.ndim == 3:
        img = img[..., 0]  # RGBA → grayscale
    img_resized = np.array(
        PILImage.fromarray(img).resize(
            (target_hw[1], target_hw[0]), resample=PILImage.BILINEAR
        )
    )
    arr = img_resized.astype(np.float32)
    # Normalise to [-1, 1] regardless of whether input is [0,1] or [0,255]
    if arr.max() <= 1.0:
        arr = arr * 2.0 - 1.0
    else:
        arr = (arr / 127.5) - 1.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


def _load_patient_frames(
    data_path: str,
    patient_id: str,
    sequences: list,
    hw: tuple,
) -> list:
    """
    Load all session frames for a patient.
    Returns a list of tensors, each shape (1, C, H, W).
    """
    csv_path = Path(data_path) / "session_table.csv"
    session_table = pd.read_csv(csv_path)
    session_table = session_table[
        session_table[sequences[0]].astype(str).str.contains(patient_id, na=False)
    ]
    session_table = session_table.sort_values("session_date").reset_index(drop=True)

    if session_table.empty:
        raise ValueError(
            f"No sessions found for patient_id='{patient_id}' in {csv_path}"
        )

    data_root = Path(data_path)
    frames = []
    for _, row in session_table.iterrows():
        channels = [_load_png_as_tensor(data_root / row[seq], hw) for seq in sequences]
        frame = torch.cat(channels, dim=0).unsqueeze(0)  # (1, C, H, W)
        frames.append(frame)
    return frames


def _tensor_to_png(tensor: "torch.Tensor", path: Path) -> str:
    """Save a single (H, W) tensor in [-1, 1] as a PNG. Returns path string."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.cpu().float().numpy()
    arr_uint8 = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    PILImage.fromarray(arr_uint8, mode="L").save(str(path))
    return str(path)


def _torch_load_compat(path: Path, map_location: str):
    """Load checkpoint across torch versions with safe compatibility defaults."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older torch versions do not support weights_only.
        return torch.load(path, map_location=map_location)


def _checkpoint_fallback_order(requested: Path) -> list[Path]:
    """Return a deterministic list of checkpoint candidates to try."""
    out = [requested]
    ckpt_dir = requested.parent
    if not ckpt_dir.exists():
        return out

    latest = ckpt_dir / "latest.pth"
    if latest.exists() and latest not in out:
        out.append(latest)

    numeric = []
    other = []
    for p in ckpt_dir.glob("*.pth"):
        if p in out:
            continue
        if p.stem.isdigit():
            numeric.append(p)
        else:
            other.append(p)

    numeric.sort(key=lambda p: int(p.stem), reverse=True)
    other.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out.extend(numeric)
    out.extend(other)
    return out


# ---------------------------------------------------------------------------
# Public API — two functions the backend guy needs
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "auto") -> dict:
    """
    Load the R-DDPM model from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pth checkpoint file (e.g. "checkpoints/latest.pth").
    device : str
        "auto" picks CUDA if available, else CPU.
        Or pass "cuda" / "cpu" explicitly.

    Returns
    -------
    dict with keys:
        "model"      — the loaded model (backend doesn't need to touch this)
        "sequences"  — list of modality names, e.g. ["FLAIR", "POST", "PRE", "T2"]
        "img_size"   — (H, W) tuple, e.g. (96, 96)
        "T"          — diffusion steps the model was trained with
        "device"     — device string being used
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    req_path = Path(checkpoint_path)
    if not req_path.is_absolute():
        req_path = (Path.cwd() / req_path).resolve()

    if not req_path.exists():
        alt = (Path(__file__).resolve().parent / checkpoint_path).resolve()
        if alt.exists():
            req_path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = None
    used_checkpoint = None
    load_errors = []
    for cand in _checkpoint_fallback_order(req_path):
        try:
            maybe = _torch_load_compat(cand, map_location=device)
            if isinstance(maybe, dict) and "model_state" in maybe and "hyperparams" in maybe:
                ckpt = maybe
                used_checkpoint = cand
                break
            load_errors.append(f"{cand}: missing required keys")
        except Exception as e:
            load_errors.append(f"{cand}: {type(e).__name__}: {e}")

    if ckpt is None:
        details = "\n  - " + "\n  - ".join(load_errors[:5])
        raise RuntimeError(
            f"Unable to load any valid checkpoint starting from '{req_path}'."
            f" Tried fallback candidates with errors:{details}"
        )

    if used_checkpoint != req_path:
        print(f"Warning: failed to load requested checkpoint; using fallback '{used_checkpoint}'.")

    hp = ckpt["hyperparams"]
    from RDDPM import RDDPM

    img_size = hp["size"]
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    else:
        img_size = tuple(img_size)

    T = int(hp.get("T", 1000))

    model = RDDPM(
        input_size=img_size,
        n_channels=len(hp["sequences"]),
        base_dim=hp["base_dim"],
        gru_n_layers=hp["gru_layers"],
        n_res_blocks=hp["res_blocks"],
        T=T,
        beta_schedule=hp["beta_schedule"],
    ).to(device)

    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError:
        # Backward-compat path for checkpoints saved before architecture updates.
        incompat = model.load_state_dict(ckpt["model_state"], strict=False)
        if hasattr(model, "model") and hasattr(model.model, "use_compat_mode"):
            model.model.use_compat_mode = True
        if getattr(incompat, "missing_keys", None):
            print(
                f"Warning: non-strict checkpoint load with {len(incompat.missing_keys)} missing keys. "
                "Enabled legacy compatibility mode."
            )
    model.eval()

    return {
        "model": model,
        "sequences": hp["sequences"],
        "img_size": img_size,
        "T": T,
        "device": device,
        "checkpoint_path": str(used_checkpoint),
    }


def generate(
    model: dict,
    data_path: str,
    patient_id: str,
    n_input: int = 5,
    n_predict: int = 10,
    out_dir: str = "output/",
) -> dict:
    """
    Generate future MRI visits for a patient.

    Parameters
    ----------
    model : dict
        The dict returned by load_model().
    data_path : str
        Path to the training_data/ folder containing session_table.csv.
    patient_id : str
        Patient ID to generate for, e.g. "P001".
    n_input : int
        Number of real known visits to warm up the GRU with before generating.
        Recommended: 5. Use 0 for a cold start (lower quality).
    n_predict : int
        Number of future visits to generate after the warmup visits.
    out_dir : str
        Folder where generated PNG files will be saved.

    Returns
    -------
    dict mapping modality name → list of PNG file path strings.
    Example:
    {
        "FLAIR": ["output/FLAIR_v5.png", "output/FLAIR_v6.png", ...],
        "POST":  ["output/POST_v5.png",  ...],
        "PRE":   ["output/PRE_v5.png",   ...],
        "T2":    ["output/T2_v5.png",    ...],
    }
    """
    net        = model["model"]
    sequences  = model["sequences"]
    img_size   = model["img_size"]
    T          = model["T"]
    device     = model["device"]

    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Load patient frames ──────────────────────────────────────────────────
    all_frames = _load_patient_frames(data_path, patient_id, sequences, img_size)
    n_available = len(all_frames)
    n_input = min(n_input, n_available)

    # Warmup frames (real visits fed to GRU)
    pre_images = [f.to(device) for f in all_frames[:n_input]]
    pre_times  = list(range(n_input))

    # Visit indices to generate
    start_visit = n_input
    end_visit   = n_input + n_predict
    lt_seq      = torch.arange(start_visit, end_visit, dtype=torch.long)

    if isinstance(img_size, int):
        H = W = img_size
    else:
        H, W = img_size
    C    = len(sequences)

    # ── Run generation ───────────────────────────────────────────────────────
    with torch.no_grad():
        generated = net.sample(
            shape=(1, C, H, W),
            T=T,
            lt_seq=lt_seq,
            pre_images=pre_images if n_input > 0 else None,
            pre_times=pre_times  if n_input > 0 else None,
            device=device,
        )

    # ── Save PNGs and build output dict ─────────────────────────────────────
    # results[modality] = [path_v0, path_v1, ...]
    results = {seq: [] for seq in sequences}

    for visit_offset, frame_tensor in enumerate(generated):
        visit_idx = start_visit + visit_offset
        for c_idx, seq_name in enumerate(sequences):
            channel = frame_tensor[0, c_idx]  # (H, W)
            fname   = out_path / f"{seq_name}_v{visit_idx}.png"
            path    = _tensor_to_png(channel, fname)
            results[seq_name].append(path)

    return results