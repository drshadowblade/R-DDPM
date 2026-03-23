
import argparse
import csv
import math
import random
import zipfile
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

# ==== CONSTANTS (from args) ====
N_FRAMES = 20
PEAK_RADIUS = 30
PEAK_AT = 13
NOISE = 8
LOCATION = "frontal"
IMG_SIZE = 128
PATIENT_ID = "P001"
BASE_DATE = "2023-01-15"
DAY_INTERVAL = 30
OUT = "training_data/"
ZIP = False
SEED = None

# ==== END CONSTANTS ====

FW, FH = 120, 120
MODALITIES = ["FLAIR", "POST", "PRE", "T2"]

# Tumour centre coords at native 120×120
LOCS = {
    "frontal":   (60, 38),
    "parietal":  (60, 52),
    "temporal":  (38, 65),
    "occipital": (60, 82),
}

MOD_PARAMS = {
    #           bg    skull  brain  tumor  edema
    "T1":    dict(bg=22,  skull=200, brain=110, tumor=75,  edema=0),
    "T2":    dict(bg=28,  skull=60,  brain=140, tumor=210, edema=180),
    "FLAIR": dict(bg=18,  skull=55,  brain=80,  tumor=200, edema=160),
}

MOD_MAP = {
    "FLAIR": "FLAIR",
    "POST":  "T1",
    "PRE":   "T1",      
    "T2":    "T2",
}

frame_seeds = [random.randint(0, 99999) for _ in range(N_FRAMES)]


def tumor_radius(fi: int) -> float:
    """Grow-then-shrink profile, matching tumorRadius() in the HTML."""
    n     = N_FRAMES
    t     = fi / max(n - 1, 1)
    gp    = (PEAK_AT - 1) / max(n - 1, 1)
    if t <= gp:
        r = PEAK_RADIUS * (t / max(gp, 1e-4))
    else:
        r = PEAK_RADIUS * (1.0 - (t - gp) / max(1.0 - gp, 1e-4))
    return max(r, 0.0)


def seeded_noise(x: np.ndarray, y: np.ndarray, seed: int, scale: float = 0.08) -> np.ndarray:
    """
    Vectorised port of seededNoise() — 4-octave sinusoidal noise.
    x, y are 2-D arrays of pixel coordinates.
    """
    s = seed * 0.0013
    v = np.zeros_like(x, dtype=np.float32)
    f = 1.0
    while f <= 8.0:
        v += np.sin(x * f * scale + s * f * 17.3) * np.cos(y * f * scale + s * f * 31.7) / f
        f *= 2.0
    return v


def render_frame(fi: int, modality_name: str) -> np.ndarray:
    """
    Render one (fi, modality) pair at FW×FH, return uint8 array (FH, FW).
    Matches drawFrame() + drawFrameMod() from the HTML.
    """
    render_mode = MOD_MAP[modality_name]
    p           = MOD_PARAMS[render_mode]
    seed        = frame_seeds[fi]
    r           = tumor_radius(fi)
    tx, ty      = LOCS[LOCATION]

    # Pixel coordinate grids
    ys, xs = np.mgrid[0:FH, 0:FW].astype(np.float32)

    # ── Background fill ──
    img = np.full((FH, FW), p["bg"], dtype=np.float32)

    # ── Skull ellipse mask ──
    cx, cy, srx, sry = 60.0, 58.0, 42.0, 48.0
    dx = (xs - cx) / srx
    dy = (ys - cy) / sry
    d2 = dx * dx + dy * dy

    skull_mask = (d2 > 0.88) & (d2 <= 1.05)
    brain_mask = d2 <= 0.88

    noise = seeded_noise(xs, ys, seed) * NOISE

    # Skull ring
    img[skull_mask] = p["skull"] + noise[skull_mask]

    # Brain parenchyma
    img[brain_mask] = p["brain"] + noise[brain_mask] * 0.6

    # Tumour + oedema (only if radius is meaningful)
    if r > 2:
        dist = np.sqrt((xs - tx) ** 2 + (ys - ty) ** 2)

        # Oedema halo
        edema_r = r * 1.45
        if p["edema"] > 0:
            edema_mask = brain_mask & (dist < edema_r)
            ef = 1.0 - dist[edema_mask] / edema_r
            img[edema_mask] += (p["edema"] - p["brain"]) * ef * 0.5 + noise[edema_mask]

        # Tumour core
        core_mask = brain_mask & (dist < r)
        tf = 1.0 - dist[core_mask] / r
        core_lum = p["tumor"] - 20 * tf if render_mode == "T1" else p["tumor"]
        img[core_mask] += (core_lum - p["brain"]) * tf

    # Contrast-enhancement overlay for POST (bright rim, matches HTML)
    if modality_name == "POST" and r > 3:
        dist = np.sqrt((xs - tx) ** 2 + (ys - ty) ** 2)
        ce_mask = brain_mask & (dist < r)
        tf = 1.0 - dist[ce_mask] / r
        img[ce_mask] = np.minimum(255, img[ce_mask] + 80 * tf)

    return np.clip(img, 0, 255).astype(np.uint8)


def make_png_bytes(arr: np.ndarray, out_hw: tuple) -> bytes:
    """Resize to out_hw (H, W) and return PNG bytes."""
    img = Image.fromarray(arr, mode="L")
    if out_hw != (FH, FW):
        img = img.resize((out_hw[1], out_hw[0]), Image.BILINEAR)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()



def session_dates() -> list[str]:
    base = date.fromisoformat(BASE_DATE)
    return [
        (base + timedelta(days=i * DAY_INTERVAL)).isoformat()
        for i in range(N_FRAMES)
    ]



def generate(out_root: Path, zip_buf: zipfile.ZipFile | None = None):
    out_hw   = (IMG_SIZE, IMG_SIZE)
    pat_id   = PATIENT_ID
    dates    = session_dates()
    csv_rows = [["session_date", "FLAIR", "POST", "PRE", "T2"]]

    for fi, d in enumerate(dates):
        r = tumor_radius(fi)
        phase = (
            "clear"     if r < 2 else
            "growing"   if fi / max(N_FRAMES - 1, 1) < (PEAK_AT - 1) / max(N_FRAMES - 1, 1) else
            "peak"      if fi == PEAK_AT - 1 else
            "shrinking"
        )
        print(f"  T{fi+1:02d}  date={d}  r={r:5.1f}px  phase={phase}")

        row = [d]
        for mod in MODALITIES:
            arr      = render_frame(fi, mod)
            png      = make_png_bytes(arr, out_hw)
            rel_path = f"images/{pat_id}/{d}/{mod}_T{fi+1:02d}.png"
            row.append(rel_path)

            if zip_buf is not None:
                zip_buf.writestr(f"training_data/{rel_path}", png)
            else:
                dest = out_root / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(png)

        csv_rows.append(row)

    # Write session_table.csv
    csv_content = "\n".join(",".join(r) for r in csv_rows) + "\n"
    if zip_buf is not None:
        zip_buf.writestr("training_data/session_table.csv", csv_content)
    else:
        csv_path = out_root / "session_table.csv"
        csv_path.write_text(csv_content)

    return csv_rows

def main():
    out_root = Path(OUT)
    print(f"Generating synthetic longitudinal MRI dataset")
    print(f"  Timepoints   : {N_FRAMES}")
    print(f"  Peak radius  : {PEAK_RADIUS}px  (at T{PEAK_AT})")
    print(f"  Location     : {LOCATION}")
    print(f"  Noise        : {NOISE}")
    print(f"  Image size   : {IMG_SIZE}×{IMG_SIZE}")
    print(f"  Patient ID   : {PATIENT_ID}")
    print(f"  Output       : {out_root.resolve()}")
    print()

    if ZIP:
        zip_path = Path("training_data.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            generate(out_root, zip_buf=zf)
        print(f"\nWrote {zip_path.resolve()}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(".")
        print(f"Extracted to {out_root.resolve()}")
    else:
        out_root.mkdir(parents=True, exist_ok=True)
        generate(out_root)
        print(f"\nDone — {N_FRAMES * len(MODALITIES)} PNGs written to {out_root.resolve()}")

    print(f"session_table.csv → {(out_root / 'session_table.csv').resolve()}")


if __name__ == "__main__":
    main()