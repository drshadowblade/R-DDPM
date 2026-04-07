"""
gen.py — Longitudinal Synthetic Brain MRI Generator
====================================================
Generates a time-series of brain MRI scans for a single patient,
simulating tumour growth then shrinkage across N_FRAMES monthly sessions.

Key design principle
--------------------
  PATIENT_SEED  — fixed for the patient; drives ALL anatomy (skull shape,
                  gyral folds, ventricle size, sulcal pattern, lesion location).
  frame_seeds   — one per timepoint; drives ONLY scanner noise and bias field.
  → The brain looks identical across frames. Only the tumour radius and
    scan noise change between sessions.
"""

import random
import zipfile
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# ==== CONSTANTS ====
N_FRAMES     = 20
PEAK_RADIUS  = 35       # px in render space (512×512)
PEAK_AT      = 13       # 1-based frame index of peak
NOISE        = 5        # base noise level
LOCATION     = "frontal"
IMG_SIZE     = 96      # final output px
PATIENT_ID   = "P001"
BASE_DATE    = "2023-01-15"
DAY_INTERVAL = 30
OUT          = "training_data/"
ZIP          = False
PATIENT_SEED = 42       # change this to get a different patient anatomy
# ==== END CONSTANTS ====

RENDER_RES = 512        # internal render res; downsampled to IMG_SIZE on output
MODALITIES = ["FLAIR", "POST", "PRE", "T2"]

# Lesion centre in render space — fixed per location
LOCS = {
    "frontal":   (256, 168),
    "parietal":  (256, 228),
    "temporal":  (165, 280),
    "occipital": (256, 340),
}

# ── Tissue intensities ────────────────────────────────────────────────────────
MOD_PARAMS = {
    "PRE": dict(
        bg=0, skull=235, diploe=215, sas=18, sulci=14, vent=14,
        gm=108, wm=172, deep_gm=92, falx=88,
        tumor=72, necrosis=48, edema=112, ce_rim=0,
    ),
    "POST": dict(
        bg=0, skull=235, diploe=215, sas=18, sulci=14, vent=14,
        gm=109, wm=173, deep_gm=93, falx=88,
        tumor=76, necrosis=52, edema=114, ce_rim=215,
    ),
    "T2": dict(
        bg=0, skull=65, diploe=90, sas=240, sulci=238, vent=240,
        gm=150, wm=118, deep_gm=138, falx=38,
        tumor=218, necrosis=228, edema=192, ce_rim=0,
    ),
    "FLAIR": dict(
        bg=0, skull=220, diploe=200, sas=8, sulci=6, vent=7,
        gm=98, wm=155, deep_gm=85, falx=80,
        tumor=208, necrosis=220, edema=178, ce_rim=0,
    ),
}

# Per-frame scanner noise seeds (vary each session; anatomy does NOT change)
_rng_seeds = random.Random(PATIENT_SEED)
frame_seeds = [_rng_seeds.randint(0, 2**31) for _ in range(N_FRAMES)]


# ── Utilities ─────────────────────────────────────────────────────────────────

def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def fbm(xs, ys, seed, octaves=6, base_scale=0.030, lacunarity=2.05, gain=0.52):
    """Fractional Brownian Motion — fast vectorised sinusoidal fBm."""
    s = seed * 0.001371
    val = np.zeros_like(xs, dtype=np.float32)
    amp, freq, norm = 1.0, 1.0, 0.0
    for _ in range(octaves):
        val += amp * (np.sin(xs * freq * base_scale + s * freq * 13.73) *
                      np.cos(ys * freq * base_scale + s * freq * 29.31))
        norm += amp
        amp  *= gain
        freq *= lacunarity
    return val / norm


def rician(arr, sigma, rng):
    r = arr + rng.normal(0.0, sigma, arr.shape).astype(np.float32)
    i =       rng.normal(0.0, sigma, arr.shape).astype(np.float32)
    return np.sqrt(r * r + i * i)


def tumor_radius(fi: int) -> float:
    """Smooth grow-then-shrink profile. Returns radius in render-space px."""
    t  = fi / max(N_FRAMES - 1, 1)
    gp = (PEAK_AT - 1) / max(N_FRAMES - 1, 1)
    r  = PEAK_RADIUS * (t / max(gp, 1e-6)) if t <= gp \
         else PEAK_RADIUS * (1 - (t - gp) / max(1 - gp, 1e-6))
    return max(r, 0.0)


# ── Anatomy builder — called ONCE per patient ─────────────────────────────────

def build_anatomy(patient_seed: int):
    """
    Build all tissue masks at RENDER_RES × RENDER_RES.
    Called once; the result is reused for every frame.

    Returns (masks_dict, cx, cy, lesion_xy, lesion_warp_fn)
      lesion_warp_fn(xs, ys) → warped distance from lesion centre
    """
    R  = RENDER_RES
    rg = make_rng(patient_seed)
    ys, xs = np.mgrid[0:R, 0:R].astype(np.float32)

    # ── Head centre + tilt ───────────────────────────────────────────────────
    cx = R * (0.500 + rg.uniform(-0.015, 0.015))
    cy = R * (0.500 + rg.uniform(-0.015, 0.015))
    tilt = rg.uniform(-5.0, 5.0) * np.pi / 180.0
    xr = (xs - cx) * np.cos(tilt) + (ys - cy) * np.sin(tilt)
    yr = -(xs - cx) * np.sin(tilt) + (ys - cy) * np.cos(tilt)

    # ── Skull: taller than wide (vertical ellipse) ───────────────────────────
    rx_sk = R * rg.uniform(0.330, 0.370)   # horizontal (narrower)
    ry_sk = R * rg.uniform(0.400, 0.450)   # vertical (taller)
    if ry_sk / rx_sk < 1.15:
        rx_sk = ry_sk / 1.15

    skull_noise = fbm(xs, ys, int(rg.integers(0, 99999)),
                      octaves=3, base_scale=0.007) * 0.020
    d_sk = np.sqrt((xr / rx_sk)**2 + (yr / ry_sk)**2) + skull_noise

    skull_ring  = (d_sk > 0.915) & (d_sk <= 1.020)
    diploe_ring = (d_sk > 0.872) & (d_sk <= 0.915)
    inside_skull = d_sk <= 0.872
    sas_mask    = (d_sk > 0.845) & (d_sk <= 0.872)

    # ── GM/WM boundary with fBm gyral folding ───────────────────────────────
    rx_br = R * rg.uniform(0.285, 0.315)   # horizontal (narrower, matching skull)
    ry_br = R * rg.uniform(0.355, 0.390)   # vertical

    fold_c = fbm(xs, ys, int(rg.integers(0, 99999)),
                 octaves=4, base_scale=0.020 + rg.uniform(0, 0.005)) * 0.092
    fold_f = fbm(xs, ys, int(rg.integers(0, 99999)),
                 octaves=5, base_scale=0.050 + rg.uniform(0, 0.008)) * 0.036
    fold   = fold_c + fold_f

    d_br = np.sqrt((xr / rx_br)**2 + (yr / ry_br)**2)
    gwb  = 1.0 + fold

    gm_band = inside_skull & (d_br >= gwb * 0.875) & (d_br < gwb)
    wm_core = inside_skull & (d_br <  gwb * 0.875)

    # ── Sulci ────────────────────────────────────────────────────────────────
    sdepth = rg.uniform(0.20, 0.28)
    thresh = np.percentile(fold_c[inside_skull], sdepth * 100)
    sulci  = gm_band & (fold_c < thresh)
    sulci |= (inside_skull & (d_br >= gwb * 0.840) & (d_br < gwb * 0.875)
              & (fold_c < thresh * 0.8))

    # ── Lateral ventricles (smaller than before) ─────────────────────────────
    vs = rg.uniform(0.55, 0.90)   # tighter range, smaller overall
    vent = np.zeros((R, R), dtype=bool)
    for side in (-1, 1):
        # Anterior horn
        vx = cx + side * R * 0.080 * vs;  vy = cy - R * 0.055 * vs
        vent |= (((xr-(vx-cx))/(R*0.048*vs))**2 + ((yr-(vy-cy))/(R*0.068*vs))**2 < 1.0) \
                & wm_core & (side*xr > R*0.014)
        # Body
        vx2 = cx + side * R * 0.070 * vs; vy2 = cy + R * 0.010 * vs
        vent |= (((xr-(vx2-cx))/(R*0.042*vs))**2 + ((yr-(vy2-cy))/(R*0.054*vs))**2 < 1.0) \
                & wm_core & (side*xr > R*0.012)
        # Posterior horn
        vx3 = cx + side * R * 0.065 * vs; vy3 = cy + R * 0.065 * vs
        vent |= (((xr-(vx3-cx))/(R*0.030*vs))**2 + ((yr-(vy3-cy))/(R*0.042*vs))**2 < 1.0) \
                & wm_core & (side*xr > R*0.010)
    # 3rd ventricle
    vent |= ((xr/(R*0.014*vs))**2 + ((yr+R*0.020*vs)/(R*0.055*vs))**2 < 1.0) & wm_core

    # ── Deep grey nuclei ─────────────────────────────────────────────────────
    deep_gm = np.zeros((R, R), dtype=bool)
    for side in (-1, 1):
        dgx = cx + side * R * 0.105;  dgy = cy + R * 0.010
        deep_gm |= (((xr-(dgx-cx))/(R*0.040))**2 + ((yr-(dgy-cy))/(R*0.052))**2 < 1.0) \
                   & wm_core & ~vent

    # ── Falx (thin — 1 px in render space) ───────────────────────────────────
    falx = (np.abs(xr) < R * 0.004) & inside_skull & ~vent

    # ── Lesion shape — domain-warped distance field, baked at anatomy time ────
    # The SHAPE is fixed; only the radius threshold changes per frame.
    lx, ly = LOCS[LOCATION]

    # Domain warp pass 1: displace sample coords
    wx = xs + fbm(xs, ys, int(rg.integers(0, 99999)), octaves=4, base_scale=0.042) \
         * (PEAK_RADIUS * 0.50)
    wy = ys + fbm(xs, ys, int(rg.integers(0, 99999)), octaves=4, base_scale=0.042) \
         * (PEAK_RADIUS * 0.50)
    # Domain warp pass 2: evaluate fBm on warped coords
    coarse = fbm(wx, wy, int(rg.integers(0, 99999)), octaves=5, base_scale=0.048) \
             * (PEAK_RADIUS * 0.32)
    fine   = fbm(wx, wy, int(rg.integers(0, 99999)), octaves=6, base_scale=0.115) \
             * (PEAK_RADIUS * 0.13)
    # Baked warped distance — we'll threshold this per-frame
    d_lesion_warped = np.sqrt((xs - lx)**2 + (ys - ly)**2) + coarse + fine

    # Separate warp for oedema halo (asymmetric vasogenic spread)
    ewx = xs + fbm(xs, ys, int(rg.integers(0, 99999)), octaves=3, base_scale=0.028) \
          * (PEAK_RADIUS * 0.28)
    ewy = ys + fbm(xs, ys, int(rg.integers(0, 99999)), octaves=3, base_scale=0.028) \
          * (PEAK_RADIUS * 0.28)
    edema_w = fbm(ewx, ewy, int(rg.integers(0, 99999)), octaves=3, base_scale=0.030) \
              * (PEAK_RADIUS * 0.20)
    d_edema_warped = np.sqrt((xs - lx)**2 + (ys - ly)**2) + edema_w

    # Smooth Euclidean distance (used for intensity gradients within masks)
    d_raw = np.sqrt((xs - lx)**2 + (ys - ly)**2)

    masks = dict(
        bg       = ~(inside_skull | skull_ring | diploe_ring),
        skull    = skull_ring,
        diploe   = diploe_ring,
        sas      = sas_mask & ~vent,
        gm       = gm_band & ~sulci & ~vent & ~deep_gm,
        wm       = wm_core & ~vent & ~deep_gm & ~falx,
        deep_gm  = deep_gm,
        sulci    = sulci,
        vent     = vent,
        falx     = falx,
        inside   = inside_skull,
    )

    return masks, cx, cy, (lx, ly), d_lesion_warped, d_edema_warped, d_raw


# ── Renderer — called once per (frame, modality) ─────────────────────────────

def render_frame(masks, d_lesion_warped, d_edema_warped, d_raw,
                 fi: int, modality: str) -> np.ndarray:
    p      = MOD_PARAMS[modality]
    R      = RENDER_RES
    rg     = make_rng(frame_seeds[fi] ^ (hash(modality) & 0xFFFFFFFF))
    rad    = tumor_radius(fi)

    ys, xs = np.mgrid[0:R, 0:R].astype(np.float32)

    # ── Base tissue fill ─────────────────────────────────────────────────────
    img = np.zeros((R, R), dtype=np.float32)
    img[masks["skull"]]   = p["skull"]
    img[masks["diploe"]]  = p["diploe"]
    img[masks["sas"]]     = p["sas"]
    img[masks["gm"]]      = p["gm"]
    img[masks["wm"]]      = p["wm"]
    img[masks["deep_gm"]] = p["deep_gm"]
    img[masks["sulci"]]   = p["sulci"]
    img[masks["vent"]]    = p["vent"]
    img[masks["falx"]]    = p["falx"]

    # ── B1 bias field (varies per frame — scanner session variation) ──────────
    bseed = int(rg.integers(0, 99999))
    brg   = make_rng(bseed)
    a = brg.uniform(0.04, 0.08);  b = brg.uniform(0.02, 0.05)
    c = brg.uniform(-0.03, 0.03)
    bias = (1.0 + a * np.cos((xs/R - 0.5)*np.pi + c)
                + b * np.cos((ys/R - 0.5)*np.pi*1.3 + c*0.7))
    img *= bias.astype(np.float32)

    # ── Anatomical texture (patient-specific seed so texture is consistent) ───
    # Use PATIENT_SEED for the texture direction so same gyral highlights appear
    # each scan; frame seed only adds the stochastic noise layer on top.
    tex = fbm(xs, ys, PATIENT_SEED ^ (hash(modality) & 0xFFFF),
              octaves=5, base_scale=0.038) * NOISE * 1.5
    wt = np.zeros((R, R), dtype=np.float32)
    wt[masks["gm"]]      = 1.00
    wt[masks["wm"]]      = 0.42
    wt[masks["deep_gm"]] = 0.80
    wt[masks["skull"]]   = 0.60
    img += tex * wt

    # ── Partial volume blurring at sulcal edges ───────────────────────────────
    pve_w = np.clip(gaussian_filter(masks["sulci"].astype(np.float32), sigma=1.6), 0, 1) * 0.38
    img   = img * (1 - pve_w) + p["sulci"] * pve_w

    # ── Tumour (radius grows/shrinks; shape is fixed via pre-baked warp) ──────
    if rad > 2.0:
        # Masks derived from baked warped distance field
        edema_r  = rad * 1.55
        em = masks["inside"] & (d_edema_warped >= rad * 0.92) & (d_edema_warped < edema_r)
        if em.any():
            ef = np.clip((edema_r - d_edema_warped[em]) / (edema_r - rad*0.92), 0,1)**0.65
            img[em] = img[em]*(1 - ef*0.55) + p["edema"]*ef*0.55

        core = masks["inside"] & (d_lesion_warped < rad * 0.88)
        if core.any():
            tf = np.clip(1 - d_raw[core] / (rad*0.88), 0, 1)
            img[core] = p["tumor"] + p["tumor"]*0.08*tf

        if rad > PEAK_RADIUS * 0.45:   # necrosis only at larger sizes
            nr = rad * 0.38
            nm = masks["inside"] & (d_lesion_warped < nr)
            if nm.any():
                nf = np.clip(1 - d_raw[nm] / nr, 0, 1)
                img[nm] = img[nm]*(1-nf) + p["necrosis"]*nf

        rim = masks["inside"] & (d_lesion_warped >= rad*0.88) & (d_lesion_warped < rad)
        if rim.any():
            rf = np.clip((d_raw[rim] - rad*0.88) / (rad*0.12), 0, 1)
            img[rim] = p["tumor"]*(1-rf) + p["edema"]*rf*0.4

        if modality == "POST" and p["ce_rim"] > 0:
            ce = (rim | core) & (d_raw < rad*1.06)
            if ce.any():
                sig = rad * 0.11
                gau = np.exp(-((d_raw[ce] - rad*0.87)**2) / (2*sig**2))
                img[ce] = np.maximum(img[ce], p["ce_rim"]*gau)

    # ── Rician acquisition noise (frame-specific) ─────────────────────────────
    sigma = NOISE * 0.52
    img   = rician(img, sigma, rg)

    # ── Background noise floor ────────────────────────────────────────────────
    img[masks["bg"]] = np.abs(rg.normal(0, sigma*0.35, (R,R))[masks["bg"]])

    return np.clip(img, 0, 255).astype(np.uint8)


# ── Output helpers ────────────────────────────────────────────────────────────

def make_png_bytes(arr: np.ndarray, out_size: int) -> bytes:
    img = Image.fromarray(arr, mode="L")
    if out_size != RENDER_RES:
        img = img.resize((out_size, out_size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def session_dates():
    base = date.fromisoformat(BASE_DATE)
    return [(base + timedelta(days=i * DAY_INTERVAL)).isoformat()
            for i in range(N_FRAMES)]


def generate(out_root: Path, zip_buf=None):
    # ── Build anatomy ONCE for this patient ───────────────────────────────────
    print(f"  Building anatomy (seed={PATIENT_SEED})...", end=" ", flush=True)
    masks, cx, cy, lesion_xy, d_les_w, d_edema_w, d_raw = build_anatomy(PATIENT_SEED)
    print("done")

    dates    = session_dates()
    csv_rows = [["session_date"] + MODALITIES]

    for fi, d in enumerate(dates):
        r = tumor_radius(fi)
        phase = ("clear"    if r < 2 else
                 "growing"  if fi / max(N_FRAMES-1,1) < (PEAK_AT-1)/max(N_FRAMES-1,1) else
                 "peak"     if fi == PEAK_AT-1 else
                 "shrinking")
        print(f"  T{fi+1:02d}  date={d}  r={r:5.1f}px  phase={phase}")

        row = [d]
        for mod in MODALITIES:
            arr = render_frame(masks, d_les_w, d_edema_w, d_raw, fi, mod)
            png = make_png_bytes(arr, IMG_SIZE)
            rel = f"images/{PATIENT_ID}/{d}/{mod}_T{fi+1:02d}.png"
            row.append(rel)
            if zip_buf:
                zip_buf.writestr(f"{OUT}{rel}", png)
            else:
                dest = out_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(png)
        csv_rows.append(row)

    csv_txt = "\n".join(",".join(r) for r in csv_rows) + "\n"
    if zip_buf:
        zip_buf.writestr(f"{OUT}session_table.csv", csv_txt)
    else:
        (out_root / "session_table.csv").write_text(csv_txt)
    return csv_rows


def main():
    out_root = Path(OUT)
    print("Generating longitudinal synthetic MRI dataset")
    print(f"  Patient    : {PATIENT_ID}  (seed={PATIENT_SEED})")
    print(f"  Timepoints : {N_FRAMES}  |  Peak T{PEAK_AT}  |  Location: {LOCATION}")
    print(f"  Output     : {out_root.resolve()}")
    print()
    out_root.mkdir(parents=True, exist_ok=True)
    generate(out_root)
    print(f"\nDone — {N_FRAMES * len(MODALITIES)} PNGs in {out_root.resolve()}")
    print(f"session_table.csv → {(out_root / 'session_table.csv').resolve()}")


if __name__ == "__main__":
    main()
