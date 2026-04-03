from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==== CONSTANTS ====
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "training_data"
GEN_PATH = BASE_DIR / "output/predict"
OUT_DIR = BASE_DIR / "output/compare"
OUT_FILE = OUT_DIR / "all_sequences_compare.png"

PATIENT_ID = "P001"
SEQUENCES = ["FLAIR", "POST", "PRE", "T2"]
START_VISIT = 5
N_VISITS = 8
# ==== END CONSTANTS ====


def _as_gray_float01(path: Path) -> np.ndarray:
	"""Load image as 2D grayscale float array in [0, 1]."""
	img = plt.imread(path)
	if img.ndim == 3:
		img = img[..., 0]
	img = img.astype(np.float32)
	if img.max() > 1.0:
		img /= 255.0
	return img


def _load_patient_rows(data_path: Path, patient_id: str) -> pd.DataFrame:
	csv_path = data_path / "session_table.csv"
	table = pd.read_csv(csv_path)

	# Use FLAIR path as a stable way to select patient rows.
	rows = table[table["FLAIR"].astype(str).str.contains(patient_id, na=False)]
	rows = rows.sort_values("session_date").reset_index(drop=True)

	if rows.empty:
		raise ValueError(f"No sessions found for patient_id='{patient_id}'")
	return rows


def create_all_comparisons(
	data_path: Path,
	gen_path: Path,
	out_file: Path,
	patient_id: str,
	sequences: list[str],
	start_visit: int,
	n_visits: int,
) -> Path:
	rows = _load_patient_rows(data_path, patient_id)
	last_needed = start_visit + n_visits - 1
	if last_needed >= len(rows):
		raise ValueError(
			f"Requested up to visit index {last_needed}, but only {len(rows)} visits exist for {patient_id}."
		)

	n_rows = len(sequences) * 2
	fig, axes = plt.subplots(n_rows, n_visits, figsize=(n_visits * 2.4, n_rows * 2.1))

	if n_visits == 1:
		axes = np.expand_dims(axes, axis=1)

	for s_idx, sequence in enumerate(sequences):
		gt_row_idx = s_idx * 2
		gen_row_idx = gt_row_idx + 1

		for i, visit_idx in enumerate(range(start_visit, start_visit + n_visits)):
			gt_rel = rows.iloc[visit_idx][sequence]
			gt_path = data_path / gt_rel
			gen_file = gen_path / f"{sequence}_v{visit_idx}.png"

			if not gt_path.exists():
				raise FileNotFoundError(f"Ground-truth image not found: {gt_path}")
			if not gen_file.exists():
				raise FileNotFoundError(f"Generated image not found: {gen_file}")

			gt_img = _as_gray_float01(gt_path)
			gen_img = _as_gray_float01(gen_file)

			axes[gt_row_idx, i].imshow(gt_img, cmap="gray", vmin=0.0, vmax=1.0)
			axes[gt_row_idx, i].set_title(f"GT v{visit_idx}")
			axes[gt_row_idx, i].axis("off")

			axes[gen_row_idx, i].imshow(gen_img, cmap="gray", vmin=0.0, vmax=1.0)
			axes[gen_row_idx, i].set_title(f"Gen v{visit_idx}")
			axes[gen_row_idx, i].axis("off")

		axes[gt_row_idx, 0].set_ylabel(f"{sequence} GT", fontsize=9)
		axes[gen_row_idx, 0].set_ylabel(f"{sequence} Gen", fontsize=9)

	fig.tight_layout()
	out_file.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_file, dpi=180)
	plt.close(fig)
	return out_file


def main() -> None:
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	out = create_all_comparisons(
		data_path=DATA_PATH,
		gen_path=GEN_PATH,
		out_file=OUT_FILE,
		patient_id=PATIENT_ID,
		sequences=SEQUENCES,
		start_visit=START_VISIT,
		n_visits=N_VISITS,
	)
	print(f"Saved comparison image: {out}")


if __name__ == "__main__":
	main()

