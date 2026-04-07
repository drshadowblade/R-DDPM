from utils import load_model, generate
from pathlib import Path

# ==== CONSTANTS ====
BASE_DIR   = Path(__file__).resolve().parent
CHECKPOINT = BASE_DIR / 'checkpoints/latest (1).pth'
DATA_PATH  = BASE_DIR / 'training_data'
PATIENT_ID = 'P001'
N_INPUT    = 5
N_PREDICT  = 15
INTERVAL   = 1.0
OUT_DIR    = BASE_DIR / 'output/predict'
# ==== END CONSTANTS ====

def main() -> None:
    model = load_model(str(CHECKPOINT))

    results = generate(
        model=model,
        data_path=str(DATA_PATH),
        patient_id=PATIENT_ID,
        n_input=N_INPUT,
        n_predict=N_PREDICT,
        interval_months=INTERVAL,
        out_dir=str(OUT_DIR),
    )

    print("\nGenerated files:")
    for seq, paths in results.items():
        print(f"  {seq}: {len(paths)} frames")
        for p in paths:
            print(f"    {p}")


if __name__ == "__main__":
    main()